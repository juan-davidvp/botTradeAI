"""
core/signal_generator.py
Fase 3 — Generador de Señales.

Flujo actualizado con capa Whale Capital:
  1. Capa Whale (opcional): RS filter → pattern detector → WhaleSignal
  2. Capa HMM (fallback)  : RegimeState + MACD + Donchian → Signal

La capa Whale tiene prioridad cuando está habilitada:
  - Descarta rezagados (RS < umbral) ANTES de calcular señales técnicas.
  - Prioriza Ballenera sobre Corto Plazo.
  - Convierte WhaleSignal al formato Signal estándar para el RiskManager.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import (
    Direction,
    Signal,
    StrategyOrchestrator,
    TechnicalConfirmation,
    LEVERAGE,
)
from core.ballenero_strategy import BalleneroOrchestrator, WhaleSignal
from core.pattern_detector import PatternSignal

logger = logging.getLogger(__name__)

MIN_BARS_REQUIRED = 60   # mínimo para EMA50, MACD y Donchian 55d
MIN_BARS_WHALE    = 252  # mínimo para cálculo de RS (ventana anual)


class SignalGenerator:
    """
    Genera señales de trading combinando el régimen HMM con los filtros
    técnicos (MACD + Donchian) y, opcionalmente, la capa Whale Capital.

    Flujo por instrumento (cada 30 segundos en el bucle principal):
      1. [Whale] RS filter: descarta rezagados vs índice de referencia.
      2. [Whale] Pattern detector: Cup&Handle, Double Bottom, Darvas, etc.
      3. [HMM fallback] StrategyOrchestrator con MACD + Donchian.
      4. Quality check.
      5. Signal → RiskManager.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        equity: float,
        whale_settings: Optional[Dict] = None,
    ):
        self.hmm_engine   = hmm_engine
        self.orchestrator = StrategyOrchestrator(equity=equity)
        self.equity       = equity

        # Capa Whale Capital (opcional — activada por settings.yaml whale.enabled)
        ws = whale_settings or {}
        self._whale_enabled = bool(ws.get("enabled", False))
        if self._whale_enabled:
            self._whale = BalleneroOrchestrator(
                equity=equity,
                initial_equity=equity,
                rs_window=int(ws.get("rs_window", 252)),
                rs_leader_threshold=float(ws.get("rs_leader_threshold", 1.15)),
                rs_laggard_threshold=float(ws.get("rs_laggard_threshold", 0.85)),
            )
            self._whale_min_confidence = float(ws.get("min_confidence", 0.45))
            logger.info("[SignalGen] Capa Whale Capital ACTIVADA")
        else:
            self._whale = None
            self._whale_min_confidence = 0.45

    # ─────────────────────────────────────────────────────────────────
    # GENERACIÓN POR INSTRUMENTO
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: Optional[RegimeState] = None,
        index_bars: Optional[pd.DataFrame] = None,
        n_open_positions: int = 0,
        available_cash: float = 0.0,
    ) -> Optional[Signal]:
        """
        Genera una señal para un instrumento dado.

        Parámetros
        ----------
        symbol           : eToro Instrument ID (como string).
        bars             : OHLCV histórico. Mínimo MIN_BARS_REQUIRED filas.
        regime_state     : Estado de régimen HMM precalculado.
        index_bars       : OHLCV del índice de referencia (SPY) para RS.
        n_open_positions : Posiciones actualmente abiertas (para límite Ballenera).
        available_cash   : Crédito disponible en eToro.
        """
        if len(bars) < MIN_BARS_REQUIRED:
            logger.warning(
                "[SignalGen] %s: %d barras disponibles (mínimo %d) — sin señal",
                symbol, len(bars), MIN_BARS_REQUIRED,
            )
            return None

        if regime_state is None:
            logger.warning("[SignalGen] %s: no se recibió RegimeState — sin señal", symbol)
            return None

        if not self.hmm_engine.model:
            logger.error("[SignalGen] HMM no entrenado — sin señal")
            return None

        # ── Capa Whale Capital (prioridad) ───────────────────────────────────
        if (self._whale is not None
                and index_bars is not None
                and len(bars) >= MIN_BARS_WHALE
                and len(index_bars) >= MIN_BARS_WHALE):

            whale_sig = self._whale.evaluate(
                symbol=symbol,
                asset_bars=bars,
                index_bars=index_bars,
                hmm_regime=regime_state.label,
                n_open_positions=n_open_positions,
                available_cash=available_cash,
            )

            if whale_sig and whale_sig.pattern.confidence >= self._whale_min_confidence:
                return self._whale_to_signal(whale_sig, regime_state)

        # ── Capa HMM + MACD + Donchian (fallback) ───────────────────────────
        strategy = self.orchestrator.get_strategy(regime_state, self.hmm_engine.regime_infos)
        signal   = strategy.generate_signal(symbol, bars, regime_state)

        if signal is None:
            return None

        if not self._quality_check(signal):
            return None

        return signal

    # ─────────────────────────────────────────────────────────────────
    # GENERACIÓN PARA TODOS LOS INSTRUMENTOS
    # ─────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
        index_bars: Optional[pd.DataFrame] = None,
        portfolio_state=None,
    ) -> List[Signal]:
        """
        Genera señales para todos los instrumentos activos.

        Parámetros
        ----------
        bars_by_symbol : {symbol: DataFrame}
        regime_state   : RegimeState compartido.
        index_bars     : OHLCV del índice de referencia (SPY) — para RS Whale.
        portfolio_state: PortfolioState — para n_open_positions y available_cash.
        """
        n_open  = len(portfolio_state.positions) if portfolio_state else 0
        cash    = float(portfolio_state.cash)    if portfolio_state else self.equity

        signals: List[Signal] = []
        for symbol, bars in bars_by_symbol.items():
            signal = self.generate(
                symbol=symbol,
                bars=bars,
                regime_state=regime_state,
                index_bars=index_bars,
                n_open_positions=n_open,
                available_cash=cash,
            )
            if signal:
                signals.append(signal)

        # Ordenar: Whale signals (mayor confianza) primero
        signals.sort(key=lambda s: (
            0 if "BALLENERA" in s.strategy_name else
            1 if "CORTO_PLAZO" in s.strategy_name else 2,
            -s.confidence,
        ))

        logger.info(
            "[SignalGen] Régimen %s — %d/%d señales válidas",
            regime_state.label, len(signals), len(bars_by_symbol),
        )
        return signals

    # ─────────────────────────────────────────────────────────────────
    # CONVERSIÓN WhaleSignal → Signal
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _whale_to_signal(ws: WhaleSignal, regime_state: RegimeState) -> Signal:
        """
        Convierte una WhaleSignal al formato Signal estándar que espera el RiskManager.

        El RiskManager aplica sus 16 checks sobre el Signal resultante,
        incluyendo el cap de 20% y el mínimo de $100.
        """
        reasoning = (
            f"Whale {ws.strategy} | {ws.pattern.pattern_name} | "
            f"RS={ws.rs_score:.3f} | RS_corr={ws.rs_correction:.3f} | "
            f"vol_ok={ws.pattern.volume_confirmed} | "
            f"compound={ws.compounding_factor:.2f}x"
        )

        return Signal(
            symbol=ws.symbol,
            direction=Direction.LONG,
            confidence=ws.pattern.confidence,
            entry_price=ws.entry_price,
            stop_loss=ws.stop_loss,
            take_profit=None,
            position_size_pct=ws.position_size_pct,
            position_size_usd=ws.position_size_usd,
            leverage=LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.utcnow(),
            reasoning=reasoning,
            strategy_name=f"WHALE_{ws.strategy}_{ws.pattern.pattern_name}",
            technical_confirmation=(
                TechnicalConfirmation.STRONG if ws.pattern.volume_confirmed
                else TechnicalConfirmation.MODERATE
            ),
            metadata={
                "whale_strategy":    ws.strategy,
                "pattern":           ws.pattern.pattern_name,
                "rs_score":          ws.rs_score,
                "rs_correction":     ws.rs_correction,
                "is_market_leader":  ws.is_market_leader,
                "pattern_depth_pct": ws.pattern.pattern_depth_pct,
                "compounding_factor": ws.compounding_factor,
                "days_in_formation": ws.pattern.days_in_formation,
            },
        )

    # ─────────────────────────────────────────────────────────────────
    # CALIDAD Y EQUITY
    # ─────────────────────────────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        """Actualiza equity cada tick — interés compuesto activo en sizing."""
        self.equity = equity
        self.orchestrator.equity = equity
        if self._whale is not None:
            self._whale.update_equity(equity)

    def _quality_check(self, signal: Signal) -> bool:
        """
        Filtros de calidad para señales HMM (la capa Whale tiene sus propios filtros).
        """
        if signal.direction == Direction.FLAT:
            logger.info("[SignalGen] %s: FLAT — descartada", signal.symbol)
            return False

        if signal.confidence < 0.30:
            logger.info(
                "[SignalGen] %s: confianza %.2f < 0.30 — descartada",
                signal.symbol, signal.confidence,
            )
            return False

        if signal.position_size_usd < 100.0:
            logger.info(
                "[SignalGen] %s: tamaño $%.2f < $100 mínimo — descartada",
                signal.symbol, signal.position_size_usd,
            )
            return False

        if (signal.technical_confirmation == TechnicalConfirmation.NONE
                and "DEFENSIV" in signal.strategy_name.upper()):
            logger.info(
                "[SignalGen] %s: alta vol sin confirmación técnica — descartada",
                signal.symbol,
            )
            return False

        return True
