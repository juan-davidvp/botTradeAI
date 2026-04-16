"""
core/signal_generator.py
Fase 3 — Generador de Señales.

Orquesta el flujo: RegimeState + OHLCV → StrategyOrchestrator → Signal.
Aplica filtros de calidad antes de emitir la señal al RiskManager.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import (
    Direction,
    Signal,
    StrategyOrchestrator,
    TechnicalConfirmation,
)

logger = logging.getLogger(__name__)

MIN_BARS_REQUIRED = 60   # mínimo para calcular EMA50, MACD y Donchian 55d


class SignalGenerator:
    """
    Genera señales de trading combinando el régimen HMM con los filtros
    técnicos (MACD + Donchian) definidos en las estrategias de régimen.

    Flujo por instrumento (cada 30 segundos en el bucle principal):
      1. Recibir RegimeState actual del HMMEngine.
      2. Seleccionar estrategia via StrategyOrchestrator.
      3. Generar signal.
      4. Filtrar por calidad (confianza, tamaño mínimo, modo incertidumbre).
      5. Retornar Signal al RiskManager para validación final.
    """

    def __init__(self, hmm_engine: HMMEngine, equity: float):
        self.hmm_engine   = hmm_engine
        self.orchestrator = StrategyOrchestrator(equity=equity)
        self.equity       = equity

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: Optional[RegimeState] = None,
    ) -> Optional[Signal]:
        """
        Genera una señal para un instrumento dado.

        Parámetros
        ----------
        symbol : str
            eToro Instrument ID (como string).
        bars : pd.DataFrame
            OHLCV histórico. Mínimo MIN_BARS_REQUIRED filas.
        regime_state : RegimeState | None
            Estado de régimen precalculado. Si None, usa el último del HMMEngine.

        Retorna
        -------
        Signal o None si no se cumplen los criterios de calidad.
        """
        if len(bars) < MIN_BARS_REQUIRED:
            logger.warning(
                "[SignalGen] %s: solo %d barras disponibles (mínimo %d) — sin señal",
                symbol, len(bars), MIN_BARS_REQUIRED,
            )
            return None

        if regime_state is None:
            logger.warning("[SignalGen] %s: no se recibió RegimeState — sin señal", symbol)
            return None

        if not self.hmm_engine.model:
            logger.error("[SignalGen] HMM no entrenado — sin señal")
            return None

        strategy = self.orchestrator.get_strategy(regime_state, self.hmm_engine.regime_infos)
        signal   = strategy.generate_signal(symbol, bars, regime_state)

        if signal is None:
            return None

        if not self._quality_check(signal):
            return None

        return signal

    def generate_all(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
    ) -> List[Signal]:
        """
        Genera señales para todos los instrumentos activos.

        Parámetros
        ----------
        bars_by_symbol : dict {symbol: DataFrame}
        regime_state   : RegimeState compartido para todos los instrumentos.

        Retorna lista de Signal aprobadas por quality_check.
        """
        signals: List[Signal] = []
        for symbol, bars in bars_by_symbol.items():
            signal = self.generate(symbol, bars, regime_state)
            if signal:
                signals.append(signal)

        signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(
            "[SignalGen] Régimen %s — %d/%d instrumentos con señal válida",
            regime_state.label, len(signals), len(bars_by_symbol),
        )
        return signals

    def _quality_check(self, signal: Signal) -> bool:
        """
        Filtros de calidad antes de emitir la señal.

        Rechaza si:
          - Confianza < 0.30
          - position_size_usd < mínimo eToro ($100)
          - Dirección FLAT (el HMM decidió no operar)
          - Confirmación técnica NONE en régimen de alta vol
        """
        if signal.direction == Direction.FLAT:
            logger.info("[SignalGen] %s: señal FLAT — descartada", signal.symbol)
            return False

        if signal.confidence < 0.30:
            logger.info(
                "[SignalGen] %s: confianza %.2f < 0.30 — descartada", signal.symbol, signal.confidence
            )
            return False

        if signal.position_size_usd < 100.0:
            logger.info(
                "[SignalGen] %s: tamaño $%.2f < $100 mínimo eToro — descartada",
                signal.symbol, signal.position_size_usd,
            )
            return False

        if (signal.technical_confirmation == TechnicalConfirmation.NONE
                and "DEFENSIV" in signal.strategy_name.upper()):
            logger.info(
                "[SignalGen] %s: alta vol sin confirmación técnica — descartada", signal.symbol
            )
            return False

        return True
