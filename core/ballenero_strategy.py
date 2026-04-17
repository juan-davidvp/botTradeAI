"""
core/ballenero_strategy.py
Whale Capital — Orquestador de Estrategias Ballenera y Corto Plazo.

Jerarquía de decisión (prioridad descendente):
  1. Filtro de régimen HMM     — bloquea CRASH / STRONG_BEAR
  2. Filtro de tendencia       — activo + mercado en tendencia alcista
  3. Fuerza Relativa (RS)      — descarta rezagados vs índice de referencia
  4. Patrón BALLENERA          — Cup&Handle, Double Bottom (alta precisión)
  5. Patrón CORTO PLAZO        — Darvas, Supply/Demand, Engulfing (cuando no hay Ballenera)
  6. Sizing con interés compuesto — equity_actual × pct_por_estrategia

Modelo de capital (interés compuesto):
  Cada nueva operación se calcula sobre el equity ACTUAL, que ya incluye
  las ganancias reinvertidas de las operaciones anteriores.
  compounding_factor = equity_actual / equity_inicial

  Ballenera : 20% por operación (máx 5 posiciones simultáneas)
  Corto Plazo: 2.5%–15% según señal, sector momentum y volatilidad
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.pattern_detector import PatternDetector, PatternSignal, ExitSignal
from core.whale_filters import RelativeStrengthFilter, TrendFilter, VolumeAnalyzer

logger = logging.getLogger(__name__)

# ── Regímenes HMM compatibles ────────────────────────────────────────────────
BALLENERA_OK_REGIMES   = {"BULL", "STRONG_BULL", "EUPHORIA", "NEUTRAL", "WEAK_BULL"}
CORTO_PLAZO_OK_REGIMES = {"BULL", "STRONG_BULL", "NEUTRAL"}
BLOCKED_REGIMES        = {"CRASH", "STRONG_BEAR"}

# ── Sizing ───────────────────────────────────────────────────────────────────
BALLENERA_POSITION_PCT   = 0.20    # 20% → máx 5 posiciones
CORTO_PLAZO_BASE_PCT: Dict[str, float] = {
    "SUPPLY_DEMAND_COMPRESSION": 0.12,
    "DARVAS_BOX":                0.10,
    "ENGULFING_CANDLE":          0.07,
}
CORTO_PLAZO_MIN_PCT = 0.025
CORTO_PLAZO_MAX_PCT = 0.15

# ── Gestión de riesgo ────────────────────────────────────────────────────────
MAX_STOP_PCT = 0.05    # stop loss máximo absoluto = 5% desde el punto de entrada
MIN_POS_USD  = 100.0   # mínimo eToro


@dataclass
class WhaleSignal:
    """Señal enriquecida con metadatos Whale Capital."""
    symbol: str
    strategy: str               # "BALLENERA" | "CORTO_PLAZO"
    pattern: PatternSignal
    rs_score: float
    rs_correction: float        # RS durante la última corrección del mercado
    is_market_leader: bool
    hmm_regime: str
    position_size_pct: float
    position_size_usd: float
    entry_price: float
    stop_loss: float
    compounding_factor: float   # equity_actual / equity_inicial


class BalleneroOrchestrator:
    """
    Buscador de Market Leaders — orquestador principal de Whale Capital.

    Para superar al mercado de manera consistente, nos enfocamos
    exclusivamente en lo MEJOR de lo MEJOR en su MEJOR timing.
    """

    def __init__(
        self,
        equity: float,
        initial_equity: float = 560.05,
        rs_window: int = 252,
        rs_leader_threshold: float = 1.15,
        rs_laggard_threshold: float = 0.85,
    ):
        self.equity           = equity
        self.initial_equity   = initial_equity
        self.rs_filter        = RelativeStrengthFilter(
            window=rs_window,
            leader_threshold=rs_leader_threshold,
            laggard_threshold=rs_laggard_threshold,
        )
        self.pattern_detector = PatternDetector()
        self.trend_filter     = TrendFilter()

    # ─────────────────────────────────────────────────────────────────
    # PUNTO DE ENTRADA PRINCIPAL
    # ─────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        asset_bars: pd.DataFrame,
        index_bars: pd.DataFrame,
        hmm_regime: str,
        n_open_positions: int,
        available_cash: float,
    ) -> Optional[WhaleSignal]:
        """
        Evalúa si un activo es un Market Leader con punto de compra óptimo (MPC).

        El MPC es la combinación de:
          - Mayor rentabilidad potencial (patrón en fase de incubación)
          - Menor riesgo (stop ceñido bajo el breakout)
          - Mayor velocidad (capital trabajando desde el primer día)
          - Mayores probabilidades de éxito (fuerza relativa + confirmación de volumen)
        """

        # 1. Filtro de régimen macro (HMM) — no luchar contra el mercado
        if hmm_regime in BLOCKED_REGIMES:
            logger.debug("[Whale] %s: régimen %s bloqueado", symbol, hmm_regime)
            return None

        # 2. Filtro de tendencia multi-nivel
        if not self.trend_filter.is_uptrend(asset_bars, strict=False):
            logger.debug("[Whale] %s: sin tendencia alcista — descartado", symbol)
            return None

        market_ok = self.trend_filter.is_market_healthy(index_bars)
        if not market_ok:
            logger.debug("[Whale] %s: mercado sin tendencia alcista — sin señal", symbol)
            return None

        # 3. Fuerza Relativa — descarte inmediato de rezagados
        is_leader, rs_score = self.rs_filter.is_market_leader(asset_bars, index_bars)
        if self.rs_filter.is_laggard(rs_score):
            logger.info("[Whale] %s: RS=%.3f — REZAGADO, descartado", symbol, rs_score)
            return None

        rs_correction = self.rs_filter.rs_during_correction(asset_bars, index_bars, days=20)

        # 4. Detección de patrones — BALLENERA tiene prioridad absoluta
        pattern: Optional[PatternSignal] = None

        if hmm_regime in BALLENERA_OK_REGIMES:
            pattern = (
                self.pattern_detector.detect_cup_and_handle(asset_bars)
                or self.pattern_detector.detect_double_bottom(asset_bars)
            )

        if pattern is None and hmm_regime in CORTO_PLAZO_OK_REGIMES:
            pattern = (
                self.pattern_detector.detect_darvas_box(asset_bars)
                or self.pattern_detector.detect_supply_demand_compression(asset_bars)
                or self.pattern_detector.detect_engulfing(asset_bars, index_bars)
            )

        if pattern is None:
            logger.debug("[Whale] %s: sin patrón detectado", symbol)
            return None

        # 5. Límite de posiciones concurrentes
        if pattern.strategy == "BALLENERA" and n_open_positions >= 5:
            logger.info("[Whale] %s: Ballenera — máx 5 posiciones ya alcanzado", symbol)
            return None

        # 6. Sizing y stop loss
        return self._build_signal(
            symbol=symbol,
            pattern=pattern,
            rs_score=rs_score,
            rs_correction=rs_correction,
            is_leader=is_leader,
            hmm_regime=hmm_regime,
            available_cash=available_cash,
        )

    # ─────────────────────────────────────────────────────────────────
    # SIZING CON INTERÉS COMPUESTO
    # ─────────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        symbol: str,
        pattern: PatternSignal,
        rs_score: float,
        rs_correction: float,
        is_leader: bool,
        hmm_regime: str,
        available_cash: float,
    ) -> Optional[WhaleSignal]:
        """
        Calcula el tamaño de posición usando el equity ACTUAL (interés compuesto).

        El coste de oportunidad se evita entrando justo en el breakout, no antes,
        para que el capital empiece a trabajar desde el primer día.
        """
        compounding = self.equity / self.initial_equity

        if pattern.strategy == "BALLENERA":
            pct = BALLENERA_POSITION_PCT
        else:
            base = CORTO_PLAZO_BASE_PCT.get(pattern.pattern_name, 0.10)
            pct  = max(CORTO_PLAZO_MIN_PCT,
                       min(CORTO_PLAZO_MAX_PCT, base * pattern.confidence))

        position_usd = round(min(self.equity * pct, available_cash), 2)

        if position_usd < MIN_POS_USD:
            logger.info("[Whale] %s: size $%.2f < $100 — omitido", symbol, position_usd)
            return None

        # Stop: usar el del patrón, pero nunca más del 5% desde el entry
        entry_price = pattern.breakout_level
        hard_stop   = entry_price * (1 - MAX_STOP_PCT)
        stop_loss   = max(pattern.stop_loss_level, hard_stop)
        stop_pct    = (entry_price - stop_loss) / entry_price

        logger.info(
            "[Whale] SEÑAL %s | %s | patrón=%s | RS=%.3f | RS_corr=%.3f | "
            "size=$%.2f (%.1f%%) | entry=%.4f | stop=%.4f (%.2f%%) | "
            "compounding=%.2fx | vol_ok=%s",
            pattern.strategy, symbol, pattern.pattern_name,
            rs_score, rs_correction,
            position_usd, pct * 100,
            entry_price, stop_loss, stop_pct * 100,
            compounding, pattern.volume_confirmed,
        )

        return WhaleSignal(
            symbol=symbol,
            strategy=pattern.strategy,
            pattern=pattern,
            rs_score=rs_score,
            rs_correction=rs_correction,
            is_market_leader=is_leader,
            hmm_regime=hmm_regime,
            position_size_pct=round(pct, 4),
            position_size_usd=position_usd,
            entry_price=entry_price,
            stop_loss=stop_loss,
            compounding_factor=round(compounding, 4),
        )

    # ─────────────────────────────────────────────────────────────────
    # SEÑALES DE SALIDA
    # ─────────────────────────────────────────────────────────────────

    def detect_exit(
        self,
        symbol: str,
        asset_bars: pd.DataFrame,
        entry_price: float,
        strategy: str = "BALLENERA",
    ) -> Optional[ExitSignal]:
        """
        Detecta una de las 3 causas de salida de Whale Capital:
        cambio de tendencia, euforia o distribución.
        """
        return self.pattern_detector.detect_exit_signal(asset_bars, entry_price, strategy)

    # ─────────────────────────────────────────────────────────────────
    # PRIORIZACIÓN
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def prioritize(signals: List[WhaleSignal]) -> List[WhaleSignal]:
        """
        Ordena señales por:
          1. Estrategia: BALLENERA siempre antes que CORTO_PLAZO
          2. RS_score   : mayor fuerza relativa primero
          3. Confianza  : mayor precisión del patrón primero
        """
        def key(s: WhaleSignal) -> tuple:
            strat = 0 if s.strategy == "BALLENERA" else 1
            return (strat, -s.rs_score, -s.pattern.confidence)

        return sorted(signals, key=key)

    # ─────────────────────────────────────────────────────────────────
    # ACTUALIZACIÓN DE ESTADO
    # ─────────────────────────────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        """
        Actualiza el equity cada tick para que el sizing refleje
        las ganancias reinvertidas (interés compuesto activo).
        """
        self.equity = equity
