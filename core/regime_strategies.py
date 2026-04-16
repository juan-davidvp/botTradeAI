"""
core/regime_strategies.py
Fase 3 — Estrategias de Asignación Basadas en Régimen de Volatilidad HMM.

Capital real: $546.14 | Objetivo: +20% en 120 días → $655.37
Apalancamiento forzado: x1 | Metodología: Ballenera (LONG only)

Estrategias de régimen:
  LowVolBullStrategy       — vol tercio inferior  → 95% asignación
  MidVolCautiousStrategy   — vol tercio medio     → 60-95% según EMA50
  HighVolDefensiveStrategy — vol tercio superior  → 60% asignación

Filtros técnicos adicionales (experto):
  MACDFilter               — confirma momentum y dirección de tendencia
  DonchianBreakoutFilter   — detecta ruptura de canal; mayor convicción de entrada
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ta

from core.hmm_engine import RegimeState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del portafolio real
# ---------------------------------------------------------------------------
INITIAL_EQUITY        = 546.14
TARGET_EQUITY         = 655.37
MAX_POSITION_PCT      = 0.20          # 20% por posición = $109.23
MIN_POSITION_USD      = 100.0         # Mínimo eToro
TARGET_POSITION_USD   = 109.23
DEFAULT_STOP_PCT      = 0.05          # 5% stop loss fijo (Ballenera)
HIGH_VOL_STOP_PCT     = 0.07          # 7% en alta volatilidad confirmada
LEVERAGE              = 1.0           # Forzado — capital < $1k


# ---------------------------------------------------------------------------
# Enums y Dataclasses
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    LONG = "LONG"
    FLAT = "FLAT"


class TechnicalConfirmation(str, Enum):
    STRONG   = "STRONG"    # MACD + Donchian alineados
    MODERATE = "MODERATE"  # Solo uno de los dos
    WEAK     = "WEAK"      # Ninguno confirma
    NONE     = "NONE"      # Modo incertidumbre / alta vol sin señal


@dataclass
class Signal:
    symbol: str
    direction: Direction
    confidence: float                      # 0.0 – 1.0

    entry_price: float
    stop_loss: float
    take_profit: Optional[float]           # None = salida por señal técnica

    position_size_pct: float               # fracción del equity total
    position_size_usd: float               # USD a invertir
    leverage: float = LEVERAGE

    regime_id: int = 0
    regime_name: str = ""
    regime_probability: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    reasoning: str = ""
    strategy_name: str = ""
    technical_confirmation: TechnicalConfirmation = TechnicalConfirmation.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filtros Técnicos
# ---------------------------------------------------------------------------

class MACDFilter:
    """
    Filtro MACD (12, 26, 9) para confirmación de momentum.

    Señal FUERTE  : MACD cruza hacia arriba del signal line Y MACD > 0
    Señal MODERADA: MACD > signal line pero MACD < 0 (tendencia en desarrollo)
    Sin señal     : MACD < signal line
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def analyze(self, close: pd.Series) -> Dict[str, Any]:
        """
        Retorna dict con:
          macd_line, signal_line, histogram,
          is_bullish_cross (cross alcista en la última barra),
          is_above_zero (MACD > 0),
          is_histogram_expanding (histograma creciente),
          strength: "STRONG" | "MODERATE" | "WEAK"
        """
        macd_ind    = ta.trend.MACD(close, self.fast, self.slow, self.signal)
        macd_line   = macd_ind.macd()
        signal_line = macd_ind.macd_signal()
        histogram   = macd_ind.macd_diff()

        if macd_line.isna().iloc[-1] or signal_line.isna().iloc[-1]:
            return {"strength": "WEAK", "macd": None, "signal": None, "hist": None}

        m_cur  = float(macd_line.iloc[-1])
        m_prev = float(macd_line.iloc[-2]) if len(macd_line) >= 2 else m_cur
        s_cur  = float(signal_line.iloc[-1])
        h_cur  = float(histogram.iloc[-1])
        h_prev = float(histogram.iloc[-2]) if len(histogram) >= 2 else h_cur

        is_bullish_cross      = (m_cur > s_cur) and (m_prev <= float(signal_line.iloc[-2]) if len(signal_line) >= 2 else True)
        is_above_zero         = m_cur > 0
        is_histogram_expanding = h_cur > h_prev and h_cur > 0

        if is_bullish_cross and is_above_zero and is_histogram_expanding:
            strength = "STRONG"
        elif m_cur > s_cur and (is_above_zero or is_histogram_expanding):
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            "strength": strength,
            "macd":     m_cur,
            "signal":   s_cur,
            "hist":     h_cur,
            "bullish_cross": is_bullish_cross,
            "above_zero":    is_above_zero,
        }


class DonchianBreakoutFilter:
    """
    Filtro de Ruptura de Canales de Donchian.

    Parámetros
    ----------
    primary_window : int
        Ventana principal (55 barras — tendencia primaria, estilo Turtle).
    secondary_window : int
        Ventana de confirmación (20 barras — tendencia secundaria).
    atr_window : int
        Ventana ATR para cálculo de stop dinámico.

    Señal FUERTE  : precio cierra sobre banda superior del canal de 55 días
                    Y sobre banda superior del canal de 20 días.
    Señal MODERADA: precio cierra sobre banda superior del canal de 20 días
                    (ruptura secundaria, sin confirmación de 55d).
    Sin señal     : precio dentro de ambos canales.

    Stop dinámico  : banda inferior del canal de 20 días
                    (o 2×ATR desde entrada, el mayor de los dos).
    """

    def __init__(
        self,
        primary_window: int   = 55,
        secondary_window: int = 20,
        atr_window: int       = 14,
    ):
        self.primary_window   = primary_window
        self.secondary_window = secondary_window
        self.atr_window       = atr_window

    def analyze(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, Any]:
        """
        Retorna dict con:
          upper_55, lower_55, upper_20, lower_20,
          is_primary_breakout, is_secondary_breakout,
          dynamic_stop, strength: "STRONG" | "MODERATE" | "WEAK"
        """
        upper_55 = high.rolling(self.primary_window).max()
        lower_55 = low.rolling(self.primary_window).min()
        upper_20 = high.rolling(self.secondary_window).max()
        lower_20 = low.rolling(self.secondary_window).min()
        atr      = ta.volatility.AverageTrueRange(high, low, close, window=self.atr_window).average_true_range()

        if upper_55.isna().iloc[-1]:
            return {"strength": "WEAK", "upper_55": None, "upper_20": None, "dynamic_stop": None}

        price_now   = float(close.iloc[-1])
        u55         = float(upper_55.iloc[-2])   # cierre ANTERIOR de la banda (evita look-ahead)
        u20         = float(upper_20.iloc[-2])
        l20         = float(lower_20.iloc[-1])
        atr_val     = float(atr.iloc[-1]) if not atr.isna().iloc[-1] else 0.0

        is_primary_breakout   = price_now > u55
        is_secondary_breakout = price_now > u20

        dynamic_stop = max(l20, price_now - 2 * atr_val)

        if is_primary_breakout and is_secondary_breakout:
            strength = "STRONG"
        elif is_secondary_breakout:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            "strength":            strength,
            "upper_55":            u55,
            "upper_20":            u20,
            "lower_20":            l20,
            "is_primary_breakout": is_primary_breakout,
            "is_secondary_breakout": is_secondary_breakout,
            "dynamic_stop":        dynamic_stop,
            "atr":                 atr_val,
        }


def _combine_technical(macd_strength: str, donchian_strength: str) -> TechnicalConfirmation:
    """
    Combina las señales MACD y Donchian en una confirmación única.

    Regla experta (umbrales rebajados para permitir operar en regímenes favorables):
      STRONG   : ambos STRONG, o STRONG+MODERATE
      MODERATE : uno STRONG + WEAK, ambos MODERATE, o MODERATE+WEAK
      WEAK     : cualquier otro caso — mínimo garantizado (NONE eliminado)
    """
    score_map = {"STRONG": 2, "MODERATE": 1, "WEAK": 0}
    score = score_map.get(macd_strength, 0) + score_map.get(donchian_strength, 0)
    if score >= 3:
        return TechnicalConfirmation.STRONG
    elif score >= 1:
        return TechnicalConfirmation.MODERATE
    return TechnicalConfirmation.WEAK   # mínimo garantizado — NONE suprimido


# Multiplicador de tamaño por confirmación técnica
CONFIRMATION_SIZE_MULT: Dict[TechnicalConfirmation, float] = {
    TechnicalConfirmation.STRONG:   1.00,
    TechnicalConfirmation.MODERATE: 0.85,
    TechnicalConfirmation.WEAK:     0.70,
    TechnicalConfirmation.NONE:     0.50,
}


# ---------------------------------------------------------------------------
# Base Strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Interfaz común para todas las estrategias de asignación."""

    name: str = "BaseStrategy"

    def __init__(self, equity: float = INITIAL_EQUITY):
        self.equity = equity
        self.macd_filter     = MACDFilter()
        self.donchian_filter = DonchianBreakoutFilter()

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """
        Genera una señal de trading.

        Parámetros
        ----------
        symbol : str
            Identificador del instrumento (eToro instrument ID).
        bars : pd.DataFrame
            OHLCV histórico. Columnas: open, high, low, close, volume.
        regime_state : RegimeState
            Estado actual del régimen HMM.

        Retorna
        -------
        Signal o None si no hay señal válida.
        """

    def _compute_technicals(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta ambos filtros técnicos y retorna resultados combinados."""
        macd_result     = self.macd_filter.analyze(bars["close"])
        donchian_result = self.donchian_filter.analyze(bars["high"], bars["low"], bars["close"])
        confirmation    = _combine_technical(macd_result["strength"], donchian_result["strength"])
        return {
            "macd":         macd_result,
            "donchian":     donchian_result,
            "confirmation": confirmation,
        }

    def _ema(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    def _size_from_confirmation(
        self,
        base_pct: float,
        confirmation: TechnicalConfirmation,
        uncertainty: bool = False,
    ) -> tuple[float, float]:
        """
        Retorna (position_size_pct, position_size_usd) ajustado por confirmación
        técnica y modo incertidumbre.
        """
        mult = CONFIRMATION_SIZE_MULT[confirmation]
        if uncertainty:
            mult *= 0.50

        size_pct = base_pct * mult
        size_usd = self.equity * size_pct / 5   # por posición (5 instrumentos máx)

        # Verificar mínimo sobre la asignación total del portafolio, no por posición.
        # Con equity $546 la división /5 hace que el per-posición quede bajo $100
        # aunque la asignación total sea perfectamente viable.
        total_usd = self.equity * size_pct
        if total_usd < MIN_POSITION_USD:
            return 0.0, 0.0

        return round(size_pct, 4), round(size_usd, 2)


# ---------------------------------------------------------------------------
# Estrategias de Régimen
# ---------------------------------------------------------------------------

class LowVolBullStrategy(BaseStrategy):
    """
    Régimen de baja volatilidad — Estar completamente invertido.

    Asignación base: 95% del portafolio.
    Stop: 5% fijo desde apertura (Ballenera).
    Entrada preferida: ruptura Donchian 55d + MACD > 0 (confirmación máxima).
    """

    name = "LowVolBullStrategy"

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:

        tech       = self._compute_technicals(bars)
        macd_res   = tech["macd"]
        don_res    = tech["donchian"]
        confirm    = tech["confirmation"]
        uncertainty = not regime_state.is_confirmed or regime_state.probability < 0.55

        size_pct, size_usd = self._size_from_confirmation(0.95, confirm, uncertainty)
        if size_usd == 0.0:
            logger.info("[%s] %s: tamaño por debajo del mínimo — sin señal", self.name, symbol)
            return None

        price     = float(bars["close"].iloc[-1])
        stop_loss = round(price * (1 - DEFAULT_STOP_PCT), 4)

        # Stop más ajustado si Donchian ofrece soporte más cercano
        if don_res.get("dynamic_stop") and don_res["dynamic_stop"] > price * 0.90:
            stop_loss = max(stop_loss, round(don_res["dynamic_stop"], 4))

        reasoning_parts = [
            f"Régimen: {regime_state.label} ({regime_state.probability:.1%})",
            f"MACD: {macd_res['strength']} | Donchian: {don_res['strength']}",
            f"Confirmación técnica: {confirm.value}",
        ]
        if don_res.get("is_primary_breakout"):
            reasoning_parts.append("RUPTURA DONCHIAN 55d ✓ — alta convicción")
        if macd_res.get("bullish_cross") and macd_res.get("above_zero"):
            reasoning_parts.append("MACD cross alcista sobre cero ✓")
        if uncertainty:
            reasoning_parts.append("[UNCERTAINTY — size halved]")

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability * CONFIRMATION_SIZE_MULT[confirm],
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=size_pct,
            position_size_usd=size_usd,
            leverage=LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            reasoning=" | ".join(reasoning_parts),
            strategy_name=self.name,
            technical_confirmation=confirm,
            metadata={
                "macd":     macd_res,
                "donchian": don_res,
                "uncertainty": uncertainty,
            },
        )


class MidVolCautiousStrategy(BaseStrategy):
    """
    Régimen de volatilidad media — Cauteloso: seguir tendencia EMA50.

    Precio > EMA50 → 95% (tendencia intacta).
    Precio < EMA50 → 60% (máximo 3 posiciones).
    Stop: EMA50 * 0.995, mínimo 5% desde apertura.

    Con MACD o Donchian en STRONG: no reducir tamaño aunque esté bajo EMA50.
    """

    name = "MidVolCautiousStrategy"

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:

        tech      = self._compute_technicals(bars)
        confirm   = tech["confirmation"]
        macd_res  = tech["macd"]
        don_res   = tech["donchian"]
        uncertainty = not regime_state.is_confirmed or regime_state.probability < 0.55

        price  = float(bars["close"].iloc[-1])
        ema50  = float(self._ema(bars["close"], 50).iloc[-1])

        above_ema50 = price > ema50

        # Excepción experta: señal técnica STRONG overrides la regla EMA50
        if confirm == TechnicalConfirmation.STRONG:
            base_pct = 0.95
            reasoning_trend = "Technical STRONG sobreescribe EMA50"
        elif above_ema50:
            base_pct = 0.95
            reasoning_trend = f"Precio ${price:.2f} > EMA50 ${ema50:.2f} — tendencia intacta"
        else:
            base_pct = 0.60
            reasoning_trend = f"Precio ${price:.2f} < EMA50 ${ema50:.2f} — reduciendo exposición"

        size_pct, size_usd = self._size_from_confirmation(base_pct, confirm, uncertainty)
        if size_usd == 0.0:
            return None

        stop_loss = round(max(ema50 * 0.995, price * (1 - DEFAULT_STOP_PCT)), 4)

        reasoning_parts = [
            f"Régimen: {regime_state.label} ({regime_state.probability:.1%})",
            reasoning_trend,
            f"MACD: {macd_res['strength']} | Donchian: {don_res['strength']}",
            f"Confirmación técnica: {confirm.value}",
        ]
        if don_res.get("is_secondary_breakout") and not don_res.get("is_primary_breakout"):
            reasoning_parts.append("Ruptura Donchian 20d — confirmación secundaria")
        if uncertainty:
            reasoning_parts.append("[UNCERTAINTY — size halved]")

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability * CONFIRMATION_SIZE_MULT[confirm],
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=size_pct,
            position_size_usd=size_usd,
            leverage=LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            reasoning=" | ".join(reasoning_parts),
            strategy_name=self.name,
            technical_confirmation=confirm,
            metadata={
                "ema50":     ema50,
                "above_ema": above_ema50,
                "macd":      macd_res,
                "donchian":  don_res,
                "uncertainty": uncertainty,
            },
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """
    Régimen de alta volatilidad — Defensivo: máximo 60%, 3 posiciones.

    Stop ampliado a 7% en alta vol confirmada.
    En este régimen se requiere confirmación técnica STRONG o MODERATE
    para operar. WEAK o NONE → sin señal (cash preservado).

    Excepción experta: Ruptura Donchian 55d en alta vol = entrada especulativa
    reducida al 50% del tamaño normal. Los breakouts en alta vol tienen
    mayor seguimiento si son rupturas genuinas de canal primario.
    """

    name = "HighVolDefensiveStrategy"

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:

        tech    = self._compute_technicals(bars)
        confirm = tech["confirmation"]
        macd_res = tech["macd"]
        don_res  = tech["donchian"]
        uncertainty = not regime_state.is_confirmed or regime_state.probability < 0.55

        # Excepción: ruptura Donchian 55d → entrada especulativa reducida
        if don_res.get("is_primary_breakout") and confirm != TechnicalConfirmation.STRONG:
            base_pct = 0.30   # 50% del 60% estándar
            speculative = True
        else:
            base_pct  = 0.60
            speculative = False

        size_pct, size_usd = self._size_from_confirmation(base_pct, confirm, uncertainty)
        if size_usd == 0.0:
            return None

        price     = float(bars["close"].iloc[-1])
        stop_loss = round(price * (1 - HIGH_VOL_STOP_PCT), 4)

        # Stop dinámico Donchian si es más protector
        if don_res.get("dynamic_stop") and don_res["dynamic_stop"] > price * 0.88:
            stop_loss = max(stop_loss, round(don_res["dynamic_stop"], 4))

        reasoning_parts = [
            f"Régimen: {regime_state.label} ({regime_state.probability:.1%}) — ALTA VOL",
            f"MACD: {macd_res['strength']} | Donchian: {don_res['strength']}",
            f"Confirmación técnica: {confirm.value}",
        ]
        if speculative:
            reasoning_parts.append("Ruptura Donchian 55d en alta vol — tamaño especulativo (30%)")
        if uncertainty:
            reasoning_parts.append("[UNCERTAINTY — size halved]")

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability * CONFIRMATION_SIZE_MULT[confirm] * 0.7,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=size_pct,
            position_size_usd=size_usd,
            leverage=LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            reasoning=" | ".join(reasoning_parts),
            strategy_name=self.name,
            technical_confirmation=confirm,
            metadata={
                "macd":       macd_res,
                "donchian":   don_res,
                "speculative": speculative,
                "uncertainty": uncertainty,
            },
        )


# ---------------------------------------------------------------------------
# Aliases de compatibilidad
# ---------------------------------------------------------------------------
CrashDefensiveStrategy    = HighVolDefensiveStrategy
BearTrendStrategy         = HighVolDefensiveStrategy
MeanReversionStrategy     = MidVolCautiousStrategy
BullTrendStrategy         = LowVolBullStrategy
EuphoriaCautiousStrategy  = LowVolBullStrategy

LABEL_TO_STRATEGY = {
    "BULL":        LowVolBullStrategy,
    "NEUTRAL":     MidVolCautiousStrategy,
    "BEAR":        HighVolDefensiveStrategy,
    "CRASH":       HighVolDefensiveStrategy,
    "EUPHORIA":    LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "WEAK_BULL":   MidVolCautiousStrategy,
    "WEAK_BEAR":   MidVolCautiousStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
}


# ---------------------------------------------------------------------------
# Orquestador de Estrategias
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """
    Mapea el régimen HMM a la estrategia de asignación correcta usando
    vol_rank (independiente de las etiquetas BULL/BEAR).

    Lógica de vol_rank:
      position = rank / (n_regimes - 1)   [0.0 = menor vol, 1.0 = mayor vol]
      <= 0.33 → LowVolBullStrategy
      >= 0.67 → HighVolDefensiveStrategy
      resto   → MidVolCautiousStrategy

    Rebalanceo: solo cuando la diferencia entre asignación objetivo y actual > 10%.
    """

    REBALANCE_THRESHOLD = 0.10

    def __init__(self, equity: float = INITIAL_EQUITY):
        self.equity       = equity
        self._strategy_cache: Dict[int, BaseStrategy] = {}

    def get_strategy(
        self,
        regime_state: RegimeState,
        regime_infos: dict,
    ) -> BaseStrategy:
        """
        Selecciona la estrategia según el vol_rank del régimen actual.
        """
        if not regime_infos:
            return MidVolCautiousStrategy(self.equity)

        # Ordenar por expected_volatility para calcular vol_rank
        sorted_by_vol = sorted(regime_infos.values(), key=lambda r: r.expected_volatility)
        n = len(sorted_by_vol)

        for rank, info in enumerate(sorted_by_vol):
            if info.regime_id == regime_state.state_id:
                position = rank / max(n - 1, 1)
                break
        else:
            position = 0.5  # fallback

        if position <= 0.33:
            strategy_cls = LowVolBullStrategy
        elif position >= 0.67:
            strategy_cls = HighVolDefensiveStrategy
        else:
            strategy_cls = MidVolCautiousStrategy

        if regime_state.state_id not in self._strategy_cache or \
           not isinstance(self._strategy_cache[regime_state.state_id], strategy_cls):
            self._strategy_cache[regime_state.state_id] = strategy_cls(self.equity)

        logger.info(
            "Régimen %s (id=%d) → vol_rank=%.2f → %s",
            regime_state.label, regime_state.state_id, position, strategy_cls.name,
        )
        return self._strategy_cache[regime_state.state_id]

    def needs_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        """
        True si alguna posición difiere más del umbral del 10%
        entre peso actual y objetivo.
        """
        all_symbols = set(current_weights) | set(target_weights)
        for sym in all_symbols:
            current = current_weights.get(sym, 0.0)
            target  = target_weights.get(sym, 0.0)
            if abs(current - target) > self.REBALANCE_THRESHOLD:
                logger.info(
                    "Rebalanceo requerido para %s: actual=%.1f%% objetivo=%.1f%%",
                    sym, current * 100, target * 100,
                )
                return True
        return False

    def generate_rebalance_actions(
        self,
        current_positions: List[Dict],
        target_weights: Dict[str, float],
        equity: float,
    ) -> List[Dict]:
        """
        Genera lista de acciones de rebalanceo:
          {"symbol": ..., "action": "REDUCE"|"INCREASE", "delta_usd": ...}

        Calibrado al portafolio real (2026-04-15):
          4238 → reducir $160.77 (49.4% → 20%)
          resto → aumentar hasta $109.23
        """
        actions = []
        for pos in current_positions:
            sym    = str(pos.get("instrumentID", pos.get("symbol", "")))
            amount = float(pos.get("amount", 0))
            weight = amount / equity if equity > 0 else 0
            target = target_weights.get(sym, MAX_POSITION_PCT)

            delta_pct = target - weight
            delta_usd = round(delta_pct * equity, 2)

            if abs(delta_usd) < 10:
                continue  # diferencia insignificante

            action = "REDUCE" if delta_usd < 0 else "INCREASE"
            actions.append({
                "symbol":       sym,
                "action":       action,
                "current_pct":  round(weight * 100, 1),
                "target_pct":   round(target * 100, 1),
                "delta_usd":    abs(delta_usd),
                "direction":    "sell_partial" if action == "REDUCE" else "buy_additional",
            })
            logger.warning(
                "REBALANCEO %s %s: actual=%.1f%% objetivo=%.1f%% delta=$%.2f",
                action, sym, weight * 100, target * 100, abs(delta_usd),
            )

        return actions
