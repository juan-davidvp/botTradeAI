"""
core/pattern_detector.py
Whale Capital — Detección Algorítmica de Patrones de Acumulación Institucional.

PATRONES BALLENERA (alta precisión, pocas oportunidades):
  Cup & Handle  — Taza con Asa (5 fases de acumulación institucional)
  Double Bottom — Doble suelo a nivel similar (1ª acumulación institucional)

PATRONES CORTO PLAZO (más frecuentes, visión 1 mes):
  Darvas Box              — Caja de consolidación lateral en tendencia alcista
  Supply/Demand Compress  — Máximos horizontales + mínimos ascendentes
  Engulfing Candle        — Vela envolvente alcista durante corrección del mercado

SEÑALES DE SALIDA (comunes a ambas estrategias):
  Trend Change  — Cierre rompe bajo SMA50 relevante
  Euphoria      — GAP alcista a máximos históricos con volumen climático
  Distribution  — Lateral en ATH + GAP bajista + volumen climático
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatternSignal:
    pattern_name: str
    strategy: str                # "BALLENERA" | "CORTO_PLAZO"
    breakout_level: float        # precio de entrada = punto de rotura
    stop_loss_level: float       # stop justo debajo del breakout (máx 5%)
    confidence: float            # 0.0–1.0
    pattern_depth_pct: float     # profundidad del patrón (contexto de sizing)
    volume_confirmed: bool       # breakout confirmado por volumen
    days_in_formation: int       # barras en las que se formó el patrón


@dataclass
class ExitSignal:
    reason: str    # "TREND_CHANGE" | "EUPHORIA" | "DISTRIBUTION"
    urgency: str   # "IMMEDIATE" | "END_OF_DAY"
    price: float


class PatternDetector:
    """
    Traduce los patrones visuales de Whale Capital en reglas algorítmicas.

    Principio base: un patrón es una batalla entre compradores y vendedores.
    El punto de compra (breakout) es el momento en que los compradores
    han abatido la defensa vendedora y tienen el campo libre por delante.
    """

    def __init__(
        self,
        min_cup_depth_pct: float    = 0.15,   # profundidad mínima de la taza
        max_cup_depth_pct: float    = 0.60,   # profundidad máxima (> 60% = inestable)
        min_handle_bars: int        = 5,      # duración mínima del handle
        max_handle_bars: int        = 65,     # duración máxima del handle
        max_handle_depth_pct: float = 0.20,   # retroceso máximo del handle = 20%
        volume_breakout_mult: float = 1.5,    # volumen en breakout ≥ 1.5× media
        volume_climactic_mult: float = 2.0,   # volumen climático (salida) ≥ 2× media
        darvas_max_range_pct: float = 0.10,   # caja Darvas max high-low = 10%
        darvas_min_bars: int        = 10,     # caja Darvas duración mínima
    ):
        self.min_cup_depth_pct     = min_cup_depth_pct
        self.max_cup_depth_pct     = max_cup_depth_pct
        self.min_handle_bars       = min_handle_bars
        self.max_handle_bars       = max_handle_bars
        self.max_handle_depth_pct  = max_handle_depth_pct
        self.volume_breakout_mult  = volume_breakout_mult
        self.volume_climactic_mult = volume_climactic_mult
        self.darvas_max_range_pct  = darvas_max_range_pct
        self.darvas_min_bars       = darvas_min_bars

    # ─────────────────────────────────────────────────────────────────
    # PATRONES BALLENERA
    # ─────────────────────────────────────────────────────────────────

    def detect_cup_and_handle(self, bars: pd.DataFrame) -> Optional[PatternSignal]:
        """
        CUP & HANDLE — Algoritmo de 7 pasos:

        1. Pivot high: máximo de los últimos 252 barras.
        2. Cup bottom: mínimo entre pivot y zona actual.
        3. Profundidad: 15%–60%.
        4. Recuperación: precio vuelve ≥ 75% del nivel del pivot.
        5. Handle: consolidación reciente (5–65 barras), retroceso ≤ 20%.
        6. Breakout: cierre ≥ 98% del máximo del handle.
        7. Volumen: ≥ 1.5× media 50 días en el breakout.

        El handle debe tener VOLUMEN SECO (bajo) — señal de acumulación silenciosa.
        """
        if len(bars) < 100:
            return None

        close  = bars["close"].astype(float)
        high   = bars["high"].astype(float)
        volume = bars["volume"].astype(float)

        # 1. Pivot high — excluyendo las últimas 5 barras del handle
        lookback = min(252, len(bars) - 5)
        pivot_idx   = int(np.argmax(close.values[:-(5)]))
        pivot_price = float(close.iloc[pivot_idx])

        if pivot_idx >= len(close) - 5:
            return None

        # 2. Cup bottom — mínimo desde el pivot hasta ahora
        cup_region  = close.iloc[pivot_idx:]
        cup_low     = float(cup_region.min())
        cup_low_pos = int(cup_region.values.argmin()) + pivot_idx

        # 3. Profundidad de la taza
        cup_depth_pct = (pivot_price - cup_low) / pivot_price
        if not (self.min_cup_depth_pct <= cup_depth_pct <= self.max_cup_depth_pct):
            return None

        # 4. Recuperación: el labio derecho debe estar cerca del pivot
        recovery_region = close.iloc[cup_low_pos:]
        if len(recovery_region) < self.min_handle_bars:
            return None

        right_lip    = float(recovery_region.max())
        recovery_pct = (right_lip - cup_low) / max(pivot_price - cup_low, 1e-8)
        if recovery_pct < 0.75:
            return None

        # 5. Handle — consolidación reciente en la parte ALTA de la taza.
        # El handle debe estar en el tercio superior (price > cup_low + 70% del recorrido).
        # Esto descarta barras del fondo/recuperación de la taza.
        handle_floor  = cup_low + 0.70 * (pivot_price - cup_low)
        above_floor   = (close > handle_floor).values[::-1]
        handle_len    = 0
        for v in above_floor:
            if v:
                handle_len += 1
            else:
                break
        handle_len = max(self.min_handle_bars, min(handle_len, self.max_handle_bars))

        if handle_len < self.min_handle_bars:
            return None

        handle_region = close.tail(handle_len)
        handle_high   = float(handle_region.max())
        handle_low    = float(handle_region.min())
        handle_depth  = (handle_high - handle_low) / handle_high if handle_high > 0 else 1.0

        if handle_depth > self.max_handle_depth_pct:
            return None

        # Handle debe estar cerca del right_lip (dentro del 15%)
        if abs(handle_high - right_lip) / max(right_lip, 1e-8) > 0.15:
            return None

        # 6. Breakout: precio sobre el máximo del handle
        current_close = float(close.iloc[-1])
        if current_close < handle_high * 0.98:
            return None

        # 7. Volumen en el breakout
        avg_vol_50   = float(volume.tail(51).iloc[:-1].mean())
        current_vol  = float(volume.iloc[-1])
        vol_ratio    = (current_vol / avg_vol_50) if avg_vol_50 > 0 else 1.0
        vol_confirmed = vol_ratio >= self.volume_breakout_mult

        # Volumen seco durante el handle (bonus de confianza)
        handle_vol   = float(volume.tail(handle_len).mean())
        prior_vol    = float(volume.tail(handle_len + 50).iloc[:-handle_len].mean())
        handle_dry   = (handle_vol / prior_vol < 0.7) if prior_vol > 0 else False

        days_formation = len(close) - pivot_idx
        stop_loss      = max(handle_low * 0.99, current_close * 0.95)

        confidence = min(1.0,
            0.35 * recovery_pct
            + 0.25 * (1.0 if vol_confirmed else 0.5)
            + 0.20 * (1.0 if handle_dry else 0.6)
            + 0.10 * min(1.0, days_formation / 120)
            + 0.10 * min(1.0, cup_depth_pct / 0.35)
        )

        logger.info(
            "[Pattern] CUP&HANDLE | depth=%.1f%% | recov=%.1f%% | vol=%.2fx | dry=%s | conf=%.2f",
            cup_depth_pct * 100, recovery_pct * 100, vol_ratio, handle_dry, confidence,
        )

        return PatternSignal(
            pattern_name="CUP_AND_HANDLE",
            strategy="BALLENERA",
            breakout_level=round(handle_high, 4),
            stop_loss_level=round(stop_loss, 4),
            confidence=round(confidence, 3),
            pattern_depth_pct=round(cup_depth_pct, 4),
            volume_confirmed=vol_confirmed,
            days_in_formation=days_formation,
        )

    def detect_double_bottom(self, bars: pd.DataFrame) -> Optional[PatternSignal]:
        """
        DOUBLE BOTTOM — Doble suelo a niveles similares.

        Algoritmo:
        1. Encontrar mínimos locales (ventana de 5 barras cada lado).
        2. Buscar par de mínimos con diferencia < 5% separados ≥ 20 barras.
        3. Neckline = máximo entre los dos suelos.
        4. El segundo mínimo debe ser reciente (últimas 30 barras).
        5. Breakout: cierre sobre la neckline con volumen.
        """
        if len(bars) < 80:
            return None

        close  = bars["close"].astype(float)
        volume = bars["volume"].astype(float)

        # Encontrar mínimos locales
        vals = close.values
        lows = [
            (i, float(vals[i]))
            for i in range(5, len(vals) - 5)
            if vals[i] == min(vals[i - 5:i + 6])
        ]

        if len(lows) < 2:
            return None

        # Buscar el mejor par de mínimos
        best_pair = None
        for i in range(len(lows)):
            for j in range(i + 1, len(lows)):
                idx1, v1 = lows[i]
                idx2, v2 = lows[j]

                if (idx2 - idx1) < 20:  # separación mínima 4 semanas
                    continue

                diff = abs(v1 - v2) / max(v1, v2)
                if diff > 0.05:  # mínimos similares (< 5% diferencia)
                    continue

                neckline = float(close.values[idx1:idx2 + 1].max())

                if best_pair is None or diff < best_pair[4]:
                    best_pair = (idx1, v1, idx2, v2, diff, neckline)

        if best_pair is None:
            return None

        idx1, v1, idx2, v2, diff, neckline = best_pair

        # El segundo mínimo debe ser reciente
        if (len(close) - 1 - idx2) > 30:
            return None

        # Breakout sobre neckline
        current_close = float(close.iloc[-1])
        if current_close < neckline * 0.98:
            return None

        avg_vol_50   = float(volume.tail(51).iloc[:-1].mean())
        current_vol  = float(volume.iloc[-1])
        vol_ratio    = (current_vol / avg_vol_50) if avg_vol_50 > 0 else 1.0
        vol_confirmed = vol_ratio >= self.volume_breakout_mult

        stop_raw  = min(v1, v2) * 0.99
        stop_loss = max(stop_raw, current_close * 0.95)

        pattern_depth  = (neckline - min(v1, v2)) / neckline
        days_formation = idx2 - idx1

        confidence = min(1.0,
            0.40 * (1 - diff / 0.05)
            + 0.30 * (1.0 if vol_confirmed else 0.5)
            + 0.20 * min(1.0, days_formation / 60)
            + 0.10 * min(1.0, pattern_depth / 0.25)
        )

        logger.info(
            "[Pattern] DOUBLE_BOTTOM | neckline=%.4f | diff=%.1f%% | vol=%.2fx | conf=%.2f",
            neckline, diff * 100, vol_ratio, confidence,
        )

        return PatternSignal(
            pattern_name="DOUBLE_BOTTOM",
            strategy="BALLENERA",
            breakout_level=round(neckline, 4),
            stop_loss_level=round(stop_loss, 4),
            confidence=round(confidence, 3),
            pattern_depth_pct=round(pattern_depth, 4),
            volume_confirmed=vol_confirmed,
            days_in_formation=days_formation,
        )

    # ─────────────────────────────────────────────────────────────────
    # PATRONES CORTO PLAZO
    # ─────────────────────────────────────────────────────────────────

    def detect_darvas_box(self, bars: pd.DataFrame) -> Optional[PatternSignal]:
        """
        CAJA DE DARVAS — Consolidación lateral comprimida en tendencia alcista.

        Algoritmo:
        1. Activo en tendencia alcista (close > EMA50).
        2. Buscar una caja (rango high-low ≤ 10%) de duración ≥ 10 barras.
        3. La caja debe estar por encima del histórico reciente (nuevos máximos).
        4. Breakout: cierre sobre el techo de la caja con volumen.
        """
        if len(bars) < 60:
            return None

        close  = bars["close"].astype(float)
        volume = bars["volume"].astype(float)

        ema50 = close.ewm(span=50, adjust=False).mean()
        if float(close.iloc[-1]) < float(ema50.iloc[-1]):
            return None

        # Buscar caja en diferentes longitudes (de mayor a menor, preferir más largas)
        for box_len in range(min(self.max_handle_bars, len(bars) - 10),
                             self.darvas_min_bars - 1, -5):
            box        = bars.tail(box_len)
            box_high   = float(box["high"].max())
            box_low    = float(box["low"].min())
            box_range  = (box_high - box_low) / box_high if box_high > 0 else 1.0

            if box_range > self.darvas_max_range_pct:
                continue

            # La caja debe estar en zona de máximos (no en zona baja)
            prior_close = close.iloc[-(box_len + 20):-(box_len)] if box_len + 20 <= len(bars) else close.head(10)
            if float(prior_close.max()) > box_high * 0.90:
                continue

            # Breakout sobre el techo
            current_close = float(close.iloc[-1])
            if current_close < box_high * 0.98:
                continue

            avg_vol   = float(volume.tail(box_len + 1).iloc[:-1].mean())
            curr_vol  = float(volume.iloc[-1])
            vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0
            vol_confirmed = vol_ratio >= self.volume_breakout_mult

            stop_loss = max(box_low * 0.99, current_close * 0.95)

            confidence = min(1.0,
                0.40 * (1 - box_range / self.darvas_max_range_pct)
                + 0.30 * (1.0 if vol_confirmed else 0.5)
                + 0.30 * min(1.0, box_len / 30)
            )

            logger.info(
                "[Pattern] DARVAS_BOX | range=%.1f%% | bars=%d | vol=%.2fx | conf=%.2f",
                box_range * 100, box_len, vol_ratio, confidence,
            )

            return PatternSignal(
                pattern_name="DARVAS_BOX",
                strategy="CORTO_PLAZO",
                breakout_level=round(box_high, 4),
                stop_loss_level=round(stop_loss, 4),
                confidence=round(confidence, 3),
                pattern_depth_pct=round(box_range, 4),
                volume_confirmed=vol_confirmed,
                days_in_formation=box_len,
            )

        return None

    def detect_supply_demand_compression(self, bars: pd.DataFrame) -> Optional[PatternSignal]:
        """
        COMPRENSIÓN DE OFERTA Y DEMANDA — Máximos horizontales + mínimos ascendentes.

        Algoritmo:
        1. Activo en tendencia alcista.
        2. Los vendedores defienden siempre la misma zona (máximos ≤ 2% de desviación).
        3. Los compradores compran cada vez más alto (mínimos ascendentes).
        4. Breakout sobre la resistencia con volumen — los compradores destruyen la muralla.
        """
        if len(bars) < 40:
            return None

        close  = bars["close"].astype(float)
        high   = bars["high"].astype(float)
        low    = bars["low"].astype(float)
        volume = bars["volume"].astype(float)

        ema50 = close.ewm(span=50, adjust=False).mean()
        if float(close.iloc[-1]) < float(ema50.iloc[-1]):
            return None

        win      = min(40, len(bars))
        w_high   = high.tail(win)
        w_low    = low.tail(win)
        w_close  = close.tail(win)

        resistance = float(w_high.max())

        # Al menos 2 máximos tocando la resistencia (zona defendida)
        at_resistance = (w_high >= resistance * 0.98).sum()
        if at_resistance < 2:
            return None

        # Mínimos ascendentes: segunda mitad tiene promedio más alto que la primera
        half       = len(w_low) // 2
        low_first  = float(w_low.values[:half].mean())
        low_second = float(w_low.values[half:].mean())
        if low_second <= low_first * 1.005:
            return None

        current_close = float(close.iloc[-1])
        if current_close < resistance * 0.99:
            return None

        avg_vol   = float(volume.tail(win + 1).iloc[:-1].mean())
        curr_vol  = float(volume.iloc[-1])
        vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0
        vol_confirmed = vol_ratio >= self.volume_breakout_mult

        stop_loss = max(float(w_low.tail(5).min()) * 0.99, current_close * 0.95)
        depth     = (resistance - low_first) / resistance

        rising_pct = (low_second / low_first - 1) * 100
        confidence = min(1.0,
            0.40 * min(1.0, at_resistance / 3)
            + 0.30 * (1.0 if vol_confirmed else 0.5)
            + 0.30 * min(1.0, rising_pct / 3)
        )

        logger.info(
            "[Pattern] SUPPLY_DEMAND | resistance=%.4f | lows+%.1f%% | vol=%.2fx | conf=%.2f",
            resistance, rising_pct, vol_ratio, confidence,
        )

        return PatternSignal(
            pattern_name="SUPPLY_DEMAND_COMPRESSION",
            strategy="CORTO_PLAZO",
            breakout_level=round(resistance, 4),
            stop_loss_level=round(stop_loss, 4),
            confidence=round(confidence, 3),
            pattern_depth_pct=round(depth, 4),
            volume_confirmed=vol_confirmed,
            days_in_formation=win,
        )

    def detect_engulfing(
        self,
        bars: pd.DataFrame,
        index_bars: Optional[pd.DataFrame] = None,
    ) -> Optional[PatternSignal]:
        """
        VELA ENVOLVENTE ALCISTA — Fuerza relativa evidente en corrección del mercado.

        Algoritmo:
        1. Mercado (índice) en corrección (últimas 5 barras bajistas).
        2. Activo en tendencia alcista (close > EMA50).
        3. Vela actual engloba completamente la anterior:
           open_t < low_{t-1}  AND  close_t > high_{t-1}.
        4. Activo muestra fortaleza ANTES que el índice (RS positivo en corrección).
        """
        if len(bars) < 60:
            return None

        close  = bars["close"].astype(float)
        high   = bars["high"].astype(float)
        low    = bars["low"].astype(float)
        open_  = bars["open"].astype(float)
        volume = bars["volume"].astype(float)

        ema50 = close.ewm(span=50, adjust=False).mean()
        if float(close.iloc[-1]) < float(ema50.iloc[-1]):
            return None

        if index_bars is not None and len(index_bars) >= 6:
            idx_c = index_bars["close"].astype(float)
            if float(idx_c.iloc[-1]) >= float(idx_c.iloc[-5]):
                return None  # el mercado no está corrigiendo

        # Buscar vela envolvente en las últimas 3 barras
        for i in range(-1, -4, -1):
            try:
                o_cur  = float(open_.iloc[i])
                c_cur  = float(close.iloc[i])
                h_prev = float(high.iloc[i - 1])
                l_prev = float(low.iloc[i - 1])

                if c_cur <= o_cur:  # vela debe ser alcista
                    continue

                if o_cur <= l_prev and c_cur >= h_prev:
                    avg_vol   = float(volume.tail(51).iloc[:-1].mean())
                    curr_vol  = float(volume.iloc[i])
                    vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0

                    stop_loss = max(float(low.iloc[i]) * 0.99, float(close.iloc[-1]) * 0.95)
                    body_pct  = (c_cur - o_cur) / c_cur

                    confidence = min(1.0,
                        0.50 * min(1.0, body_pct * 5)
                        + 0.30 * min(1.0, vol_ratio / self.volume_breakout_mult)
                        + 0.20 * (1.0 if index_bars is not None else 0.5)
                    )

                    logger.info(
                        "[Pattern] ENGULFING | body=%.1f%% | vol=%.2fx | conf=%.2f",
                        body_pct * 100, vol_ratio, confidence,
                    )

                    return PatternSignal(
                        pattern_name="ENGULFING_CANDLE",
                        strategy="CORTO_PLAZO",
                        breakout_level=round(c_cur, 4),
                        stop_loss_level=round(stop_loss, 4),
                        confidence=round(confidence, 3),
                        pattern_depth_pct=round(body_pct, 4),
                        volume_confirmed=vol_ratio >= 1.2,
                        days_in_formation=1,
                    )
            except (IndexError, ZeroDivisionError):
                continue

        return None

    # ─────────────────────────────────────────────────────────────────
    # SEÑALES DE SALIDA
    # ─────────────────────────────────────────────────────────────────

    def detect_exit_signal(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        strategy: str = "BALLENERA",
    ) -> Optional[ExitSignal]:
        """
        Evalúa las 3 causas de salida de Whale Capital:
        cambio de tendencia, euforia, distribución.
        """
        for checker in (
            self._check_trend_change,
            self._check_euphoria,
            self._check_distribution,
        ):
            sig = checker(bars)
            if sig:
                return sig
        return None

    def _check_trend_change(self, bars: pd.DataFrame) -> Optional[ExitSignal]:
        """
        Cambio de tendencia: cross-under de SMA50 tras haber estado por encima.
        Para medio plazo, el mínimo relevante es cuando el precio toca la SMA50.
        """
        close = bars["close"].astype(float)
        sma50 = close.rolling(50).mean()

        if len(close) < 55:
            return None

        was_above = (close.tail(30) > sma50.tail(30)).any()
        if not was_above:
            return None

        c_cur, c_prev   = float(close.iloc[-1]),  float(close.iloc[-2])
        s_cur, s_prev   = float(sma50.iloc[-1]), float(sma50.iloc[-2])

        if c_prev > s_prev and c_cur < s_cur:
            logger.warning(
                "[Pattern] EXIT TREND_CHANGE | close=%.4f cruzó bajo SMA50=%.4f",
                c_cur, s_cur,
            )
            return ExitSignal(reason="TREND_CHANGE", urgency="END_OF_DAY", price=c_cur)

        return None

    def _check_euphoria(self, bars: pd.DataFrame) -> Optional[ExitSignal]:
        """
        Euforia: GAP alcista ≥ 5% a máximos históricos con volumen ≥ 2× media
        tras una larga tendencia alcista (≥ 40 días sobre SMA50).
        """
        close  = bars["close"].astype(float)
        high   = bars["high"].astype(float)
        open_  = bars["open"].astype(float)
        volume = bars["volume"].astype(float)

        if len(bars) < 100:
            return None

        gap_pct = (float(open_.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2])
        if gap_pct < 0.05:
            return None

        if float(close.iloc[-1]) < float(high.max()) * 0.99:
            return None

        avg_vol  = float(volume.tail(51).iloc[:-1].mean())
        vol_mult = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0
        if vol_mult < self.volume_climactic_mult:
            return None

        sma50       = close.rolling(50).mean()
        days_above  = int((close.tail(60) > sma50.tail(60)).sum())
        if days_above < 40:
            return None

        logger.warning(
            "[Pattern] EXIT EUPHORIA | gap=%.1f%% | vol=%.1fx atip",
            gap_pct * 100, vol_mult,
        )
        return ExitSignal(
            reason="EUPHORIA",
            urgency="IMMEDIATE",
            price=float(close.iloc[-1]),
        )

    def _check_distribution(self, bars: pd.DataFrame) -> Optional[ExitSignal]:
        """
        Distribución: tendencia → lateral en ATH → GAP bajista con volumen climático.
        """
        close  = bars["close"].astype(float)
        open_  = bars["open"].astype(float)
        volume = bars["volume"].astype(float)

        if len(bars) < 60:
            return None

        gap_pct = (float(open_.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2])
        if gap_pct > -0.03:
            return None

        if float(close.iloc[-2]) < float(close.tail(20).max()) * 0.97:
            return None

        avg_vol  = float(volume.tail(51).iloc[:-1].mean())
        vol_mult = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0
        if vol_mult < self.volume_climactic_mult:
            return None

        logger.warning(
            "[Pattern] EXIT DISTRIBUTION | gap_down=%.1f%% | vol=%.1fx",
            abs(gap_pct) * 100, vol_mult,
        )
        return ExitSignal(
            reason="DISTRIBUTION",
            urgency="IMMEDIATE",
            price=float(close.iloc[-1]),
        )
