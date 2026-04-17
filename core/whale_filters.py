"""
core/whale_filters.py
Whale Capital — Filtros de Fuerza Relativa y Market Leaders.

Definición matemática:
  RS_score(t, N) = (close_asset(t) / close_asset(t-N))
                 / (close_index(t) / close_index(t-N))

  RS > 1.0 → sube más (o cae menos) que el índice → Outperformer
  RS < 1.0 → sube menos (o cae más) que el índice → Rezagado (descartar)

Umbrales:
  RS ≥ 1.15 → Market Leader (candidato a entrar)
  RS ≤ 0.85 → Rezagado (descarte inmediato)
  0.85 < RS < 1.15 → Neutral (evaluar patrón antes de decidir)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RS_LEADER_THRESHOLD  = 1.15   # +15% sobre el índice → Market Leader
RS_LAGGARD_THRESHOLD = 0.85   # -15% bajo el índice  → Descarte inmediato
RS_WINDOW            = 252    # ventana de 1 año bursátil


class RelativeStrengthFilter:
    """
    Calcula la Fuerza Relativa de un activo vs. un índice de referencia.

    RS > 1.0 = empresa que sube más o cae menos que el mercado.
    Esto es la "huella del tiburón": el institucional que acumula en silencio
    imprime una demanda plus que mantiene el precio resistente.
    """

    def __init__(
        self,
        window: int = RS_WINDOW,
        leader_threshold: float = RS_LEADER_THRESHOLD,
        laggard_threshold: float = RS_LAGGARD_THRESHOLD,
    ):
        self.window            = window
        self.leader_threshold  = leader_threshold
        self.laggard_threshold = laggard_threshold

    def compute_rs_score(
        self,
        asset_bars: pd.DataFrame,
        index_bars: pd.DataFrame,
    ) -> float:
        """
        RS_score = retorno_activo(N) / retorno_índice(N).
        Retorna 1.0 (neutral) si hay datos insuficientes.
        """
        try:
            asset_close = asset_bars["close"].dropna()
            index_close = index_bars["close"].dropna()

            aligned = pd.concat(
                {"asset": asset_close, "index": index_close}, axis=1
            ).dropna()

            if len(aligned) < self.window:
                logger.warning(
                    "[WhaleRS] Datos insuficientes para RS: %d barras (mínimo %d)",
                    len(aligned), self.window,
                )
                return 1.0

            w = aligned.tail(self.window)
            asset_ret = float(w["asset"].iloc[-1] / w["asset"].iloc[0])
            index_ret = float(w["index"].iloc[-1] / w["index"].iloc[0])

            return round(asset_ret / index_ret, 4) if index_ret != 0 else 1.0

        except Exception as exc:
            logger.error("[WhaleRS] Error calculando RS: %s", exc)
            return 1.0

    def compute_rs_series(
        self,
        asset_bars: pd.DataFrame,
        index_bars: pd.DataFrame,
    ) -> pd.Series:
        """RS histórico rolling — útil para ver la trayectoria de la fuerza."""
        asset_close = asset_bars["close"].dropna()
        index_close = index_bars["close"].dropna()

        aligned = pd.concat(
            {"asset": asset_close, "index": index_close}, axis=1
        ).dropna()

        if len(aligned) < self.window:
            return pd.Series(dtype=float)

        asset_ret = aligned["asset"] / aligned["asset"].shift(self.window)
        index_ret = aligned["index"] / aligned["index"].shift(self.window)
        return (asset_ret / index_ret.replace(0, np.nan)).dropna()

    def is_market_leader(
        self,
        asset_bars: pd.DataFrame,
        index_bars: pd.DataFrame,
    ) -> Tuple[bool, float]:
        """
        (is_leader, rs_score).
        is_leader=True si RS >= leader_threshold.
        """
        rs = self.compute_rs_score(asset_bars, index_bars)
        is_leader = rs >= self.leader_threshold
        if not is_leader:
            logger.debug("[WhaleRS] RS=%.3f < %.2f — no es Market Leader", rs, self.leader_threshold)
        return is_leader, rs

    def is_laggard(self, rs_score: float) -> bool:
        """True = rezagado claro → descarte inmediato."""
        return rs_score < self.laggard_threshold

    def rs_during_correction(
        self,
        asset_bars: pd.DataFrame,
        index_bars: pd.DataFrame,
        days: int = 20,
    ) -> float:
        """
        RS durante los últimos N días (período correctivo del mercado).
        Este es el RS más revelador — la "pista del tiburón" en días negativos.
        RS > 1.0 en corrección = acumulación institucional confirmada.
        """
        asset_close = asset_bars["close"].dropna().tail(days)
        index_close = index_bars["close"].dropna().tail(days)

        aligned = pd.concat(
            {"asset": asset_close, "index": index_close}, axis=1
        ).dropna()

        if len(aligned) < 5:
            return 1.0

        a = float(aligned["asset"].iloc[-1] / aligned["asset"].iloc[0])
        i = float(aligned["index"].iloc[-1] / aligned["index"].iloc[0])
        return round(a / i, 4) if i != 0 else 1.0


class TrendFilter:
    """
    Filtro de tendencia multi-nivel (activo + sector + mercado).

    Regla Ballenera: el activo, el sector Y el índice deben estar en tendencia alcista.
    La inercia de la tendencia es una de las fuerzas más poderosas — nunca luches contra ella.
    """

    @staticmethod
    def is_uptrend(bars: pd.DataFrame, strict: bool = False) -> bool:
        """
        True si el activo está en tendencia alcista.
        strict=True : close > EMA20 > EMA50 > EMA200
        strict=False: close > EMA50 AND EMA50 > EMA200 (suficiente para filtrado)
        """
        close = bars["close"].astype(float)
        if len(close) < 201:
            return False

        ema20  = close.ewm(span=20,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        c, e20, e50, e200 = (
            float(close.iloc[-1]), float(ema20.iloc[-1]),
            float(ema50.iloc[-1]), float(ema200.iloc[-1]),
        )

        if strict:
            return c > e20 and e20 > e50 and e50 > e200
        return c > e50 and e50 > e200

    @staticmethod
    def days_above_sma50(bars: pd.DataFrame) -> int:
        """Días consecutivos con cierre > SMA50 (mide la salud de la tendencia)."""
        close = bars["close"].astype(float)
        sma50 = close.rolling(50).mean()
        above = close > sma50
        count = 0
        for v in reversed(above.values):
            if v:
                count += 1
            else:
                break
        return count

    @staticmethod
    def ema50_slope(bars: pd.DataFrame, lookback: int = 20) -> float:
        """Pendiente de EMA50 en los últimos N días. Positiva = tendencia acelerando."""
        close = bars["close"].astype(float)
        ema50 = close.ewm(span=50, adjust=False).mean().tail(lookback)
        if len(ema50) < 2:
            return 0.0
        return float((ema50.iloc[-1] - ema50.iloc[0]) / ema50.iloc[0])

    @staticmethod
    def is_market_healthy(index_bars: pd.DataFrame) -> bool:
        """True si el índice de referencia (SPY/QQQ) está en tendencia alcista."""
        return TrendFilter.is_uptrend(index_bars, strict=False)


class VolumeAnalyzer:
    """
    "El precio manda, el volumen indica su relevancia." — Whale Capital

    El volumen confirma la convicción detrás de cada movimiento.
    Sin volumen, cualquier breakout es sospechoso.
    """

    @staticmethod
    def volume_ratio(bars: pd.DataFrame, window: int = 50) -> float:
        """Volumen actual / media de N días. >1 = mayor actividad."""
        if len(bars) < window + 1:
            return 1.0
        avg = float(bars["volume"].tail(window + 1).iloc[:-1].mean())
        cur = float(bars["volume"].iloc[-1])
        return round(cur / avg, 2) if avg > 0 else 1.0

    @staticmethod
    def is_climactic_volume(bars: pd.DataFrame, threshold: float = 2.0) -> bool:
        """True si el volumen actual ≥ threshold × media 50d (volumen climático)."""
        return VolumeAnalyzer.volume_ratio(bars) >= threshold

    @staticmethod
    def is_volume_dry(bars: pd.DataFrame, days: int = 5, ratio: float = 0.50) -> bool:
        """
        True si el volumen promedio de los últimos N días es < ratio × media 50d.
        Volumen seco durante el handle = acumulación silenciosa (señal alcista).
        """
        if len(bars) < 56:
            return False
        avg_50     = float(bars["volume"].tail(56).iloc[:-5].mean())
        recent_avg = float(bars["volume"].tail(days).mean())
        return (recent_avg / avg_50 < ratio) if avg_50 > 0 else False
