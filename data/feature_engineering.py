"""
data/feature_engineering.py
Fase 2 — Ingeniería de Características para el Motor HMM.

Funciones puras computadas desde datos OHLCV.
Todas las características se normalizan con rolling z-score de 252 periodos.
"""

import numpy as np
import pandas as pd
import ta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Normaliza una serie con z-score rodante."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Características individuales
# ---------------------------------------------------------------------------

def compute_returns(close: pd.Series) -> pd.DataFrame:
    """Retornos logarítmicos a 1, 5 y 20 periodos."""
    log_ret = np.log(close / close.shift(1))
    return pd.DataFrame({
        "ret_1":  log_ret,
        "ret_5":  np.log(close / close.shift(5)),
        "ret_20": np.log(close / close.shift(20)),
    })


def compute_volatility(close: pd.Series) -> pd.DataFrame:
    """Volatilidad realizada (20p) y ratio vol 5p/20p."""
    log_ret = np.log(close / close.shift(1))
    vol_20  = log_ret.rolling(20).std()
    vol_5   = log_ret.rolling(5).std()
    return pd.DataFrame({
        "vol_20":    vol_20,
        "vol_ratio": vol_5 / vol_20.replace(0, np.nan),
    })


def compute_volume_features(volume: pd.Series) -> pd.DataFrame:
    """Z-score de volumen (vs media 50p) y tendencia (pendiente SMA 10p)."""
    vol_mean_50 = volume.rolling(50).mean()
    vol_std_50  = volume.rolling(50).std()
    vol_zscore  = (volume - vol_mean_50) / vol_std_50.replace(0, np.nan)

    sma10 = volume.rolling(10).mean()
    vol_trend = sma10.diff()

    return pd.DataFrame({
        "vol_zscore": vol_zscore,
        "vol_trend":  vol_trend,
    })


def compute_trend_features(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """ADX(14) y pendiente de SMA(50)."""
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    adx = adx_indicator.adx()

    sma50       = close.rolling(50).mean()
    sma50_slope = sma50.diff()

    return pd.DataFrame({
        "adx":        adx,
        "sma50_slope": sma50_slope,
    })


def compute_mean_reversion(close: pd.Series) -> pd.DataFrame:
    """Z-score de RSI(14) y distancia porcentual a SMA(200)."""
    rsi      = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi_mean = rsi.rolling(252, min_periods=126).mean()
    rsi_std  = rsi.rolling(252, min_periods=126).std()
    rsi_zscore = (rsi - rsi_mean) / rsi_std.replace(0, np.nan)

    sma200 = close.rolling(200).mean()
    dist_sma200 = (close - sma200) / sma200.replace(0, np.nan)

    return pd.DataFrame({
        "rsi_zscore":   rsi_zscore,
        "dist_sma200":  dist_sma200,
    })


def compute_momentum(close: pd.Series) -> pd.DataFrame:
    """Rate of Change a 10 y 20 periodos."""
    roc10 = ta.momentum.ROCIndicator(close, window=10).roc()
    roc20 = ta.momentum.ROCIndicator(close, window=20).roc()
    return pd.DataFrame({
        "roc_10": roc10,
        "roc_20": roc20,
    })


def compute_range_features(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """ATR(14) normalizado por precio de cierre."""
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr_norm = atr / close.replace(0, np.nan)
    return pd.DataFrame({"atr_norm": atr_norm})


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, zscore_window: int = 252) -> pd.DataFrame:
    """
    Construye el vector de características completo a partir de un DataFrame OHLCV.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener columnas: open, high, low, close, volume.
    zscore_window : int
        Ventana para normalización rolling (default 252 = 1 año bursátil).

    Retorna
    -------
    pd.DataFrame
        Características normalizadas, sin filas con NaN.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing}")

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    parts = [
        compute_returns(close),
        compute_volatility(close),
        compute_volume_features(volume),
        compute_trend_features(high, low, close),
        compute_mean_reversion(close),
        compute_momentum(close),
        compute_range_features(high, low, close),
    ]

    features = pd.concat(parts, axis=1)

    # Normalización rolling z-score por columna
    for col in features.columns:
        features[col] = _rolling_zscore(features[col], window=zscore_window)

    features.dropna(inplace=True)
    return features


FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_20",
    "vol_20", "vol_ratio",
    "vol_zscore", "vol_trend",
    "adx", "sma50_slope",
    "rsi_zscore", "dist_sma200",
    "roc_10", "roc_20",
    "atr_norm",
]
