"""
data/market_data.py
Fase 6 — Acceso a Datos de Mercado.

Precios en vivo  : eToro REST API  — GET /market-data/instruments/rates (1 ID por request)
Velas históricas : yfinance        — el endpoint de candles de eToro no existe en la API
                                     pública v1 (/history/candles/buy/* → 404).
Detección de gaps: alerta si gap > 5% entre cierre anterior y apertura actual.

Mapa de símbolos: settings.yaml → broker.instrument_symbols
  Ejemplo: {4238: "VOO", 14328: "MELI", 9408: "PLTR", 6218: "APP", 2488: "DAVE"}
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from broker.etoro_client import EToroClient

logger = logging.getLogger(__name__)

GAP_ALERT_PCT   = 0.05     # gap > 5% → registrar evento
DEFAULT_COUNT   = 756      # ~3 años de velas diarias (yfinance: period="3y")
YF_PERIOD_MAP   = {        # count → yfinance period string
    252: "1y",
    504: "2y",
    756: "3y",
}
YF_DEFAULT_PERIOD = "3y"


class MarketData:
    """
    Acceso a datos de mercado.

    Parámetros
    ----------
    client           : EToroClient  — instancia autenticada (para precios en vivo)
    instrument_symbols : dict        — {instrument_id: yahoo_ticker}
                         Cargado desde settings['broker']['instrument_symbols']
    """

    def __init__(
        self,
        client: EToroClient,
        instrument_symbols: Optional[Dict[int, str]] = None,
    ):
        self.client             = client
        self._symbol_map: Dict[int, str] = {
            int(k): v for k, v in (instrument_symbols or {}).items()
        }
        self._candle_cache: Dict[int, pd.DataFrame] = {}
        self._rates_cache:  Dict[str, Dict]         = {}

    # ------------------------------------------------------------------
    # Velas históricas  (fuente: yfinance)
    # ------------------------------------------------------------------

    def get_historical_candles(
        self,
        instrument_id: int,
        count: int = DEFAULT_COUNT,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Obtiene velas diarias históricas vía yfinance.

        El endpoint de eToro para candles históricos no está disponible en la
        API pública v1.  Este método resuelve el ticker Yahoo Finance a partir
        de instrument_symbols (settings.yaml → broker.instrument_symbols) y
        descarga los datos con yfinance.

        Parámetros
        ----------
        instrument_id : int   eToro instrument ID (ej. 4238)
        count         : int   Número de velas aproximado (default 756 ≈ 3 años)
        use_cache     : bool  Si True, retorna caché si existe

        Retorna
        -------
        pd.DataFrame con columnas: open, high, low, close, volume
        Índice: DatetimeIndex tz-aware (UTC)
        """
        if use_cache and instrument_id in self._candle_cache:
            return self._candle_cache[instrument_id]

        ticker = self._symbol_map.get(instrument_id)
        if not ticker:
            logger.error(
                "[MarketData] No hay símbolo mapeado para instrID=%d. "
                "Agregar en settings.yaml → broker.instrument_symbols",
                instrument_id,
            )
            return pd.DataFrame()

        period = YF_PERIOD_MAP.get(count, YF_DEFAULT_PERIOD)
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error(
                "[MarketData] yfinance falló para %s (instrID=%d): %s",
                ticker, instrument_id, exc,
            )
            return pd.DataFrame()

        df = self._parse_yf_dataframe(raw, ticker)

        if df.empty:
            logger.warning(
                "[MarketData] yfinance devolvió datos vacíos para %s (instrID=%d)",
                ticker, instrument_id,
            )
            return df

        self._detect_gaps(df, instrument_id)
        self._candle_cache[instrument_id] = df

        logger.info(
            "[MarketData] Candles yfinance | instrID=%d (%s) | filas=%d | %s → %s",
            instrument_id, ticker, len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
        return df

    def _parse_yf_dataframe(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Normaliza el DataFrame de yfinance al formato estándar del sistema:
        columnas lowercase (open, high, low, close, volume), índice UTC.

        yfinance con auto_adjust=True devuelve MultiIndex en columnas cuando
        se descarga un solo ticker: ('Close', 'VOO'), ('Open', 'VOO'), etc.
        """
        if raw is None or raw.empty:
            return pd.DataFrame()

        # Aplanar MultiIndex si existe
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)

        # Normalizar nombres de columnas
        raw.columns = [c.lower() for c in raw.columns]

        needed = {"open", "high", "low", "close", "volume"}
        missing = needed - set(raw.columns)
        if missing:
            logger.warning("[MarketData] yfinance: columnas faltantes %s", missing)
            return pd.DataFrame()

        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df = df[df["close"] > 0].dropna(subset=["close"])

        # Asegurar índice tz-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        return df

    # ------------------------------------------------------------------
    # Precios en tiempo real
    # ------------------------------------------------------------------

    def get_current_rates(self, instrument_ids: List[int]) -> Dict[int, Dict]:
        """
        Obtiene precios bid/ask actuales para una lista de instrumentos.

        GET /market-data/instruments/rates?instrumentIds=4238,2488,...

        Retorna
        -------
        dict {instrumentId: {"bid": float, "ask": float, "mid": float}}
        """
        raw    = self.client.get_instrument_rates(instrument_ids)
        result: Dict[int, Dict] = {}

        # Normalizar respuesta (eToro puede usar distintos wrappers)
        rates_list = []
        if isinstance(raw, list):
            rates_list = raw
        elif isinstance(raw, dict):
            rates_list = (
                raw.get("rates") or
                raw.get("data")  or
                raw.get("instruments") or
                list(raw.values())
            )
            if isinstance(rates_list, dict):
                rates_list = [{"instrumentId": k, **v} for k, v in rates_list.items()]

        for item in rates_list:
            try:
                # eToro devuelve "instrumentID" (D mayúscula)
                iid = int(
                    item.get("instrumentID") or
                    item.get("instrumentId") or
                    item.get("id") or 0
                )
                bid = float(item.get("bid") or item.get("sellPrice") or 0)
                ask = float(item.get("ask") or item.get("buyPrice")  or 0)
                mid = (bid + ask) / 2 if (bid + ask) > 0 else float(
                    item.get("rate") or item.get("price") or 0
                )
                if iid > 0 and mid > 0:
                    result[iid] = {"bid": bid, "ask": ask, "mid": mid}
                    # Actualizar caché de strings (para PositionTracker)
                    self._rates_cache[str(iid)] = {"bid": bid, "ask": ask, "mid": mid}
            except (TypeError, ValueError):
                continue

        logger.debug(
            "[MarketData] Rates actualizados: %d instrumentos", len(result)
        )
        return result

    def get_rates_cache(self) -> Dict[str, Dict]:
        """Retorna el caché de rates en formato {str(instrID): {bid,ask,mid}}."""
        return dict(self._rates_cache)

    # ------------------------------------------------------------------
    # Última vela cerrada
    # ------------------------------------------------------------------

    def get_latest_candle(self, instrument_id: int) -> Optional[pd.Series]:
        """
        Retorna la última vela diaria cerrada para feature engineering.
        Llama al endpoint con count=2 para garantizar que la última es completa.
        """
        df = self.get_historical_candles(instrument_id, count=2, use_cache=False)
        if df.empty:
            return None
        return df.iloc[-1]

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    def calculate_spread_pct(self, instrument_id: int) -> float:
        """
        Calcula spread porcentual: (ask - bid) / mid.
        Retorna float. El RiskManager rechaza señales con spread > 0.5%.
        """
        rates = self.get_current_rates([instrument_id])
        if instrument_id not in rates:
            logger.warning(
                "[MarketData] No se obtuvo rate para instrumento %d", instrument_id
            )
            return 1.0   # spread alto forzado → RiskManager rechazará

        r   = rates[instrument_id]
        bid = r["bid"]
        ask = r["ask"]
        mid = r["mid"]

        if mid <= 0:
            return 1.0

        spread = (ask - bid) / mid
        if spread > 0.005:
            logger.warning(
                "[MarketData] SPREAD EXCESIVO instrID=%d: %.3f%% (umbral 0.5%%)",
                instrument_id, spread * 100,
            )
        return spread

    # ------------------------------------------------------------------
    # Detección de gaps
    # ------------------------------------------------------------------

    def _detect_gaps(self, df: pd.DataFrame, instrument_id: int) -> None:
        """
        Detecta gaps >5% entre cierre anterior y apertura actual.
        Registra cada evento como WARNING con detalles.
        """
        if len(df) < 2:
            return

        prev_close = df["close"].shift(1)
        gap_pct    = (df["open"] - prev_close) / prev_close.replace(0, float("nan"))
        large_gaps = gap_pct[gap_pct.abs() > GAP_ALERT_PCT].dropna()

        for ts, gap in large_gaps.items():
            logger.warning(
                "[MarketData] GAP DETECTADO | instrID=%d | fecha=%s | gap=%.2f%% "
                "| prev_close=%.4f | open=%.4f",
                instrument_id,
                ts.date() if hasattr(ts, "date") else ts,
                gap * 100,
                float(prev_close.loc[ts]),
                float(df.loc[ts, "open"]),
            )

    # ------------------------------------------------------------------
    # Actualización incremental de candles
    # ------------------------------------------------------------------

    def update_candles_cache(
        self,
        instrument_id: int,
        new_candle: Dict,
    ) -> pd.DataFrame:
        """
        Añade una nueva vela al caché existente (para actualización diaria
        sin recargar 756 velas completas).
        """
        existing = self._candle_cache.get(instrument_id, pd.DataFrame())
        try:
            ts    = datetime.fromisoformat(
                str(new_candle.get("timestamp", "")).replace("Z", "+00:00")
            )
            row = pd.DataFrame([{
                "open":   float(new_candle.get("open",   0)),
                "high":   float(new_candle.get("high",   0)),
                "low":    float(new_candle.get("low",    0)),
                "close":  float(new_candle.get("close",  0)),
                "volume": float(new_candle.get("volume", 0)),
            }], index=pd.DatetimeIndex([ts], tz="UTC"))

            if existing.empty:
                self._candle_cache[instrument_id] = row
            else:
                updated = pd.concat([existing, row])
                updated = updated[~updated.index.duplicated(keep="last")]
                self._candle_cache[instrument_id] = updated

        except Exception as exc:
            logger.error(
                "[MarketData] Error actualizando caché para instrID=%d: %s",
                instrument_id, exc,
            )

        return self._candle_cache.get(instrument_id, pd.DataFrame())

    def clear_cache(self, instrument_id: Optional[int] = None) -> None:
        """Limpia el caché de velas. Sin argumento limpia todo."""
        if instrument_id is not None:
            self._candle_cache.pop(instrument_id, None)
        else:
            self._candle_cache.clear()
        logger.debug("[MarketData] Caché limpiado.")
