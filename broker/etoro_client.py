"""
broker/etoro_client.py
Fase 6 — Cliente eToro Public API v1.

Base URL : https://public-api.etoro.com/api/v1
Auth     : x-api-key  → os.environ["ETORO_API_KEY"]
           x-user-key → os.environ["ETORO_USER_KEY"]
           x-request-id → UUID por petición

Sin SDK oficial. Implementación directa con requests.
Exponential backoff: 3 intentos, delays 1s → 2s → 4s.
"""

import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

logger = logging.getLogger(__name__)

BASE_URL    = "https://public-api.etoro.com/api/v1"
CID_REAL    = 34044505
MAX_RETRIES = 3
BACKOFF     = [1, 2, 4]          # segundos entre reintentos


class EToroAPIError(Exception):
    """Excepción controlada para errores de la API de eToro."""
    def __init__(self, status_code: int, message: str, endpoint: str):
        self.status_code = status_code
        self.endpoint    = endpoint
        super().__init__(f"[{status_code}] {endpoint} — {message}")


class EToroClient:
    """
    Wrapper completo para la eToro Public REST API v1.

    Parámetros
    ----------
    environment : str
        "real" o "demo". Si es "real" solicita confirmación manual por consola.
    base_url    : str
        Sobreescribe BASE_URL (útil para tests con mock server).
    """

    def __init__(self, environment: str = "demo", base_url: str = BASE_URL):
        self.environment = environment
        self.base_url    = base_url
        self._cid: Optional[int] = None

        self._validate_env_vars()

        if environment == "real":
            self._confirm_live_trading()

    # ------------------------------------------------------------------
    # Validación de entorno
    # ------------------------------------------------------------------

    def _validate_env_vars(self) -> None:
        missing = [v for v in ("ETORO_API_KEY", "ETORO_USER_KEY") if not os.environ.get(v)]
        if missing:
            logger.critical("Variables de entorno faltantes: %s", missing)
            sys.exit(1)

    def _confirm_live_trading(self) -> None:
        confirm = input(
            "\n⚠️  LIVE TRADING EN ETORO — CUENTA REAL\n"
            "   Capital: $560.05 USD\n"
            "   Escribe 'CONFIRMO' para continuar: "
        ).strip()
        if confirm != "CONFIRMO":
            logger.info("Confirmación rechazada por el usuario. Saliendo.")
            sys.exit(0)
        logger.warning("LIVE TRADING CONFIRMADO — Entorno: REAL")

    # ------------------------------------------------------------------
    # Cabeceras
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "x-request-id": str(uuid.uuid4()),
            "x-api-key":    os.environ["ETORO_API_KEY"],
            "x-user-key":   os.environ["ETORO_USER_KEY"],
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }

    # ------------------------------------------------------------------
    # HTTP con retry + exponential backoff
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: Dict) -> Any:
        return self._request("POST", path, json=body)

    def _delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate(BACKOFF, start=1):
            try:
                resp = requests.request(
                    method, url,
                    headers=self._headers(),
                    timeout=15,
                    **kwargs,
                )
                if resp.status_code == 401:
                    raise EToroAPIError(401, "Credenciales inválidas o expiradas", path)
                if resp.status_code == 403:
                    raise EToroAPIError(403, "Acceso denegado", path)
                resp.raise_for_status()
                return resp.json()

            except (ConnectionError, Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "[eToro] %s %s — intento %d/%d — red: %s. Reintentando en %ds...",
                    method, path, attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)

            except HTTPError as exc:
                last_exc = exc
                sc = exc.response.status_code if exc.response else 0
                if sc in (400, 422):
                    # Errores de validación — no reintentar
                    raise EToroAPIError(sc, str(exc), path)
                logger.warning(
                    "[eToro] %s %s — HTTP %d intento %d/%d. Reintentando en %ds...",
                    method, path, sc, attempt, MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except EToroAPIError:
                raise

        raise ConnectionError(
            f"eToro API no disponible tras {MAX_RETRIES} intentos: {path}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Verifica conectividad y valida CID de la cuenta.
        Retorna True si CID == 34044505.
        """
        identity = self.get_identity()
        cid = identity.get("realCid") or identity.get("realCID") or identity.get("cid") or identity.get("id")
        self._cid = int(cid) if cid else None

        if self._cid != CID_REAL:
            logger.error(
                "[eToro] Health check FALLIDO — CID esperado: %d, obtenido: %s",
                CID_REAL, self._cid,
            )
            return False

        logger.info("[eToro] Health check OK — CID: %d | Entorno: %s", self._cid, self.environment)
        return True

    # ------------------------------------------------------------------
    # Endpoints de cuenta
    # ------------------------------------------------------------------

    def get_identity(self) -> Dict:
        """GET /me — Info de cuenta (CID, nombre, estado)."""
        return self._get("/me")

    def get_portfolio(self) -> Dict:
        """
        GET /trading/info/portfolio
        Retorna posiciones, crédito disponible y órdenes pendientes.
        """
        return self._get("/trading/info/portfolio")

    def get_pnl(self) -> Dict:
        """
        GET /trading/info/real/pnl
        Equity y P&L en tiempo real.
        """
        return self._get("/trading/info/real/pnl")

    def get_trade_history(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict:
        """
        GET /trading/info/trade/history
        Historial de operaciones cerradas.
        """
        return self._get("/trading/info/trade/history", params={
            "page":     page,
            "pageSize": page_size,
        })

    # ------------------------------------------------------------------
    # Endpoints de mercado
    # ------------------------------------------------------------------

    def get_instrument_rates(self, instrument_ids: List[int]) -> Dict:
        """
        GET /market-data/instruments/rates?instrumentIds={id}
        La API pública v1 de eToro devuelve 500 con múltiples IDs en una sola
        petición.  Se llama una vez por instrumento y se fusionan los resultados
        en el formato estándar {"rates": [...]}.
        """
        merged: List[Dict] = []
        for iid in instrument_ids:
            try:
                resp = self._get(
                    "/market-data/instruments/rates",
                    params={"instrumentIds": str(iid)},
                )
                merged.extend(resp.get("rates", []))
            except Exception as exc:
                logger.warning(
                    "[eToro] get_instrument_rates falló para instrID=%d: %s", iid, exc
                )
        return {"rates": merged}

    def get_historical_candles(
        self,
        instrument_id: int,
        count: int = 756,
        interval: int = 1440,
    ) -> Dict:
        """
        NOTA: el endpoint /market-data/instruments/{id}/history/candles/buy/
        no está disponible en la API pública v1 de eToro (retorna 404).
        Este método se mantiene por compatibilidad pero siempre lanza
        NotImplementedError.  Usar MarketData.get_historical_candles()
        que obtiene los datos via yfinance.
        """
        raise NotImplementedError(
            f"El endpoint de candles históricos de eToro no está disponible "
            f"en la API pública v1 (instrID={instrument_id}). "
            f"Usar MarketData.get_historical_candles() con yfinance."
        )

    def search_instrument(self, query: str) -> Dict:
        """
        GET /market-data/search?query={q}
        Busca instrumentos por ticker o nombre.
        """
        return self._get("/market-data/search", params={"query": query})

    # ------------------------------------------------------------------
    # Endpoints de ejecución
    # ------------------------------------------------------------------

    def open_market_order(self, body: Dict) -> Dict:
        """
        POST /trading/execution/market-open-orders/by-amount
        Abre una posición de mercado. stopLossRate es OBLIGATORIO.
        """
        return self._post("/trading/execution/market-open-orders/by-amount", body)

    def close_position(self, position_id: int, units: float) -> Dict:
        """
        POST /trading/execution/market-close-orders/positions/{positionId}
        Cierra total o parcialmente una posición por units.
        """
        return self._post(
            f"/trading/execution/market-close-orders/positions/{position_id}",
            {"units": units},
        )

    def open_limit_order(self, body: Dict) -> Dict:
        """
        POST /trading/execution/limit-orders
        Orden límite (Market-if-Touched).
        """
        return self._post("/trading/execution/limit-orders", body)

    def cancel_limit_order(self, order_id: int) -> Dict:
        """
        DELETE /trading/execution/limit-orders/{orderId}
        Cancela una orden límite pendiente.
        """
        return self._delete(f"/trading/execution/limit-orders/{order_id}")
