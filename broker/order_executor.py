"""
broker/order_executor.py
Fase 6 — Ejecución de Órdenes vía eToro REST API.

Principios:
  - LONG ONLY. isBuy siempre True.
  - stopLossRate OBLIGATORIO en toda apertura.
  - Leverage SIEMPRE 1.
  - Modificación de stop = cerrar + reabrir con nuevo stopLossRate (eToro no permite PATCH).
  - Trazabilidad: trade_id vincula signal → risk_decision → order_response → positionId.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from broker.etoro_client import EToroClient, EToroAPIError
from core.regime_strategies import Signal

logger = logging.getLogger(__name__)

DEFAULT_STOP_PCT = 0.05
MIN_AMOUNT_USD   = 100.0


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    trade_id: str                        # UUID de trazabilidad
    success: bool
    position_id: Optional[int]
    instrument_id: int
    amount_usd: float
    stop_loss_rate: float
    open_rate: Optional[float]
    timestamp: datetime
    raw_response: Dict[str, Any]
    error: str = ""


@dataclass
class CloseResult:
    trade_id: str
    success: bool
    position_id: int
    units_closed: float
    timestamp: datetime
    raw_response: Dict[str, Any]
    error: str = ""


@dataclass
class StopAdjustResult:
    trade_id: str
    success: bool
    old_position_id: int
    new_position_id: Optional[int]
    old_stop: float
    new_stop: float
    amount_usd: float
    timestamp: datetime
    error: str = ""


# ---------------------------------------------------------------------------
# Order Executor
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Ejecuta órdenes de apertura, cierre y ajuste de stop en eToro.

    Parámetros
    ----------
    client : EToroClient
        Instancia autenticada del cliente eToro.
    dry_run : bool
        Si True, simula las órdenes sin enviarlas (para --dry-run CLI).
    """

    def __init__(self, client: EToroClient, dry_run: bool = False):
        self.client  = client
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Apertura de posición
    # ------------------------------------------------------------------

    def submit_order(self, signal: Signal) -> OrderResult:
        """
        Abre una posición de mercado a partir de una Signal aprobada por RiskManager.

        Flujo:
          1. Construir body con stopLossRate obligatorio.
          2. POST /trading/execution/market-open-orders/by-amount.
          3. Extraer positionId de la respuesta.
          4. Loguear con trade_id para trazabilidad completa.
        """
        trade_id      = str(uuid.uuid4())
        instrument_id = int(signal.symbol)
        amount_usd    = signal.position_size_usd
        stop_rate     = signal.stop_loss
        tp_rate       = signal.take_profit

        if amount_usd < MIN_AMOUNT_USD:
            return OrderResult(
                trade_id=trade_id, success=False,
                position_id=None, instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_rate,
                open_rate=None, timestamp=datetime.utcnow(),
                raw_response={},
                error=f"Monto ${amount_usd:.2f} inferior al mínimo eToro ${MIN_AMOUNT_USD}",
            )

        if not stop_rate or stop_rate <= 0.001:
            return OrderResult(
                trade_id=trade_id, success=False,
                position_id=None, instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_rate or 0,
                open_rate=None, timestamp=datetime.utcnow(),
                raw_response={},
                error="stopLossRate ausente o inválido — orden rechazada",
            )

        body = {
            "instrumentId":    instrument_id,
            "isBuy":           True,
            "amount":          round(amount_usd, 2),
            "leverage":        1,
            "stopLossRate":    round(stop_rate, 4),
            "isTslEnabled":    False,
        }
        if tp_rate and tp_rate > 0:
            body["takeProfitRate"] = round(tp_rate, 4)

        logger.info(
            "[OrderExec] OPEN | trade_id=%s | instrID=%d | amount=$%.2f | SL=%.4f | %s",
            trade_id, instrument_id, amount_usd, stop_rate,
            "DRY_RUN" if self.dry_run else "REAL",
        )

        if self.dry_run:
            simulated_pos_id = int(str(instrument_id) + "0000")
            return OrderResult(
                trade_id=trade_id, success=True,
                position_id=simulated_pos_id,
                instrument_id=instrument_id,
                amount_usd=amount_usd,
                stop_loss_rate=stop_rate,
                open_rate=signal.entry_price,
                timestamp=datetime.utcnow(),
                raw_response={"dry_run": True, "positionId": simulated_pos_id},
            )

        try:
            resp       = self.client.open_market_order(body)
            pos_id     = self._extract_position_id(resp)
            open_rate  = self._extract_open_rate(resp)

            logger.info(
                "[OrderExec] ABIERTA | trade_id=%s | posID=%s | instrID=%d | "
                "openRate=%.4f | SL=%.4f | $%.2f",
                trade_id, pos_id, instrument_id, open_rate or 0, stop_rate, amount_usd,
            )
            return OrderResult(
                trade_id=trade_id, success=True,
                position_id=pos_id, instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_rate,
                open_rate=open_rate, timestamp=datetime.utcnow(),
                raw_response=resp,
            )

        except EToroAPIError as exc:
            logger.error("[OrderExec] FALLIDA | trade_id=%s | %s", trade_id, exc)
            return OrderResult(
                trade_id=trade_id, success=False,
                position_id=None, instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_rate,
                open_rate=None, timestamp=datetime.utcnow(),
                raw_response={}, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Cierre total
    # ------------------------------------------------------------------

    def close_position(
        self,
        position_id: int,
        units: float,
        reason: str = "MANUAL",
    ) -> CloseResult:
        """
        Cierra totalmente una posición.
        Retorna CloseResult con éxito/fallo.
        """
        trade_id = str(uuid.uuid4())
        logger.info(
            "[OrderExec] CLOSE | trade_id=%s | posID=%d | units=%.6f | reason=%s | %s",
            trade_id, position_id, units, reason,
            "DRY_RUN" if self.dry_run else "REAL",
        )

        if self.dry_run:
            return CloseResult(
                trade_id=trade_id, success=True,
                position_id=position_id, units_closed=units,
                timestamp=datetime.utcnow(),
                raw_response={"dry_run": True},
            )

        try:
            resp = self.client.close_position(position_id, units)
            logger.info("[OrderExec] CERRADA | trade_id=%s | posID=%d", trade_id, position_id)
            return CloseResult(
                trade_id=trade_id, success=True,
                position_id=position_id, units_closed=units,
                timestamp=datetime.utcnow(), raw_response=resp,
            )
        except EToroAPIError as exc:
            logger.error("[OrderExec] CIERRE FALLIDO | trade_id=%s | %s", trade_id, exc)
            return CloseResult(
                trade_id=trade_id, success=False,
                position_id=position_id, units_closed=0,
                timestamp=datetime.utcnow(), raw_response={}, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Cierre parcial
    # ------------------------------------------------------------------

    def partial_close(
        self,
        position_id: int,
        units_to_close: float,
        reason: str = "REBALANCE",
    ) -> CloseResult:
        """Cierra parcialmente una posición por un número de units."""
        trade_id = str(uuid.uuid4())
        logger.info(
            "[OrderExec] PARTIAL_CLOSE | trade_id=%s | posID=%d | units=%.6f | reason=%s",
            trade_id, position_id, units_to_close, reason,
        )

        if self.dry_run:
            return CloseResult(
                trade_id=trade_id, success=True,
                position_id=position_id, units_closed=units_to_close,
                timestamp=datetime.utcnow(), raw_response={"dry_run": True},
            )

        try:
            resp = self.client.close_position(position_id, units_to_close)
            return CloseResult(
                trade_id=trade_id, success=True,
                position_id=position_id, units_closed=units_to_close,
                timestamp=datetime.utcnow(), raw_response=resp,
            )
        except EToroAPIError as exc:
            logger.error("[OrderExec] CIERRE PARCIAL FALLIDO | trade_id=%s | %s", trade_id, exc)
            return CloseResult(
                trade_id=trade_id, success=False,
                position_id=position_id, units_closed=0,
                timestamp=datetime.utcnow(), raw_response={}, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Ajuste de Stop Loss (close + reopen)
    # ------------------------------------------------------------------

    def adjust_stop_loss(
        self,
        position: Dict[str, Any],
        new_stop_rate: float,
        tighten_only: bool = True,
    ) -> StopAdjustResult:
        """
        Ajusta el stop loss de una posición existente mediante:
          1. Cierre total de la posición.
          2. Reapertura con el nuevo stopLossRate.

        CRÍTICO: eToro NO permite modificar stopLossRate de posiciones abiertas.

        Parámetros
        ----------
        position     : dict con campos eToro (positionID, openRate, units, amount, instrumentID)
        new_stop_rate: nuevo stopLossRate (solo se aplica si es MENOR que el actual — tighten)
        tighten_only : si True, rechaza ajustes que amplíen el stop (NUNCA ampliar riesgo)
        """
        trade_id   = str(uuid.uuid4())
        pos_id     = int(position["positionID"])
        instr_id   = int(position["instrumentID"])
        units      = float(position["units"])
        amount     = float(position["amount"])
        open_rate  = float(position.get("openRate", 0))
        old_stop   = float(position.get("stopLossRate", 0))

        if tighten_only and new_stop_rate < old_stop and old_stop > 0.001:
            logger.warning(
                "[OrderExec] STOP_ADJUST rechazado — nuevo stop %.4f < actual %.4f "
                "(solo se permite tighten)",
                new_stop_rate, old_stop,
            )
            return StopAdjustResult(
                trade_id=trade_id, success=False,
                old_position_id=pos_id, new_position_id=None,
                old_stop=old_stop, new_stop=new_stop_rate,
                amount_usd=amount, timestamp=datetime.utcnow(),
                error="TIGHTEN_ONLY: nuevo stop más amplio que el actual",
            )

        logger.info(
            "[OrderExec] STOP_ADJUST | trade_id=%s | posID=%d | instrID=%d | "
            "stop: %.4f → %.4f | $%.2f",
            trade_id, pos_id, instr_id, old_stop, new_stop_rate, amount,
        )

        # Paso 1: cerrar la posición actual
        close_res = self.close_position(pos_id, units, reason="STOP_ADJUST_STEP1")
        if not close_res.success:
            return StopAdjustResult(
                trade_id=trade_id, success=False,
                old_position_id=pos_id, new_position_id=None,
                old_stop=old_stop, new_stop=new_stop_rate,
                amount_usd=amount, timestamp=datetime.utcnow(),
                error=f"CIERRE FALLIDO: {close_res.error}",
            )

        # Paso 2: reabrir con el nuevo stopLossRate
        if self.dry_run:
            new_pos_id = pos_id + 1
            logger.info("[OrderExec] STOP_ADJUST DRY_RUN completado | new_posID=%d", new_pos_id)
            return StopAdjustResult(
                trade_id=trade_id, success=True,
                old_position_id=pos_id, new_position_id=new_pos_id,
                old_stop=old_stop, new_stop=new_stop_rate,
                amount_usd=amount, timestamp=datetime.utcnow(),
            )

        body = {
            "instrumentId": instr_id,
            "isBuy":        True,
            "amount":       round(amount, 2),
            "leverage":     1,
            "stopLossRate": round(new_stop_rate, 4),
            "isTslEnabled": False,
        }
        try:
            resp       = self.client.open_market_order(body)
            new_pos_id = self._extract_position_id(resp)
            logger.info(
                "[OrderExec] STOP_ADJUST completado | old_posID=%d → new_posID=%s | "
                "stop: %.4f → %.4f",
                pos_id, new_pos_id, old_stop, new_stop_rate,
            )
            return StopAdjustResult(
                trade_id=trade_id, success=True,
                old_position_id=pos_id, new_position_id=new_pos_id,
                old_stop=old_stop, new_stop=new_stop_rate,
                amount_usd=amount, timestamp=datetime.utcnow(),
            )
        except EToroAPIError as exc:
            logger.error(
                "[OrderExec] STOP_ADJUST REAPERTURA FALLIDA | trade_id=%s | %s "
                "⚠️  POSICIÓN CERRADA SIN REABRIR — intervención manual requerida",
                trade_id, exc,
            )
            return StopAdjustResult(
                trade_id=trade_id, success=False,
                old_position_id=pos_id, new_position_id=None,
                old_stop=old_stop, new_stop=new_stop_rate,
                amount_usd=amount, timestamp=datetime.utcnow(),
                error=f"REAPERTURA_FALLIDA: {exc}",
            )

    # ------------------------------------------------------------------
    # Orden límite
    # ------------------------------------------------------------------

    def submit_limit_order(
        self,
        instrument_id: int,
        trigger_price: float,
        amount_usd: float,
        stop_loss_rate: float,
    ) -> OrderResult:
        """
        Orden Market-if-Touched (orden límite eToro).
        POST /trading/execution/limit-orders
        """
        trade_id = str(uuid.uuid4())
        body = {
            "instrumentId": instrument_id,
            "isBuy":        True,
            "rate":         round(trigger_price, 4),
            "amount":       round(amount_usd, 2),
            "leverage":     1,
            "stopLossRate": round(stop_loss_rate, 4),
        }
        logger.info(
            "[OrderExec] LIMIT_ORDER | trade_id=%s | instrID=%d | trigger=%.4f | "
            "amount=$%.2f | SL=%.4f",
            trade_id, instrument_id, trigger_price, amount_usd, stop_loss_rate,
        )

        if self.dry_run:
            return OrderResult(
                trade_id=trade_id, success=True,
                position_id=None, instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_loss_rate,
                open_rate=None, timestamp=datetime.utcnow(),
                raw_response={"dry_run": True},
            )

        try:
            resp = self.client.open_limit_order(body)
            return OrderResult(
                trade_id=trade_id, success=True,
                position_id=self._extract_position_id(resp),
                instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_loss_rate,
                open_rate=trigger_price, timestamp=datetime.utcnow(),
                raw_response=resp,
            )
        except EToroAPIError as exc:
            return OrderResult(
                trade_id=trade_id, success=False,
                position_id=None, instrument_id=instrument_id,
                amount_usd=amount_usd, stop_loss_rate=stop_loss_rate,
                open_rate=None, timestamp=datetime.utcnow(),
                raw_response={}, error=str(exc),
            )

    def cancel_limit_order(self, order_id: int) -> bool:
        """Cancela una orden límite. Retorna True si éxito."""
        try:
            self.client.cancel_limit_order(order_id)
            logger.info("[OrderExec] LIMIT_ORDER CANCELADA | orderID=%d", order_id)
            return True
        except EToroAPIError as exc:
            logger.error("[OrderExec] CANCELACIÓN FALLIDA | orderID=%d | %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_position_id(self, response: Dict) -> Optional[int]:
        for key in ("positionId", "positionID", "id"):
            if key in response:
                return int(response[key])
        # Buscar anidado en estructuras comunes de eToro
        for container in ("position", "data", "result"):
            if container in response and isinstance(response[container], dict):
                for key in ("positionId", "positionID", "id"):
                    if key in response[container]:
                        return int(response[container][key])
        logger.warning("[OrderExec] No se encontró positionId en la respuesta: %s", response)
        return None

    def _extract_open_rate(self, response: Dict) -> Optional[float]:
        for key in ("openRate", "rate", "price"):
            if key in response:
                return float(response[key])
        for container in ("position", "data", "result"):
            if container in response and isinstance(response[container], dict):
                for key in ("openRate", "rate", "price"):
                    if key in response[container]:
                        return float(response[container][key])
        return None
