"""
broker/position_tracker.py
Fase 6 — Seguimiento de Posiciones via Polling (eToro no tiene WebSocket).

Ciclo de polling: 30 segundos.
Reconcilia portafolio en vivo con state_snapshot.json.
Alerta posiciones con stopLossRate <= 0.001 (sin stop efectivo).
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import schedule

from broker.etoro_client import EToroClient
from core.risk_manager import PortfolioState

logger = logging.getLogger(__name__)

SNAPSHOT_FILE   = "state_snapshot.json"
POLL_INTERVAL   = 30          # segundos
STOP_ALERT_THR  = 0.001       # stopLossRate <= este valor → alerta crítica
NO_STOP_POS_IDS = {3403421461, 3403428716, 3403433830, 3403418899}


class PositionTracker:
    """
    Sincroniza el estado del portafolio eToro cada 30 segundos
    y construye el PortfolioState para el RiskManager.

    Parámetros
    ----------
    client          : EToroClient autenticado.
    on_state_update : callback opcional llamado con el nuevo PortfolioState.
    on_stop_alert   : callback opcional para posiciones sin stop efectivo.
    """

    def __init__(
        self,
        client: EToroClient,
        on_state_update: Optional[Callable[[PortfolioState], None]] = None,
        on_stop_alert:   Optional[Callable[[List[Dict]], None]] = None,
    ):
        self.client          = client
        self.on_state_update = on_state_update
        self.on_stop_alert   = on_stop_alert

        # Estado interno
        self._positions: List[Dict[str, Any]] = []
        self._portfolio_state: Optional[PortfolioState] = None
        self._current_prices: Dict[str, Dict] = {}   # {instrID: {bid, ask, mid}}
        self._entry_regimes: Dict[str, str]   = {}   # positionID → régimen de entrada
        self._last_sync: Optional[datetime]   = None
        self._peak_equity: float              = 0.0
        self._day_start_equity: float         = 0.0
        self._week_start_equity: float        = 0.0
        self._daily_trades: int               = 0
        self._last_order_ts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Inicio y scheduler
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Arranca el loop de polling via schedule.
        Llamar en el hilo principal del bucle de main.py.
        """
        self._sync_positions()   # sincronización inicial inmediata
        schedule.every(POLL_INTERVAL).seconds.do(self._sync_positions)
        logger.info("[PositionTracker] Polling iniciado cada %ds", POLL_INTERVAL)

    def tick(self) -> None:
        """Avanza el scheduler. Llamar en el bucle principal."""
        schedule.run_pending()

    # ------------------------------------------------------------------
    # Sincronización principal
    # ------------------------------------------------------------------

    def _sync_positions(self) -> None:
        """
        Sincroniza posiciones desde eToro y actualiza el estado interno.
        Detecta automáticamente posiciones sin stop efectivo.
        """
        try:
            pnl_data = self.client.get_pnl()
        except Exception as exc:
            logger.error("[PositionTracker] Error al sincronizar: %s", exc)
            return

        # El endpoint /trading/info/real/pnl devuelve clientPortfolio con
        # positions enriquecidas (closeRate, pnL, unitsBaseValueDollars).
        pnl_portfolio = pnl_data.get("clientPortfolio", {}) if pnl_data else {}
        raw_positions = pnl_portfolio.get("positions", [])
        credit        = float(pnl_portfolio.get("credit", 0))

        # Equity real = cash disponible + valor de mercado de todas las posiciones
        equity = self._parse_equity(pnl_portfolio, credit, raw_positions)

        # P&L diario/semanal: rastreado internamente vs equity inicial de sesión
        if self._day_start_equity == 0.0:
            self._day_start_equity = equity
        if self._week_start_equity == 0.0:
            self._week_start_equity = equity
        daily_pnl  = round(equity - self._day_start_equity,  2)
        weekly_pnl = round(equity - self._week_start_equity, 2)

        # Enriquecer posiciones con campos calculados
        enriched = self._enrich_positions(raw_positions)
        self._positions = enriched

        # Actualizar pico de equity
        self._peak_equity = max(self._peak_equity, equity)
        dd_pct = (equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0.0

        # Construir PortfolioState
        invested = sum(float(p.get("amount", 0)) for p in enriched)
        state = PortfolioState(
            equity=equity,
            cash=credit,
            buying_power=credit,
            positions=enriched,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            peak_equity=self._peak_equity,
            drawdown_pct=dd_pct,
            circuit_breaker_status="OK",
            flicker_rate=0.0,
            daily_trades_count=self._daily_trades,
            last_order_by_instrument=dict(self._last_order_ts),
        )
        self._portfolio_state = state
        self._last_sync = datetime.now(timezone.utc)

        # Alertas de stop loss
        alerts = self._check_stop_alerts(enriched)
        if alerts and self.on_stop_alert:
            self.on_stop_alert(alerts)

        # Callback de actualización
        if self.on_state_update:
            self.on_state_update(state)

        logger.info(
            "[PositionTracker] Sync OK | equity=$%.2f | cash=$%.2f | posiciones=%d | "
            "DD=%.2f%%",
            equity, credit, len(enriched), dd_pct * 100,
        )

    # ------------------------------------------------------------------
    # Enriquecimiento de posiciones
    # ------------------------------------------------------------------

    def _enrich_positions(self, raw: List[Dict]) -> List[Dict]:
        """
        Añade campos calculados a cada posición:
          current_price, unrealized_pnl, holding_days,
          entry_regime, current_regime.
        """
        enriched = []
        for pos in raw:
            p = dict(pos)

            # eToro API devuelve positionId / instrumentId (camelCase, d minúscula).
            # Soportamos también PascalCase (positionID/instrumentID) para retro-compatibilidad.
            pos_id    = str(p.get("positionId") or p.get("positionID", ""))
            instr_id  = str(p.get("instrumentId") or p.get("instrumentID", ""))
            open_rate = float(p.get("openRate", 0))
            units     = float(p.get("units", 0))

            # P&L no realizado: campo directo (spec) o anidado (guide).
            api_pnl = self._get_pos_pnl(p)
            p["unrealized_pnl"] = round(api_pnl, 2)

            # Precio actual:
            #  1. closeRate del endpoint /real/pnl (precio de cierre actual).
            #  2. Caché de rates del polling de precios (instrumentId en minúscula).
            #  3. Estimado desde openRate + pnL / units (cuando el mercado devuelve 0).
            close_rate = float(p.get("closeRate", 0))
            rate_data  = self._current_prices.get(instr_id, {})

            if close_rate > 0 and close_rate != open_rate:
                current_price = close_rate
            elif rate_data:
                current_price = float(rate_data.get("mid", open_rate))
            elif units > 0 and api_pnl != 0:
                current_price = round(open_rate + api_pnl / units, 4)
            else:
                current_price = open_rate

            p["current_price"] = current_price

            # Días en cartera
            open_dt = p.get("openDateTime", "")
            try:
                dt = datetime.fromisoformat(open_dt.replace("Z", "+00:00"))
                p["holding_days"] = (datetime.now(timezone.utc) - dt).days
            except Exception:
                p["holding_days"] = 0

            # Regímenes HMM
            p["entry_regime"]   = self._entry_regimes.get(pos_id, "UNKNOWN")
            p["current_regime"] = "UNKNOWN"   # actualizado por main.py tras predicción HMM

            enriched.append(p)
        return enriched

    # ------------------------------------------------------------------
    # Alertas de stop loss
    # ------------------------------------------------------------------

    def _check_stop_alerts(self, positions: List[Dict]) -> List[Dict]:
        """
        Detecta posiciones con stop loss inefectivo.
        Retorna lista de posiciones problemáticas con nivel de alerta.
        """
        alerts = []
        for pos in positions:
            pos_id  = int(pos.get("positionId") or pos.get("positionID", 0))
            slr     = float(pos.get("stopLossRate", 0))
            no_stop = bool(pos.get("isNoStopLoss", False))

            if no_stop or slr <= STOP_ALERT_THR:
                severity = "CRÍTICO" if pos_id in NO_STOP_POS_IDS else "WARNING"
                alerts.append({
                    "positionID":   pos_id,
                    "instrumentID": pos.get("instrumentID"),
                    "stopLossRate": slr,
                    "isNoStopLoss": no_stop,
                    "severity":     severity,
                })
                logger.warning(
                    "[PositionTracker] %s — POSICIÓN SIN STOP EFECTIVO | "
                    "posID=%d | instrID=%s | stopLossRate=%.4f",
                    severity, pos_id, pos.get("instrumentID"), slr,
                )
        return alerts

    # ------------------------------------------------------------------
    # Persistencia (state_snapshot.json)
    # ------------------------------------------------------------------

    def save_snapshot(self) -> None:
        """Guarda el estado actual en state_snapshot.json para recuperación."""
        snapshot = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "positions":       self._positions,
            "peak_equity":     self._peak_equity,
            "entry_regimes":   self._entry_regimes,
            "last_order_ts":   self._last_order_ts,
            "daily_trades":    self._daily_trades,
        }
        try:
            with open(SNAPSHOT_FILE, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)
            logger.info("[PositionTracker] Snapshot guardado: %s", SNAPSHOT_FILE)
        except OSError as exc:
            logger.error("[PositionTracker] Error guardando snapshot: %s", exc)

    def load_snapshot(self) -> bool:
        """
        Carga el snapshot previo y reconcilia con la API.
        Retorna True si el snapshot existe y fue cargado.
        """
        if not os.path.exists(SNAPSHOT_FILE):
            logger.info("[PositionTracker] No existe snapshot previo.")
            return False

        try:
            with open(SNAPSHOT_FILE) as f:
                snap = json.load(f)

            self._peak_equity    = float(snap.get("peak_equity", 0))
            self._entry_regimes  = snap.get("entry_regimes", {})
            self._last_order_ts  = snap.get("last_order_ts", {})
            self._daily_trades   = int(snap.get("daily_trades", 0))

            # Reconciliar posiciones snapshot vs API (evitar duplicados)
            snap_ids = {str(p["positionID"]) for p in snap.get("positions", [])}
            api_ids  = {str(p["positionID"]) for p in self._positions}
            ghost    = snap_ids - api_ids   # en snapshot pero no en API → cerradas
            new_pos  = api_ids - snap_ids   # en API pero no en snapshot → abiertas post-crash

            if ghost:
                logger.info(
                    "[PositionTracker] Posiciones cerradas desde último snapshot: %s", ghost
                )
            if new_pos:
                logger.info(
                    "[PositionTracker] Posiciones nuevas desde último snapshot: %s", new_pos
                )

            logger.info(
                "[PositionTracker] Snapshot cargado | peak_equity=$%.2f | "
                "%d posiciones reconciliadas",
                self._peak_equity, len(api_ids),
            )
            return True

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("[PositionTracker] Error cargando snapshot: %s", exc)
            return False

    # ------------------------------------------------------------------
    # API pública de estado
    # ------------------------------------------------------------------

    def get_portfolio_state(self) -> Optional[PortfolioState]:
        """Retorna el último PortfolioState calculado."""
        return self._portfolio_state

    def get_positions(self) -> List[Dict]:
        """Retorna la lista de posiciones enriquecidas."""
        return list(self._positions)

    def get_position_by_instrument(self, instrument_id: int) -> Optional[Dict]:
        """Busca una posición activa por instrumentID."""
        for pos in self._positions:
            if int(pos.get("instrumentID", -1)) == instrument_id:
                return pos
        return None

    def update_prices(self, rates: Dict[str, Dict]) -> None:
        """
        Actualiza el caché de precios actuales.
        Llamar desde el bucle principal tras polling de rates.
        rates = {instrumentId_str: {"bid": float, "ask": float, "mid": float}}
        """
        self._current_prices.update(rates)

    def update_regime(self, position_id: str, regime: str) -> None:
        """Actualiza el régimen HMM actual de una posición."""
        for pos in self._positions:
            if str(pos.get("positionID")) == position_id:
                pos["current_regime"] = regime

    def record_entry_regime(self, position_id: str, regime: str) -> None:
        """Registra el régimen HMM en el momento de apertura de la posición."""
        self._entry_regimes[position_id] = regime

    def record_order(self, instrument_id: str) -> None:
        """Registra timestamp de la última orden para bloqueo de duplicados."""
        self._last_order_ts[instrument_id] = time.time()
        self._daily_trades += 1

    def reset_daily_trades(self) -> None:
        """Llamar al inicio de cada día de trading."""
        self._daily_trades = 0
        logger.info("[PositionTracker] Contador de operaciones diarias reseteado.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_equity(
        self,
        pnl_portfolio: Dict,
        credit: float,
        positions: List[Dict],
    ) -> float:
        """
        Portfolio Value = cash disponible + capital invertido + P&L no realizado.

        Fórmula oficial eToro (calculate-equity guide):
          equity = credit + Σ(positions[i].amount) + Σ(positions[i].pnL)

        Nota: unitsBaseValueDollars == amount (inversión inicial, NO valor de mercado).
        El P&L realizado se suma por separado desde el campo pnL / unrealizedPnL.pnL.
        """
        if positions:
            total = sum(
                float(p.get("amount", p.get("initialAmountInDollars", 0)))
                + self._get_pos_pnl(p)
                for p in positions
            )
            return round(credit + total, 2)

        return round(credit, 2)

    @staticmethod
    def _get_pos_pnl(pos: Dict) -> float:
        """Extrae el P&L no realizado de una posición (spec: pnL; guide: unrealizedPnL.pnL)."""
        direct = pos.get("pnL")
        if direct is not None:
            return float(direct)
        nested = pos.get("unrealizedPnL")
        if isinstance(nested, dict):
            return float(nested.get("pnL", 0))
        return 0.0
