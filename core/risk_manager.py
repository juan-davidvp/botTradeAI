"""
core/risk_manager.py
Fase 5 — Capa de Gestión de Riesgos.

Principio arquitectónico:
  El RiskManager opera de forma INDEPENDIENTE del HMM y tiene
  PODER DE VETO ABSOLUTO sobre cualquier señal.

Capital base: $546.14 USD (2026-04-14)
Todos los umbrales se cargan desde settings.yaml.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.regime_strategies import Signal, Direction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes calibradas al portafolio real ($546.14)
# ---------------------------------------------------------------------------
EQUITY_BASE           = 546.14
MAX_EXPOSURE_PCT      = 0.92           # 92% = $502.45
MAX_POSITION_PCT      = 0.20           # 20% = $109.23
MAX_POSITION_USD      = 109.23
MIN_POSITION_USD      = 100.00         # Mínimo eToro
MAX_CORRELATED_PCT    = 0.30           # 30% mismo sector
MAX_CONCURRENT        = 5
MAX_DAILY_TRADES      = 10
MAX_LEVERAGE          = 1.0
RISK_PER_TRADE_PCT    = 0.01           # 1% = $5.46
DEFAULT_STOP_PCT      = 0.05
GAP_RISK_MULTIPLIER   = 3.0            # riesgo de gap = 3× stop

# Circuit Breakers (USD sobre $546.14)
CB_DAILY_REDUCE_PCT   = 0.02           # $10.92
CB_DAILY_HALT_PCT     = 0.03           # $16.38
CB_WEEKLY_REDUCE_PCT  = 0.05           # $27.31
CB_WEEKLY_HALT_PCT    = 0.07           # $38.23
CB_PEAK_DD_PCT        = 0.10           # $54.61

# Correlación
CORR_REDUCE_THRESHOLD = 0.70
CORR_REJECT_THRESHOLD = 0.85
CORR_WINDOW_DAYS      = 60

# Orden duplicada
DUPLICATE_LOCK_SECONDS = 60

# Spread máximo aceptable
MAX_SPREAD_PCT        = 0.005          # 0.5%

LOCK_FILE             = "trading_halted.lock"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    equity: float
    cash: float                          # clientPortfolio.credit
    buying_power: float                  # cash disponible para nuevas posiciones
    positions: List[Dict[str, Any]]      # Lista de posiciones eToro
    daily_pnl: float
    weekly_pnl: float
    peak_equity: float
    drawdown_pct: float                  # (equity - peak) / peak
    circuit_breaker_status: str          # "OK" | "REDUCE_50" | "HALT" | "LOCKED"
    flicker_rate: float                  # cambios de régimen / flicker_window
    daily_trades_count: int = 0
    last_order_by_instrument: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskDecision:
    approved: bool
    modified_signal: Optional[Signal]   # None si rechazado
    rejection_reason: str               # "" si aprobado
    modifications_list: List[str]       # lista de ajustes aplicados

    def __str__(self) -> str:
        if self.approved:
            mods = f" [{'; '.join(self.modifications_list)}]" if self.modifications_list else ""
        else:
            mods = f" RECHAZADO: {self.rejection_reason}"
        return f"RiskDecision(approved={self.approved}{mods})"


@dataclass
class CircuitBreakerEvent:
    timestamp: datetime
    breaker_type: str
    dd_usd: float
    dd_pct: float
    equity: float
    regime: str
    positions_closed: List[str]


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Evalúa y activa los interruptores automáticos según P&L real de eToro.

    Umbrales calibrados a $546.14:
      DD Diario  > 2% ($10.92)  → REDUCE_50
      DD Diario  > 3% ($16.38)  → HALT_DAY
      DD Semanal > 5% ($27.31)  → REDUCE_50_WEEK
      DD Semanal > 7% ($38.23)  → HALT_WEEK
      Peak DD    > 10% ($54.61) → LOCK (requiere intervención manual)
    """

    def __init__(self, base_equity: float = EQUITY_BASE):
        self.base_equity    = base_equity
        self.peak_equity    = base_equity
        self._day_start_eq  = base_equity
        self._week_start_eq = base_equity
        self._current_day   = date.today()
        self._current_week  = date.today().isocalendar()[1]
        self._history: List[CircuitBreakerEvent] = []
        self.status: str = "OK"           # OK | REDUCE_50 | HALT | LOCKED

    def update(self, equity: float, regime: str = "UNKNOWN") -> str:
        """
        Actualiza el estado del circuit breaker con la equity actual.
        Retorna el estado: "OK" | "REDUCE_50" | "HALT" | "LOCKED".
        """
        today = date.today()
        week  = today.isocalendar()[1]

        # Reset diario / semanal
        if today != self._current_day:
            self._day_start_eq  = equity
            self._current_day   = today
            if self.status in ("REDUCE_50", "HALT") and "PEAK" not in self.status:
                self.status = "OK"
                logger.info("[CB] Reset diario — status → OK")

        if week != self._current_week:
            self._week_start_eq = equity
            self._current_week  = week
            if self.status not in ("LOCKED",):
                self.status = "OK"
                logger.info("[CB] Reset semanal — status → OK")

        # Actualizar pico
        self.peak_equity = max(self.peak_equity, equity)

        # ── Calcular DDs ────────────────────────────────────────────────
        dd_daily  = (equity - self._day_start_eq)  / self._day_start_eq  if self._day_start_eq  > 0 else 0.0
        dd_weekly = (equity - self._week_start_eq) / self._week_start_eq if self._week_start_eq > 0 else 0.0
        dd_peak   = (equity - self.peak_equity)    / self.peak_equity    if self.peak_equity    > 0 else 0.0

        # ── Evaluación de breakers (orden de severidad) ─────────────────
        if dd_peak < -CB_PEAK_DD_PCT:
            self._activate("PEAK_DD_LOCK", dd_peak, equity, regime, [])
            self.status = "LOCKED"

        elif dd_weekly < -CB_WEEKLY_HALT_PCT:
            self._activate("WEEKLY_HALT", dd_weekly, equity, regime, [])
            self.status = "HALT"

        elif dd_weekly < -CB_WEEKLY_REDUCE_PCT and self.status == "OK":
            self._activate("WEEKLY_REDUCE_50", dd_weekly, equity, regime, [])
            self.status = "REDUCE_50"

        elif dd_daily < -CB_DAILY_HALT_PCT:
            self._activate("DAILY_HALT", dd_daily, equity, regime, [])
            self.status = "HALT"

        elif dd_daily < -CB_DAILY_REDUCE_PCT and self.status == "OK":
            self._activate("DAILY_REDUCE_50", dd_daily, equity, regime, [])
            self.status = "REDUCE_50"

        return self.status

    def _activate(
        self, breaker_type: str, dd: float, equity: float,
        regime: str, positions_closed: List[str]
    ) -> None:
        dd_usd = abs(dd * equity)
        event = CircuitBreakerEvent(
            timestamp=datetime.utcnow(),
            breaker_type=breaker_type,
            dd_usd=round(dd_usd, 2),
            dd_pct=round(dd * 100, 2),
            equity=round(equity, 2),
            regime=regime,
            positions_closed=positions_closed,
        )
        self._history.append(event)

        logger.warning(
            "[CB] ACTIVADO %s | DD=$%.2f (%.2f%%) | Equity=$%.2f | Régimen=%s",
            breaker_type, dd_usd, dd * 100, equity, regime,
        )

        if breaker_type == "PEAK_DD_LOCK":
            with open(LOCK_FILE, "w") as f:
                f.write(
                    f"TRADING HALTED — {datetime.utcnow().isoformat()}\n"
                    f"Peak DD: {dd*100:.2f}% (${dd_usd:.2f})\n"
                    f"Equity: ${equity:.2f}\n"
                    f"Requiere intervención manual para reanudar.\n"
                )
            logger.critical("[CB] LOCK FILE generado: %s", LOCK_FILE)

    def check(self) -> str:
        """Retorna el estado actual del circuit breaker."""
        if os.path.exists(LOCK_FILE):
            return "LOCKED"
        return self.status

    def size_multiplier(self) -> float:
        """Multiplicador de tamaño de posición según estado del CB."""
        status = self.check()
        if status in ("HALT", "LOCKED"):
            return 0.0
        if status == "REDUCE_50":
            return 0.50
        return 1.0

    def reset_daily(self, equity: float) -> None:
        self._day_start_eq = equity
        if self.status not in ("LOCKED", "HALT"):
            self.status = "OK"
        logger.info("[CB] Reset diario manual")

    def reset_weekly(self, equity: float) -> None:
        self._week_start_eq = equity
        if self.status not in ("LOCKED",):
            self.status = "OK"
        logger.info("[CB] Reset semanal manual")

    def get_history(self) -> List[CircuitBreakerEvent]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Gestión de riesgos con poder de veto absoluto sobre cualquier señal.

    Parámetros
    ----------
    settings : dict
        Diccionario con los parámetros de settings.yaml (sección 'risk').
    """

    def __init__(self, settings: Optional[Dict] = None):
        s = settings or {}
        self.base_equity       = float(s.get("initial_equity",       EQUITY_BASE))
        self.max_exposure      = float(s.get("max_exposure",          MAX_EXPOSURE_PCT))
        self.max_position_pct  = float(s.get("max_single_position",   MAX_POSITION_PCT))
        self.max_leverage      = float(s.get("max_leverage",          MAX_LEVERAGE))
        self.max_concurrent    = int(s.get("max_concurrent",          MAX_CONCURRENT))
        self.max_daily_trades  = int(s.get("max_daily_trades",        MAX_DAILY_TRADES))
        self.risk_per_trade    = float(s.get("max_risk_per_trade",    RISK_PER_TRADE_PCT))
        self.stop_pct          = float(s.get("default_stop_loss_pct", DEFAULT_STOP_PCT))
        self.min_pos_usd       = float(s.get("min_position_size_usd", MIN_POSITION_USD))
        self.max_pos_usd       = float(s.get("target_position_size_usd", MAX_POSITION_USD))

        self.circuit_breaker   = CircuitBreaker(self.base_equity)

        # Historial de retornos para correlación
        self._returns_history: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        price_history: Optional[Dict[str, pd.Series]] = None,
        current_spread_pct: float = 0.0,
    ) -> RiskDecision:
        """
        Valida una señal contra todas las reglas de riesgo.

        Retorna RiskDecision con:
          approved=True  → señal aprobada (posiblemente modificada)
          approved=False → señal rechazada con motivo estructurado
        """
        modifications: List[str] = []
        modified = self._clone_signal(signal)

        # 0. Lock file (máxima prioridad)
        if os.path.exists(LOCK_FILE):
            return self._reject("LOCK_FILE_ACTIVO — requiere intervención manual", signal)

        # 1. Circuit Breaker
        cb_mult = self.circuit_breaker.size_multiplier()
        cb_status = self.circuit_breaker.check()

        if cb_mult == 0.0:
            return self._reject(
                f"CIRCUIT_BREAKER activo [{cb_status}] — trading pausado", signal
            )
        if cb_mult < 1.0:
            old_usd = modified.position_size_usd
            modified.position_size_usd = round(old_usd * cb_mult, 2)
            modified.position_size_pct = round(modified.position_size_pct * cb_mult, 4)
            modifications.append(f"CB_REDUCE_50: ${old_usd:.2f}→${modified.position_size_usd:.2f}")

        # 2. Flicker — forzar apalancamiento 1.0x y reducir 50%
        if portfolio_state.flicker_rate > 0.2:
            modified.leverage = 1.0
            modified.position_size_usd = round(modified.position_size_usd * 0.50, 2)
            modified.position_size_pct = round(modified.position_size_pct * 0.50, 4)
            modifications.append("FLICKER_REDUCCION_50")

        # 3. Límite de leverage
        if signal.leverage > self.max_leverage:
            modified.leverage = self.max_leverage
            modifications.append(f"LEVERAGE_CAPPED:{signal.leverage}→{self.max_leverage}")

        # 4. Más de 3 posiciones → forzar leverage 1.0x
        if len(portfolio_state.positions) >= 3 and modified.leverage > 1.0:
            modified.leverage = 1.0
            modifications.append("LEVERAGE_FORZADO_1x_POR_POSICIONES")

        # 5. Stop loss obligatorio
        if not self._has_valid_stop(modified):
            return self._reject(
                f"STOP_LOSS_AUSENTE o inválido (stopLossRate={modified.stop_loss})", signal
            )

        # 6. Duplicado (mismo instrumento en < 60s)
        dup_check = self._check_duplicate(signal.symbol, portfolio_state)
        if dup_check:
            return self._reject(dup_check, signal)

        # 7. Spread bid-ask
        if current_spread_pct > MAX_SPREAD_PCT:
            return self._reject(
                f"SPREAD_EXCESIVO: {current_spread_pct:.3%} > {MAX_SPREAD_PCT:.1%}", signal
            )

        # 8. Límite de operaciones diarias
        if portfolio_state.daily_trades_count >= self.max_daily_trades:
            return self._reject(
                f"MAX_DAILY_TRADES alcanzado: {portfolio_state.daily_trades_count}/{self.max_daily_trades}",
                signal,
            )

        # 9. Posiciones concurrentes máximas
        n_open = len(portfolio_state.positions)
        if n_open >= self.max_concurrent:
            return self._reject(
                f"MAX_CONCURRENT alcanzado: {n_open}/{self.max_concurrent} posiciones abiertas",
                signal,
            )

        # 10. Saldo disponible (credit)
        if portfolio_state.cash < modified.position_size_usd:
            if portfolio_state.cash < self.min_pos_usd:
                return self._reject(
                    f"CASH_INSUFICIENTE: disponible=${portfolio_state.cash:.2f} "
                    f"mínimo=${self.min_pos_usd:.2f}",
                    signal,
                )
            old_usd = modified.position_size_usd
            modified.position_size_usd = round(
                min(modified.position_size_usd, portfolio_state.cash), 2
            )
            modifications.append(f"REDUCIDO_POR_CASH: ${old_usd:.2f}→${modified.position_size_usd:.2f}")

        # 11. Exposición total máxima
        current_invested = sum(float(p.get("amount", 0)) for p in portfolio_state.positions)
        max_allowed_usd  = portfolio_state.equity * self.max_exposure
        if current_invested + modified.position_size_usd > max_allowed_usd:
            available = max_allowed_usd - current_invested
            if available < self.min_pos_usd:
                return self._reject(
                    f"EXPOSICION_MAXIMA: invertido=${current_invested:.2f} "
                    f"+ nueva=${modified.position_size_usd:.2f} > "
                    f"máx=${max_allowed_usd:.2f}",
                    signal,
                )
            modified.position_size_usd = round(available, 2)
            modifications.append(f"LIMITADO_EXPOSICION_92PCT: ${modified.position_size_usd:.2f}")

        # 12. Límite por posición individual (20% = $109.23)
        max_pos = min(
            portfolio_state.equity * self.max_position_pct,
            self.max_pos_usd,
        )
        if modified.position_size_usd > max_pos:
            old_usd = modified.position_size_usd
            modified.position_size_usd = round(max_pos, 2)
            modifications.append(f"CAPPED_20PCT: ${old_usd:.2f}→${modified.position_size_usd:.2f}")

        # 13. Mínimo $100 eToro
        if modified.position_size_usd < self.min_pos_usd:
            return self._reject(
                f"DEBAJO_MINIMO_ETORO: ${modified.position_size_usd:.2f} < ${self.min_pos_usd:.2f}",
                signal,
            )

        # 14. Riesgo máximo por operación (gap-adjusted)
        risk_usd     = modified.position_size_usd * self.stop_pct
        gap_risk_usd = risk_usd * GAP_RISK_MULTIPLIER
        max_risk_usd = portfolio_state.equity * self.risk_per_trade
        if gap_risk_usd > max_risk_usd * 3:
            old_usd = modified.position_size_usd
            modified.position_size_usd = round(max_risk_usd / (self.stop_pct * GAP_RISK_MULTIPLIER), 2)
            modifications.append(
                f"GAP_RISK_AJUSTADO: ${old_usd:.2f}→${modified.position_size_usd:.2f} "
                f"(gap_risk=${gap_risk_usd:.2f} > ${max_risk_usd*3:.2f})"
            )

        # 15. Correlación con posiciones existentes
        if price_history:
            corr_result, corr_val = self._check_correlation(
                signal.symbol, portfolio_state, price_history
            )
            if corr_result == "REJECT":
                return self._reject(
                    f"CORRELACION_EXCESIVA: {corr_val:.2f} > {CORR_REJECT_THRESHOLD}", signal
                )
            if corr_result == "REDUCE":
                old_usd = modified.position_size_usd
                modified.position_size_usd = round(old_usd * 0.50, 2)
                modifications.append(
                    f"CORRELACION_REDUCCION_50: corr={corr_val:.2f} > {CORR_REDUCE_THRESHOLD}"
                )

        # 16. Corroboración final del stop loss
        if modified.stop_loss <= 0 or modified.stop_loss >= modified.entry_price:
            # Recalcular stop por defecto
            modified.stop_loss = round(modified.entry_price * (1 - self.stop_pct), 4)
            modifications.append(f"STOP_RECALCULADO: {modified.stop_loss}")

        # ── Aprobado ─────────────────────────────────────────────────────
        if modifications:
            modified.reasoning += f" | RISK_MODS: {'; '.join(modifications)}"
            logger.info(
                "[Risk] %s APROBADO con modificaciones: %s",
                signal.symbol, modifications,
            )
        else:
            logger.info("[Risk] %s APROBADO sin modificaciones", signal.symbol)

        return RiskDecision(
            approved=True,
            modified_signal=modified,
            rejection_reason="",
            modifications_list=modifications,
        )

    # ------------------------------------------------------------------
    # Verificaciones auxiliares
    # ------------------------------------------------------------------

    def _has_valid_stop(self, signal: Signal) -> bool:
        """Verifica que el stop loss sea válido (> 0.001 y < entry_price)."""
        return (
            signal.stop_loss is not None
            and signal.stop_loss > 0.001
            and signal.stop_loss < signal.entry_price
        )

    def _check_duplicate(self, symbol: str, state: PortfolioState) -> str:
        """Retorna mensaje de error si es una orden duplicada, '' si está OK."""
        last_ts = state.last_order_by_instrument.get(symbol)
        if last_ts is not None:
            elapsed = time.time() - last_ts
            if elapsed < DUPLICATE_LOCK_SECONDS:
                return (
                    f"ORDEN_DUPLICADA: instrumentID={symbol} — "
                    f"última orden hace {elapsed:.0f}s < {DUPLICATE_LOCK_SECONDS}s"
                )
        return ""

    def _check_correlation(
        self,
        symbol: str,
        state: PortfolioState,
        price_history: Dict[str, pd.Series],
    ) -> Tuple[str, float]:
        """
        Retorna ("OK"|"REDUCE"|"REJECT", max_correlation).
        Correlación rodante de 60 días con posiciones existentes.
        """
        if symbol not in price_history:
            return "OK", 0.0

        target_rets = price_history[symbol].pct_change().dropna().tail(CORR_WINDOW_DAYS)
        if len(target_rets) < 20:
            return "OK", 0.0

        max_corr = 0.0
        for pos in state.positions:
            pos_sym = str(pos.get("instrumentId") or pos.get("instrumentID") or pos.get("symbol", ""))
            if pos_sym == symbol or pos_sym not in price_history:
                continue
            pos_rets = price_history[pos_sym].pct_change().dropna().tail(CORR_WINDOW_DAYS)
            aligned  = target_rets.align(pos_rets, join="inner")[0]
            aligned_pos = target_rets.align(pos_rets, join="inner")[1]
            if len(aligned) < 20:
                continue
            corr = float(aligned.corr(aligned_pos))
            if np.isnan(corr):
                continue
            max_corr = max(max_corr, corr)

        if max_corr > CORR_REJECT_THRESHOLD:
            return "REJECT", max_corr
        if max_corr > CORR_REDUCE_THRESHOLD:
            return "REDUCE", max_corr
        return "OK", max_corr

    # ------------------------------------------------------------------
    # Urgent stops (verificación al inicio del sistema)
    # ------------------------------------------------------------------

    def check_urgent_stops(
        self,
        positions: List[Dict[str, Any]],
        urgent_stops_config: List[Dict],
    ) -> List[Dict]:
        """
        Detecta posiciones con stop loss inválido (isNoStopLoss=True o
        stopLossRate <= 0.001) y retorna la lista con los stopLossRate
        urgentes ya calculados desde settings.yaml.

        Retorna lista de {position_id, current_stop, required_stop, action}.
        """
        urgent_map = {str(s["position_id"]): s["stop_loss_rate"] for s in urgent_stops_config}
        alerts = []

        for pos in positions:
            pid   = str(pos.get("positionId") or pos.get("positionID", ""))
            instr = pos.get("instrumentId") or pos.get("instrumentID")
            slr   = float(pos.get("stopLossRate", 0))
            nosl  = bool(pos.get("isNoStopLoss", False))

            if nosl or slr <= 0.001:
                required = urgent_map.get(pid)
                alerts.append({
                    "position_id":    pid,
                    "instrument_id":  instr,
                    "current_stop":   slr,
                    "required_stop":  required,
                    "action":         "CLOSE_REOPEN_WITH_STOP",
                })
                logger.warning(
                    "[RiskManager] POSICIÓN SIN STOP EFECTIVO | posID=%s | instrID=%s | "
                    "stopLossRate=%.4f | requerido=%.4f",
                    pid, instr, slr, required or 0,
                )

        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reject(self, reason: str, original: Signal) -> RiskDecision:
        logger.warning("[Risk] %s RECHAZADA — %s", original.symbol, reason)
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason=reason,
            modifications_list=[],
        )

    def _clone_signal(self, signal: Signal) -> Signal:
        """Crea una copia del signal para modificar sin alterar el original."""
        import copy
        return copy.deepcopy(signal)

    # ------------------------------------------------------------------
    # Actualizar estado del circuit breaker desde el bucle principal
    # ------------------------------------------------------------------

    def update_circuit_breaker(
        self,
        current_equity: float,
        regime: str = "UNKNOWN",
    ) -> str:
        """
        Llamar en cada ciclo de polling (30s).
        Retorna el status actual del CB.
        """
        return self.circuit_breaker.update(current_equity, regime)

    def get_position_size(self, equity: float, stop_pct: Optional[float] = None) -> float:
        """
        Calcula el tamaño de posición basado en riesgo fijo:
          position_size = risk_amount / stop_loss_pct
          = (equity × 1%) / 5% = equity × 20%
        Calibrado: $546.14 × 1% / 5% = $109.23
        """
        sp = stop_pct if stop_pct and stop_pct > 0 else self.stop_pct
        risk_usd = equity * self.risk_per_trade
        raw      = risk_usd / sp
        return round(min(raw, equity * self.max_position_pct, self.max_pos_usd), 2)
