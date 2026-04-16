"""
monitoring/alerts.py
Fase 8 — Gestión de Alertas y Notificaciones.

Disparadores:
  1.  Cambio de régimen HMM
  2.  Circuit breaker activado (umbrales en USD)
  3.  ALERTA INMEDIATA — isNoStopLoss=True o stopLossRate <= 0.001
  4.  ALERTA INMEDIATA — posición con peso > 20%
  5.  P&L inusual > ±5% diario
  6.  3 fallos consecutivos de API eToro
  7.  Error 401 / pérdida de conexión
  8.  HMM re-entrenado exitosamente
  9.  Tasa de parpadeo (flicker) excedida
  10. ROI TRACKER: equity < curva objetivo 7 días consecutivos

Rate limit : 1 alerta por tipo cada 15 minutos.
           (excepto alertas INMEDIATAS: sin rate limit)
Entrega    : consola (logging), archivo (logger.py), email (opcional), webhook (opcional).
"""

import logging
import smtplib
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

from monitoring.logger import log_alert, log_circuit_breaker

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 15 * 60   # 15 minutos


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Centraliza el envío de alertas con rate limiting por tipo de evento.

    Parámetros
    ----------
    settings : dict
        Sección 'monitoring' de settings.yaml.

    Claves opcionales en settings['monitoring']:
      alert_rate_limit_minutes : int   (default 15)
      email_to                 : str   destinatario
      email_from               : str   remitente
      email_smtp_host          : str
      email_smtp_port          : int   (default 587)
      email_smtp_user          : str
      email_smtp_password      : str
      webhook_url              : str   URL Discord/Slack
    """

    def __init__(self, settings: Optional[Dict] = None):
        cfg                    = settings or {}
        self._rate_limit       = int(cfg.get("alert_rate_limit_minutes", 15)) * 60
        self._last_sent        : Dict[str, float] = {}
        self._below_target_days: int = 0

        # ── Email (opcional) ────────────────────────────────────────────────
        self._email_to         = cfg.get("email_to")
        self._email_from       = cfg.get("email_from")
        self._smtp_host        = cfg.get("email_smtp_host")
        self._smtp_port        = int(cfg.get("email_smtp_port", 587))
        self._smtp_user        = cfg.get("email_smtp_user")
        self._smtp_password    = cfg.get("email_smtp_password")
        self._email_enabled    = all([
            self._email_to, self._email_from,
            self._smtp_host, self._smtp_user, self._smtp_password,
        ])

        # ── Webhook (opcional) ──────────────────────────────────────────────
        self._webhook_url      = cfg.get("webhook_url")
        self._webhook_enabled  = bool(self._webhook_url)

    # -----------------------------------------------------------------------
    # Rate limiter
    # -----------------------------------------------------------------------

    def _can_send(self, alert_type: str) -> bool:
        now  = time.time()
        last = self._last_sent.get(alert_type, 0)
        if now - last >= self._rate_limit:
            self._last_sent[alert_type] = now
            return True
        return False

    # -----------------------------------------------------------------------
    # Núcleo de entrega
    # -----------------------------------------------------------------------

    def _emit(
        self,
        alert_type: str,
        message: str,
        level: str = "WARNING",
        immediate: bool = False,
        regime: Optional[str] = None,
        probability: Optional[float] = None,
        equity: Optional[float] = None,
        positions_count: Optional[int] = None,
        daily_pnl: Optional[float] = None,
        **extra_fields,
    ) -> None:
        """Emite la alerta por todos los canales habilitados."""
        if not immediate and not self._can_send(alert_type):
            return

        # ── Canal 1: log estructurado (consola + archivo) ───────────────────
        log_alert(
            message,
            alert_type=alert_type,
            level=level,
            regime=regime,
            probability=probability,
            equity=equity,
            positions_count=positions_count,
            daily_pnl=daily_pnl,
            **extra_fields,
        )

        subject = f"[regime-trader] {alert_type}: {message[:80]}"

        # ── Canal 2: email ───────────────────────────────────────────────────
        if self._email_enabled:
            self._send_email(subject, message, level)

        # ── Canal 3: webhook (Discord/Slack) ─────────────────────────────────
        if self._webhook_enabled:
            self._send_webhook(subject, message, level)

    def _send_email(self, subject: str, body: str, level: str) -> None:
        try:
            msg           = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"]    = self._email_from
            msg["To"]      = self._email_to
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
                smtp.starttls()
                smtp.login(self._smtp_user, self._smtp_password)
                smtp.sendmail(self._email_from, [self._email_to], msg.as_string())
        except Exception as exc:
            logger.warning("Email no enviado (%s): %s", subject[:60], exc)

    def _send_webhook(self, subject: str, body: str, level: str) -> None:
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🔴", "CRITICAL": "🚨"}.get(
            level.upper(), "📢"
        )
        payload = {"content": f"{emoji} **{subject}**\n```\n{body}\n```"}
        try:
            resp = requests.post(self._webhook_url, json=payload, timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Webhook no enviado: %s", exc)

    # -----------------------------------------------------------------------
    # Alertas específicas — Disparador 1: cambio de régimen
    # -----------------------------------------------------------------------

    def send_regime_change(
        self,
        old: str,
        new: str,
        probability: float,
        equity: Optional[float] = None,
        positions_count: Optional[int] = None,
        daily_pnl: Optional[float] = None,
    ) -> None:
        self._emit(
            "REGIME_CHANGE",
            f"Régimen cambiado: {old} → {new} ({probability:.1%})",
            level="WARNING",
            regime=new,
            probability=probability,
            equity=equity,
            positions_count=positions_count,
            daily_pnl=daily_pnl,
            old_regime=old,
        )

    # -----------------------------------------------------------------------
    # Disparador 2: circuit breaker
    # -----------------------------------------------------------------------

    def send_circuit_breaker(
        self,
        breaker_type: str,
        dd_usd: float,
        dd_pct: float,
        equity: float,
        regime: str,
        positions_closed: Optional[List[str]] = None,
    ) -> None:
        """
        Umbrales USD:
          REDUCE_DAILY  $10.92 (2%)  | HALT_DAILY   $16.38 (3%)
          REDUCE_WEEKLY $27.31 (5%)  | HALT_WEEKLY  $38.23 (7%)
          PEAK_LOCK     $54.61 (10%)
        """
        msg = (
            f"Circuit Breaker [{breaker_type}] | "
            f"DD=${dd_usd:.2f} ({dd_pct:.2f}%) | "
            f"Equity=${equity:.2f} | Régimen={regime}"
        )
        log_circuit_breaker(
            breaker_type=breaker_type,
            dd_usd=dd_usd,
            dd_pct=dd_pct,
            equity=equity,
            regime=regime,
            positions_closed=positions_closed,
        )
        # Entrega adicional por email/webhook
        if self._email_enabled or self._webhook_enabled:
            if self._can_send(f"CB_{breaker_type}"):
                subject = f"[regime-trader] CB_{breaker_type}: {msg[:80]}"
                if self._email_enabled:
                    self._send_email(subject, msg, "ERROR")
                if self._webhook_enabled:
                    self._send_webhook(subject, msg, "ERROR")

    # -----------------------------------------------------------------------
    # Disparador 3: posición sin stop efectivo (INMEDIATA)
    # -----------------------------------------------------------------------

    def send_stop_loss_alert(self, positions_without_stop: List[Dict]) -> None:
        """ALERTA INMEDIATA — sin rate limit."""
        for pos in positions_without_stop:
            msg = (
                f"POSICIÓN SIN STOP EFECTIVO | "
                f"posID={pos.get('positionID')} | "
                f"instrID={pos.get('instrumentID')} | "
                f"stopLossRate={float(pos.get('stopLossRate', 0)):.4f}"
            )
            self._emit(
                "NO_STOP",
                msg,
                level="ERROR",
                immediate=True,
                position_id=pos.get("positionID"),
                instrument_id=pos.get("instrumentID"),
            )

    # -----------------------------------------------------------------------
    # Disparador 4: posición con peso > 20% (INMEDIATA)
    # -----------------------------------------------------------------------

    def send_overweight_alert(
        self, symbol: Any, weight_pct: float, equity: float
    ) -> None:
        """ALERTA INMEDIATA — sin rate limit."""
        msg = (
            f"PESO EXCESIVO | instrID={symbol} | "
            f"peso={weight_pct:.1f}% (máx 20%) | "
            f"valor=${equity * weight_pct / 100:.2f}"
        )
        self._emit(
            "OVERWEIGHT",
            msg,
            level="ERROR",
            immediate=True,
            equity=equity,
            instrument_id=symbol,
        )

    # -----------------------------------------------------------------------
    # Disparador 5: P&L inusual
    # -----------------------------------------------------------------------

    def send_unusual_pnl(self, daily_pnl_pct: float, equity: float) -> None:
        self._emit(
            "UNUSUAL_PNL",
            f"P&L inusual: {daily_pnl_pct:+.2f}% | Equity=${equity:.2f}",
            level="WARNING",
            equity=equity,
            daily_pnl=equity * daily_pnl_pct / 100,
        )

    # -----------------------------------------------------------------------
    # Disparador 6: fallos consecutivos de API
    # -----------------------------------------------------------------------

    def send_api_failure(self, consecutive_fails: int) -> None:
        self._emit(
            "API_FAILURE",
            f"Fallo de API eToro — {consecutive_fails} intentos consecutivos fallidos",
            level="ERROR",
        )

    # -----------------------------------------------------------------------
    # Disparador 7: credenciales expiradas / error 401
    # -----------------------------------------------------------------------

    def send_credentials_expired(self) -> None:
        self._emit(
            "CREDENTIALS_EXPIRED",
            "Error 401 — Credenciales eToro inválidas o expiradas. Renovar ETORO_API_KEY.",
            level="CRITICAL",
        )

    # -----------------------------------------------------------------------
    # Disparador 8: HMM re-entrenado
    # -----------------------------------------------------------------------

    def send_hmm_retrained(self, n_regimes: int, bic: float) -> None:
        self._emit(
            "HMM_RETRAINED",
            f"HMM re-entrenado exitosamente | n_regimes={n_regimes} | BIC={bic:.2f}",
            level="INFO",
        )

    # -----------------------------------------------------------------------
    # Disparador 9: flicker excedido
    # -----------------------------------------------------------------------

    def send_flicker_alert(self, regime_history: List[str]) -> None:
        self._emit(
            "FLICKER",
            f"Tasa de parpadeo excedida | historial={regime_history[-10:]}",
            level="WARNING",
        )

    # -----------------------------------------------------------------------
    # Disparador 10: ROI TRACKER — equity < curva objetivo 7 días seguidos
    # -----------------------------------------------------------------------

    def send_roi_below_target(
        self, equity: float, target: float, days: int
    ) -> None:
        self._below_target_days += 1
        if self._below_target_days >= 7:
            self._emit(
                "ROI_BELOW_TARGET",
                (
                    f"Equity ${equity:.2f} < objetivo ${target:.2f} | "
                    f"{self._below_target_days} días consecutivos"
                ),
                level="WARNING",
                equity=equity,
            )

    def reset_below_target_counter(self) -> None:
        """Llamar cuando equity vuelve a estar por encima de la curva."""
        self._below_target_days = 0

    # -----------------------------------------------------------------------
    # Error de sistema genérico
    # -----------------------------------------------------------------------

    def send_system_error(self, error: str) -> None:
        self._emit(
            "SYSTEM_ERROR",
            f"Error no controlado: {error}",
            level="ERROR",
        )
