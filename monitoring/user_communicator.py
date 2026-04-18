"""
monitoring/user_communicator.py

Canal de Comunicación Directa: traduce eventos internos del bot
a mensajes amigables en español para el usuario final.

Uso (cero configuración, cualquier módulo):
    from monitoring.user_communicator import get_communicator
    comm = get_communicator()
    comm.emit("regime_change", old="NEUTRAL", new="STRONG_BULL", prob=0.87)

Thread-safe. Escribe a user_messages.json en la raíz del proyecto.
Nunca lanza excepciones: diseñado para no interferir con el motor de trading.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_MESSAGES   = 50
MESSAGES_FILE  = "user_messages.json"

# Cooldowns en segundos para no saturar el chat con mensajes repetitivos
_COOLDOWNS_S: Dict[str, int] = {
    "scan_start":    300,    # 5 min — tick rutinario
    "no_signals":    300,    # 5 min
    "market_closed": 1800,   # 30 min
}

INSTRUMENT_NAMES: Dict[int, str] = {
    4238:   "VOO (S&P 500)",
    14328:  "MercadoLibre",
    9408:   "Palantir",
    6218:   "AppLovin",
    2488:   "Dave Inc.",
    100000: "BTC/USD",
    99001:  "NU",
    99002:  "Rivian",
    99003:  "Robinhood",
    99004:  "Duolingo",
    99005:  "Reddit",
    99006:  "Astera Labs",
    99007:  "CAVA Group",
    99008:  "FIG",
    99009:  "Circle Internet",
    99010:  "Omada Health",
}

REGIME_IMPLICATIONS: Dict[str, str] = {
    "STRONG_BULL": "El mercado tiene impulso alcista fuerte. Operaré activamente buscando oportunidades.",
    "EUPHORIA":    "El mercado muestra señales de euforia. Seré muy selectivo y reduciré el tamaño de posiciones.",
    "WEAK_BULL":   "La tendencia alcista es moderada. Esperaré solo las mejores configuraciones técnicas.",
    "NEUTRAL":     "El mercado no tiene dirección clara. Mantendré la postura defensiva hasta ver una señal más fuerte.",
    "BEAR":        "El mercado está en tendencia bajista. Reduciré la exposición al mínimo.",
    "STRONG_BEAR": "El mercado cae con fuerza. Protejo el capital y no abriré nuevas posiciones.",
    "CRASH":       "El mercado está en caída extrema. He pausado todas las operaciones automáticamente.",
}

KIND_LABELS: Dict[str, str] = {
    "scan_start":       "Escaneo iniciado",
    "no_signals":       "Sin señales",
    "regime_change":    "Cambio de régimen",
    "signal_found":     "Señal detectada",
    "signal_rejected":  "Señal descartada",
    "signal_approved":  "Señal aprobada",
    "order_executed":   "Orden ejecutada",
    "order_failed":     "Orden fallida",
    "circuit_breaker":  "Circuit Breaker",
    "hmm_retrained":    "Modelo actualizado",
    "drawdown_warning": "Alerta de drawdown",
    "position_closed":  "Posición cerrada",
    "api_failure":      "Problema de conexión",
    "market_closed":    "Mercado cerrado",
    "bot_startup":      "Sistema iniciado",
}


@dataclass
class BotMessage:
    ts:    str    # ISO-8601 UTC
    kind:  str    # slug de evento
    icon:  str    # emoji
    text:  str    # mensaje en español (con **bold** y _italic_ Markdown)
    level: str    # "info" | "success" | "warning" | "danger"
    label: str    # etiqueta legible del tipo de evento


class UserCommunicator:
    """
    Traduce eventos del motor de trading a mensajes en lenguaje natural.
    Persiste en user_messages.json (máx 50 mensajes, más reciente primero).
    """

    def __init__(self, root_dir: str = ".") -> None:
        self._path       = os.path.join(root_dir, MESSAGES_FILE)
        self._lock       = threading.Lock()
        self._last_emit  : Dict[str, float] = {}   # kind → epoch seconds

    # ── API pública ───────────────────────────────────────────────────────────

    def emit(self, kind: str, **ctx: Any) -> None:
        """Emite un evento. Nunca lanza excepciones."""
        try:
            # Cooldown: evita repetir mensajes de baja prioridad cada 30s
            cooldown = _COOLDOWNS_S.get(kind, 0)
            if cooldown:
                last = self._last_emit.get(kind, 0.0)
                import time
                now = time.time()
                if now - last < cooldown:
                    return
                self._last_emit[kind] = now

            msg = self._build(kind, ctx)
            if msg:
                self._append(msg)
        except Exception as exc:
            logger.debug("[Comm] emit error kind=%s: %s", kind, exc)

    # ── Construcción de mensajes ──────────────────────────────────────────────

    def _build(self, kind: str, ctx: Dict[str, Any]) -> Optional[BotMessage]:
        builders = {
            "scan_start":       self._b_scan_start,
            "no_signals":       self._b_no_signals,
            "regime_change":    self._b_regime_change,
            "signal_found":     self._b_signal_found,
            "signal_rejected":  self._b_signal_rejected,
            "signal_approved":  self._b_signal_approved,
            "order_executed":   self._b_order_executed,
            "order_failed":     self._b_order_failed,
            "circuit_breaker":  self._b_circuit_breaker,
            "hmm_retrained":    self._b_hmm_retrained,
            "drawdown_warning": self._b_drawdown_warning,
            "position_closed":  self._b_position_closed,
            "api_failure":      self._b_api_failure,
            "market_closed":    self._b_market_closed,
            "bot_startup":      self._b_bot_startup,
        }
        fn = builders.get(kind)
        if fn is None:
            return None
        msg = fn(ctx)
        if msg:
            msg.label = KIND_LABELS.get(kind, kind)
        return msg

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sym(symbol: Any) -> str:
        try:
            return INSTRUMENT_NAMES.get(int(symbol), f"instrumento #{symbol}").split("(")[0].strip()
        except Exception:
            return str(symbol)

    # ── Constructores individuales ────────────────────────────────────────────

    def _b_scan_start(self, ctx: Dict) -> BotMessage:
        n      = ctx.get("n_instruments", 0)
        regime = ctx.get("regime", "desconocido")
        impl   = REGIME_IMPLICATIONS.get(regime, f"régimen {regime}")
        return BotMessage(
            ts=self._now(), kind="scan_start", icon="🔍", level="info", label="",
            text=(
                f"Iniciando análisis de **{n} instrumentos** en régimen "
                f"**{regime}**. {impl}"
            ),
        )

    def _b_no_signals(self, ctx: Dict) -> BotMessage:
        instruments = ctx.get("instruments", "los instrumentos monitoreados")
        regime      = ctx.get("regime", "actual")
        return BotMessage(
            ts=self._now(), kind="no_signals", icon="💤", level="info", label="",
            text=(
                f"He finalizado el análisis de {instruments}. "
                f"Actualmente, ningún activo cumple los parámetros de la estrategia "
                f"en el régimen **{regime}**. Seré paciente y esperaré la oportunidad correcta."
            ),
        )

    def _b_regime_change(self, ctx: Dict) -> BotMessage:
        old  = ctx.get("old", "—")
        new  = ctx.get("new", "—")
        prob = float(ctx.get("prob", 0)) * 100
        impl = REGIME_IMPLICATIONS.get(new, f"Nuevo régimen: {new}.")
        level = (
            "danger"  if new in ("CRASH", "STRONG_BEAR") else
            "warning" if new in ("BEAR", "EUPHORIA") else
            "success"
        )
        return BotMessage(
            ts=self._now(), kind="regime_change", icon="🔄", level=level, label="",
            text=(
                f"El mercado cambió de **{old}** a **{new}** "
                f"con **{prob:.0f}%** de confianza. {impl}"
            ),
        )

    def _b_signal_found(self, ctx: Dict) -> BotMessage:
        sym      = self._sym(ctx.get("symbol", "?"))
        strategy = ctx.get("strategy", "técnica")
        conf     = float(ctx.get("confidence", 0)) * 100
        return BotMessage(
            ts=self._now(), kind="signal_found", icon="📊", level="info", label="",
            text=(
                f"Detecté una señal potencial en **{sym}** con la estrategia "
                f"_{strategy}_ y **{conf:.0f}%** de confianza. "
                f"Verificando con los filtros de riesgo..."
            ),
        )

    def _b_signal_rejected(self, ctx: Dict) -> BotMessage:
        sym         = self._sym(ctx.get("symbol", "?"))
        reason      = ctx.get("reason", "")
        cash        = float(ctx.get("cash", 0))
        n_pos       = int(ctx.get("n_pos", 0))
        max_pos     = int(ctx.get("max_pos", 5))
        daily_count = int(ctx.get("daily_count", 0))
        max_trades  = int(ctx.get("max_trades", 10))
        size        = float(ctx.get("size", 0))
        cb_status   = ctx.get("cb_status", "HALT")

        if "LOCK_FILE" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="⛔", level="danger", label="",
                text=(
                    "El sistema está en **pausa de emergencia**. "
                    "Se requiere intervención manual para reanudar operaciones."
                ),
            )
        if "CIRCUIT_BREAKER" in reason or cb_status in ("HALT", "LOCKED"):
            msgs = {
                "HALT":   f"Encontré una señal en **{sym}**, pero el sistema tiene una pausa de seguridad activa. Protejo tu capital.",
                "LOCKED": "El sistema está **bloqueado** por protección máxima de capital. Se requiere intervención manual.",
            }
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="⏸", level="danger", label="",
                text=msgs.get(cb_status, msgs["HALT"]),
            )
        if "CASH_INSUFICIENTE" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="💰", level="warning", label="",
                text=(
                    f"¡Encontré una oportunidad óptima en **{sym}**! Sin embargo, "
                    f"el efectivo disponible (**${cash:.2f}**) no es suficiente para "
                    f"la inversión mínima requerida. La desestimo para proteger la cuenta."
                ),
            )
        if "MAX_CONCURRENT" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="📋", level="info", label="",
                text=(
                    f"Detecté una entrada potencial en **{sym}**, pero ya tengo "
                    f"**{n_pos}** posiciones abiertas (límite: {max_pos}). "
                    f"Esperaré a que se libere espacio antes de abrir más."
                ),
            )
        if "MAX_DAILY_TRADES" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="📊", level="info", label="",
                text=(
                    f"Buena señal en **{sym}**, pero ya alcancé el límite de "
                    f"**{daily_count}/{max_trades}** operaciones permitidas hoy. "
                    f"Continuaré mañana con energía renovada."
                ),
            )
        if "EXPOSICION_MAXIMA" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="⚖️", level="warning", label="",
                text=(
                    f"La nueva posición en **{sym}** superaría mi límite de exposición del 92% "
                    f"del portafolio. Priorizo la seguridad y diversificación."
                ),
            )
        if "DEBAJO_MINIMO" in reason or "MINIMO_ETORO" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="🔢", level="info", label="",
                text=(
                    f"La señal en **{sym}** resultó en un tamaño ajustado de **${size:.2f}**, "
                    f"por debajo del mínimo de $100 que requiere eToro. La omito."
                ),
            )
        if "CORRELACION" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="🔗", level="info", label="",
                text=(
                    f"No abro posición en **{sym}** porque está muy correlacionado "
                    f"con activos que ya tengo en el portafolio. Diversifico el riesgo."
                ),
            )
        if "STOP_LOSS" in reason:
            return BotMessage(
                ts=self._now(), kind="signal_rejected", icon="🛡️", level="warning", label="",
                text=(
                    f"La señal en **{sym}** no tiene un stop loss válido. "
                    f"Nunca opero sin protección de capital definida."
                ),
            )
        short = reason[:80] if reason else "filtros internos de riesgo"
        return BotMessage(
            ts=self._now(), kind="signal_rejected", icon="❌", level="info", label="",
            text=f"La señal en **{sym}** fue descartada: _{short}_.",
        )

    def _b_signal_approved(self, ctx: Dict) -> BotMessage:
        sym      = self._sym(ctx.get("symbol", "?"))
        size     = float(ctx.get("size", 0))
        strategy = ctx.get("strategy", "")
        mods     = ctx.get("modifications", [])
        mod_note = f" (ajustado: {', '.join(mods[:2])})" if mods else ""
        return BotMessage(
            ts=self._now(), kind="signal_approved", icon="✅", level="success", label="",
            text=(
                f"Señal aprobada para **{sym}**{mod_note}: "
                f"inversión de **${size:.2f}** con estrategia _{strategy}_. "
                f"Enviando orden al broker..."
            ),
        )

    def _b_order_executed(self, ctx: Dict) -> BotMessage:
        sym   = self._sym(ctx.get("symbol", "?"))
        price = float(ctx.get("price", 0))
        stop  = float(ctx.get("stop", 0))
        size  = float(ctx.get("size", 0))
        return BotMessage(
            ts=self._now(), kind="order_executed", icon="🎯", level="success", label="",
            text=(
                f"¡Orden ejecutada! Abrí posición en **{sym}** a **${price:.2f}** "
                f"por **${size:.2f}**. Stop loss fijado en ${stop:.2f} "
                f"para proteger la inversión."
            ),
        )

    def _b_order_failed(self, ctx: Dict) -> BotMessage:
        sym   = self._sym(ctx.get("symbol", "?"))
        error = str(ctx.get("error", "error desconocido"))[:80]
        return BotMessage(
            ts=self._now(), kind="order_failed", icon="⚠️", level="warning", label="",
            text=(
                f"Intenté abrir posición en **{sym}**, pero el broker respondió "
                f"con un error: _{error}_. Lo reintentaré en el próximo ciclo."
            ),
        )

    def _b_circuit_breaker(self, ctx: Dict) -> BotMessage:
        status = ctx.get("status", "HALT")
        dd_usd = float(ctx.get("dd_usd", 0))
        map_ = {
            "REDUCE_50": (
                "⚠️", "warning",
                f"La pérdida del día alcanzó **${dd_usd:.2f}**. "
                f"Reduciré el tamaño de las próximas posiciones al **50%** como medida preventiva.",
            ),
            "HALT": (
                "⏸", "danger",
                f"Pérdida diaria de **${dd_usd:.2f}** activó la pausa de seguridad. "
                f"No abriré nuevas posiciones hoy. Tu capital está protegido.",
            ),
            "LOCKED": (
                "🔒", "danger",
                f"Pérdida acumulada de **${dd_usd:.2f}** activó el bloqueo de emergencia. "
                f"Se requiere intervención manual para reanudar. Capital completamente protegido.",
            ),
            "OK": (
                "✅", "success",
                "El sistema se normalizó. Reanudando operaciones con parámetros estándar.",
            ),
        }
        icon, level, text = map_.get(status, ("⚠️", "warning", f"Circuit Breaker: {status}"))
        return BotMessage(ts=self._now(), kind="circuit_breaker", icon=icon, level=level, label="", text=text)

    def _b_hmm_retrained(self, ctx: Dict) -> BotMessage:
        n   = ctx.get("n_regimes", 4)
        bic = float(ctx.get("bic_score", 0))
        return BotMessage(
            ts=self._now(), kind="hmm_retrained", icon="🧠", level="info", label="",
            text=(
                f"Completé el reentrenamiento semanal del modelo de análisis de mercado "
                f"(**{n} regímenes**, BIC: {bic:.0f}). "
                f"El sistema está calibrado con los datos más recientes."
            ),
        )

    def _b_drawdown_warning(self, ctx: Dict) -> BotMessage:
        dd_usd = float(ctx.get("dd_usd", 0))
        dd_pct = float(ctx.get("dd_pct", 0))
        return BotMessage(
            ts=self._now(), kind="drawdown_warning", icon="📉", level="warning", label="",
            text=(
                f"El portafolio retrocedió **${dd_usd:.2f}** (**{dd_pct:.1f}%**) "
                f"desde su punto más alto. Monitoreando de cerca. "
                f"Reduciré el tamaño de posiciones si el retroceso continúa."
            ),
        )

    def _b_position_closed(self, ctx: Dict) -> BotMessage:
        sym     = self._sym(ctx.get("symbol", "?"))
        pnl     = float(ctx.get("pnl", 0))
        pnl_pct = float(ctx.get("pnl_pct", 0))
        sign    = "+" if pnl >= 0 else ""
        level   = "success" if pnl >= 0 else "warning"
        icon    = "🟢" if pnl >= 0 else "🔴"
        note    = "¡Buen resultado!" if pnl >= 0 else "Pérdida dentro de los parámetros de riesgo definidos."
        return BotMessage(
            ts=self._now(), kind="position_closed", icon=icon, level=level, label="",
            text=(
                f"La posición en **{sym}** fue cerrada con "
                f"**{sign}${abs(pnl):.2f}** ({sign}{pnl_pct:.1f}%). {note}"
            ),
        )

    def _b_api_failure(self, ctx: Dict) -> BotMessage:
        count = int(ctx.get("count", 1))
        return BotMessage(
            ts=self._now(), kind="api_failure", icon="📡", level="warning", label="",
            text=(
                f"Tuve **{count}** problema(s) consecutivos para conectarme a eToro. "
                f"Reintentando automáticamente. No tomaré decisiones de trading "
                f"hasta recuperar la conexión."
            ),
        )

    def _b_market_closed(self, ctx: Dict) -> BotMessage:
        return BotMessage(
            ts=self._now(), kind="market_closed", icon="🌙", level="info", label="",
            text=(
                "El mercado está **cerrado**. Mantendré un ojo en las posiciones "
                "abiertas y reanudaré el análisis completo cuando abra."
            ),
        )

    def _b_bot_startup(self, ctx: Dict) -> BotMessage:
        mode = ctx.get("mode", "dry-run")
        env  = ctx.get("env", "simulación")
        return BotMessage(
            ts=self._now(), kind="bot_startup", icon="🚀", level="success", label="",
            text=(
                f"Sistema iniciado en modo **{mode}** ({env}). "
                f"Cargando modelo de análisis de mercado y conectando a eToro. "
                f"Estaré listo para operar en segundos."
            ),
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _append(self, msg: BotMessage) -> None:
        with self._lock:
            try:
                msgs: List[Dict] = []
                if os.path.exists(self._path):
                    with open(self._path, encoding="utf-8") as fh:
                        msgs = json.load(fh)
                msgs.insert(0, asdict(msg))    # más reciente primero
                msgs = msgs[:MAX_MESSAGES]
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(msgs, fh, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.debug("[Comm] _append error: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance      : Optional[UserCommunicator] = None
_instance_lock : threading.Lock             = threading.Lock()


def get_communicator(root_dir: str = ".") -> UserCommunicator:
    """Retorna la instancia singleton de UserCommunicator (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = UserCommunicator(root_dir)
    return _instance
