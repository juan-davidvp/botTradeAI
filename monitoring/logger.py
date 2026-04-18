"""
monitoring/logger.py
Fase 8 — Logging JSON Estructurado con Archivos Rotativos.

Archivos:
  main.log   : eventos generales del sistema
  trades.log : ejecuciones, cambios de posición y rebalanceos
  alerts.log : historial de alertas disparadas
  regime.log : cambios de régimen y probabilidades HMM

Límites  : 10 MB por archivo | 30 archivos de backup (≈30 días)
Campos obligatorios por entrada:
  timestamp, regime, probability, equity, positions_count, daily_pnl
"""

import collections
import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LOG_DIR        = "logs"
MAX_BYTES      = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT   = 30
LOG_BUFFER_MAX = 500                  # entradas máximas en memoria


# ---------------------------------------------------------------------------
# Formateador JSON Estructurado
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Serializa cada LogRecord como una línea JSON con los campos obligatorios:
      timestamp, level, logger, message,
      regime, probability, equity, positions_count, daily_pnl
    Campos adicionales se pasan via extra={}.
    """

    REQUIRED_FIELDS = (
        "regime", "probability", "equity", "positions_count", "daily_pnl"
    )

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "level":            record.levelname,
            "logger":           record.name,
            "message":          record.getMessage(),
            # Campos de contexto del sistema de trading (default None)
            "regime":           getattr(record, "regime",           None),
            "probability":      getattr(record, "probability",      None),
            "equity":           getattr(record, "equity",           None),
            "positions_count":  getattr(record, "positions_count",  None),
            "daily_pnl":        getattr(record, "daily_pnl",        None),
        }

        # Campos opcionales de contexto
        for field in (
            "log_type", "trade_id", "position_id", "instrument_id",
            "amount_usd", "stop_loss_rate", "open_rate",
            "circuit_breaker", "alert_type", "strategy",
        ):
            val = getattr(record, field, None)
            if val is not None:
                entry[field] = val

        # Traceback si existe
        if record.exc_info:
            entry["exception"] = {
                "type":    record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Eliminar claves None para mantener JSON limpio
        entry = {k: v for k, v in entry.items() if v is not None}

        return json.dumps(entry, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Handler en memoria — puente hacia la futura UI web
# ---------------------------------------------------------------------------

class MemoryLogHandler(logging.Handler):
    """
    Almacena las últimas LOG_BUFFER_MAX entradas de log en un deque en memoria.
    Thread-safe (usa el lock interno de logging.Handler).
    La futura API web lee este buffer sin tocar los archivos de disco.
    """

    def __init__(self, maxlen: int = LOG_BUFFER_MAX) -> None:
        super().__init__()
        self._buffer: collections.deque = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level":     record.levelname,
                "logger":    record.name,
                "message":   record.getMessage(),
                "log_type":  getattr(record, "log_type", "main"),
            }
            # Campos de contexto opcionales
            for field in ("regime", "equity", "daily_pnl", "alert_type",
                          "trade_id", "instrument_id", "circuit_breaker"):
                val = getattr(record, field, None)
                if val is not None:
                    entry[field] = val
            self._buffer.append(entry)
        except Exception:
            self.handleError(record)

    def get_recent(self, n: int = 100, log_type: Optional[str] = None) -> List[Dict]:
        """
        Retorna hasta n entradas recientes.
        Si log_type es 'trade', 'alert', 'regime' o 'main', filtra por tipo.
        """
        records = list(self._buffer)
        if log_type:
            records = [r for r in records if r.get("log_type") == log_type]
        return records[-n:]


# Instancia singleton — inicializada por setup_logging()
_memory_handler: Optional[MemoryLogHandler] = None


def get_recent_logs(n: int = 100, log_type: Optional[str] = None) -> List[Dict]:
    """
    Retorna las últimas n entradas del buffer en memoria.
    Llamar desde la futura capa web (FastAPI/Flask) sin bloquear el bot.
    """
    if _memory_handler is None:
        return []
    return _memory_handler.get_recent(n=n, log_type=log_type)


# ---------------------------------------------------------------------------
# Filtros por tipo de log
# ---------------------------------------------------------------------------

class _TypeFilter(logging.Filter):
    def __init__(self, log_type: str):
        super().__init__()
        self._type = log_type

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "log_type", "main") == self._type


class MainFilter(logging.Filter):
    """Acepta todo lo que NO tiene log_type específico (va a main.log)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "log_type", "main") not in ("trade", "alert", "regime")


# ---------------------------------------------------------------------------
# Builder de handlers rotativos
# ---------------------------------------------------------------------------

def _make_handler(
    filename: str,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    level: int = logging.DEBUG,
    log_filter: Optional[logging.Filter] = None,
) -> logging.handlers.RotatingFileHandler:
    os.makedirs(LOG_DIR, exist_ok=True)
    path    = os.path.join(LOG_DIR, filename)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    if log_filter:
        handler.addFilter(log_filter)
    return handler


# ---------------------------------------------------------------------------
# Setup principal
# ---------------------------------------------------------------------------

def setup_logging(monitoring_cfg: Optional[Dict] = None) -> None:
    """
    Configura el sistema de logging con cuatro archivos rotativos + consola.
    Debe llamarse UNA SOLA VEZ al inicio de main.py.

    Parámetros
    ----------
    monitoring_cfg : dict
        Sección 'monitoring' de settings.yaml.
        Claves: log_dir, log_max_bytes, log_backup_count.
    """
    cfg          = monitoring_cfg or {}
    max_bytes    = int(cfg.get("log_max_bytes",    MAX_BYTES))
    backup_count = int(cfg.get("log_backup_count", BACKUP_COUNT))
    global LOG_DIR
    LOG_DIR      = cfg.get("log_dir", LOG_DIR)

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        # Ya configurado (evitar duplicados en tests)
        return
    root.setLevel(logging.DEBUG)

    # ── Consola (INFO+, formato legible por humanos) ─────────────────────────
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console_h)

    # ── Buffer en memoria (puente hacia la futura UI web) ────────────────────
    global _memory_handler
    _memory_handler = MemoryLogHandler(maxlen=LOG_BUFFER_MAX)
    _memory_handler.setLevel(logging.INFO)
    root.addHandler(_memory_handler)

    # ── main.log — eventos generales ────────────────────────────────────────
    root.addHandler(_make_handler(
        "main.log", max_bytes, backup_count,
        level=logging.DEBUG,
        log_filter=MainFilter(),
    ))

    # ── trades.log — ejecuciones y posiciones ───────────────────────────────
    root.addHandler(_make_handler(
        "trades.log", max_bytes, backup_count,
        level=logging.INFO,
        log_filter=_TypeFilter("trade"),
    ))

    # ── alerts.log — alertas disparadas ─────────────────────────────────────
    root.addHandler(_make_handler(
        "alerts.log", max_bytes, backup_count,
        level=logging.WARNING,
        log_filter=_TypeFilter("alert"),
    ))

    # ── regime.log — cambios de régimen HMM ─────────────────────────────────
    root.addHandler(_make_handler(
        "regime.log", max_bytes, backup_count,
        level=logging.INFO,
        log_filter=_TypeFilter("regime"),
    ))

    # Silenciar librerías ruidosas
    for lib in ("urllib3", "requests", "schedule", "hmmlearn"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configurado | dir=%s | max=%dMB | backup=%d archivos",
        LOG_DIR, max_bytes // (1024 * 1024), backup_count,
    )


# ---------------------------------------------------------------------------
# Helpers de contexto enriquecido
# ---------------------------------------------------------------------------

def _ctx(
    regime: Optional[str] = None,
    probability: Optional[float] = None,
    equity: Optional[float] = None,
    positions_count: Optional[int] = None,
    daily_pnl: Optional[float] = None,
    **extra,
) -> Dict[str, Any]:
    """Construye el dict extra para los métodos log_*."""
    d: Dict[str, Any] = {}
    if regime          is not None: d["regime"]          = regime
    if probability     is not None: d["probability"]     = round(probability, 4)
    if equity          is not None: d["equity"]          = round(equity, 2)
    if positions_count is not None: d["positions_count"] = positions_count
    if daily_pnl       is not None: d["daily_pnl"]       = round(daily_pnl, 2)
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# API pública de logging tipado
# ---------------------------------------------------------------------------

_log = logging.getLogger("regime_trader")


def log_trade(
    message: str,
    trade_id: Optional[str] = None,
    position_id: Optional[int] = None,
    instrument_id: Optional[int] = None,
    amount_usd: Optional[float] = None,
    stop_loss_rate: Optional[float] = None,
    open_rate: Optional[float] = None,
    regime: Optional[str] = None,
    probability: Optional[float] = None,
    equity: Optional[float] = None,
    positions_count: Optional[int] = None,
    daily_pnl: Optional[float] = None,
) -> None:
    """Emite un evento a trades.log con campos de trazabilidad completos."""
    extra = _ctx(regime, probability, equity, positions_count, daily_pnl,
                 log_type="trade")
    for k, v in (
        ("trade_id",      trade_id),
        ("position_id",   position_id),
        ("instrument_id", instrument_id),
        ("amount_usd",    amount_usd),
        ("stop_loss_rate",stop_loss_rate),
        ("open_rate",     open_rate),
    ):
        if v is not None:
            extra[k] = v
    _log.info(message, extra=extra)


def log_alert(
    message: str,
    alert_type: str = "GENERIC",
    level: str = "WARNING",
    regime: Optional[str] = None,
    probability: Optional[float] = None,
    equity: Optional[float] = None,
    positions_count: Optional[int] = None,
    daily_pnl: Optional[float] = None,
    **extra_fields,
) -> None:
    """Emite un evento a alerts.log."""
    extra = _ctx(regime, probability, equity, positions_count, daily_pnl,
                 log_type="alert", alert_type=alert_type)
    extra.update(extra_fields)
    fn = getattr(_log, level.lower(), _log.warning)
    fn(message, extra=extra)


def log_regime_change(
    old_regime: str,
    new_regime: str,
    probability: float,
    equity: Optional[float] = None,
    positions_count: Optional[int] = None,
    daily_pnl: Optional[float] = None,
    consecutive_bars: Optional[int] = None,
    is_confirmed: bool = False,
) -> None:
    """Emite un cambio de régimen a regime.log."""
    extra = _ctx(new_regime, probability, equity, positions_count, daily_pnl,
                 log_type="regime",
                 old_regime=old_regime,
                 consecutive_bars=consecutive_bars,
                 is_confirmed=is_confirmed)
    _log.warning(
        "CAMBIO DE RÉGIMEN: %s → %s (%.1f%%)",
        old_regime, new_regime, probability * 100,
        extra=extra,
    )


def log_regime_update(
    regime: str,
    probability: float,
    equity: Optional[float] = None,
    positions_count: Optional[int] = None,
    daily_pnl: Optional[float] = None,
    consecutive_bars: Optional[int] = None,
    is_confirmed: bool = False,
) -> None:
    """Emite actualización periódica de régimen a regime.log."""
    extra = _ctx(regime, probability, equity, positions_count, daily_pnl,
                 log_type="regime",
                 consecutive_bars=consecutive_bars,
                 is_confirmed=is_confirmed)
    _log.info(
        "RÉGIMEN: %s %.1f%% | barras=%s | confirmado=%s",
        regime, probability * 100, consecutive_bars, is_confirmed,
        extra=extra,
    )


def log_circuit_breaker(
    breaker_type: str,
    dd_usd: float,
    dd_pct: float,
    equity: float,
    regime: str,
    positions_closed: Optional[List[str]] = None,
) -> None:
    """Emite activación de circuit breaker a alerts.log."""
    extra = _ctx(regime, None, equity,
                 log_type="alert",
                 alert_type=f"CB_{breaker_type}",
                 circuit_breaker=breaker_type,
                 dd_usd=dd_usd,
                 dd_pct=round(dd_pct, 4),
                 positions_closed=positions_closed or [])
    _log.error(
        "CIRCUIT BREAKER [%s] | DD=$%.2f (%.2f%%) | equity=$%.2f",
        breaker_type, dd_usd, dd_pct * 100, equity,
        extra=extra,
    )
