"""
monitoring/ui_manager.py
Puente de datos thread-safe entre el bot y la futura UI web.

No tiene dependencias de Qt ni de ningún framework de UI.
La clase DataBridge actúa como un store central en memoria que:
  - Recibe actualizaciones desde el hilo del bot (push_*)
  - Expone el estado actual vía get_state() para la futura API HTTP
  - Lee state_snapshot.json de forma no bloqueante

Arquitectura prevista para el servidor GCP:
  [Bot polling thread] ──push──▶ DataBridge ──get_state()──▶ [FastAPI/Flask]
                                       │
                              state_snapshot.json (lectura async)
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Metadatos de regímenes HMM (reutilizados por la futura UI web) ───────────

REGIME_COPY: Dict[str, Dict[str, str]] = {
    "STRONG_BULL": {
        "title":    "Impulso alcista fuerte",
        "subtitle": "El bot está operando activamente",
        "color":    "bull",
        "icon":     "▲▲",
    },
    "EUPHORIA": {
        "title":    "Mercado en zona de euforia",
        "subtitle": "El bot opera con precaución selectiva",
        "color":    "bull",
        "icon":     "◆",
    },
    "WEAK_BULL": {
        "title":    "Tendencia alcista moderada",
        "subtitle": "El bot busca oportunidades con cautela",
        "color":    "bull",
        "icon":     "▲",
    },
    "NEUTRAL": {
        "title":    "Mercado sin dirección clara",
        "subtitle": "El bot espera una señal más fuerte",
        "color":    "neutral",
        "icon":     "—",
    },
    "BEAR": {
        "title":    "Mercado bajista",
        "subtitle": "El bot reduce la exposición al riesgo",
        "color":    "bear",
        "icon":     "▼",
    },
    "STRONG_BEAR": {
        "title":    "Caída fuerte del mercado",
        "subtitle": "El bot está protegiendo tu capital",
        "color":    "bear",
        "icon":     "▼▼",
    },
    "CRASH": {
        "title":    "Caída extrema detectada",
        "subtitle": "El bot ha pausado operaciones automáticamente",
        "color":    "crash",
        "icon":     "⬛",
    },
}

REGIME_COPY_DEFAULT: Dict[str, str] = {
    "title":    "Analizando el mercado…",
    "subtitle": "El bot está recopilando datos históricos",
    "color":    "neutral",
    "icon":     "○",
}

CB_COPY: Dict[str, Dict[str, str]] = {
    "OK":        {"label": "Operación normal",                     "level": "ok"},
    "REDUCE_50": {"label": "Riesgo reducido al 50% preventivamente", "level": "warn"},
    "HALT":      {"label": "Pausa de seguridad activada",           "level": "halt"},
    "LOCKED":    {"label": "Bloqueado — requiere intervención manual", "level": "halt"},
}


# ─────────────────────────────────────────────────────────────────────────────
# DataBridge: store central thread-safe para el estado del bot
# ─────────────────────────────────────────────────────────────────────────────

class DataBridge:
    """
    Store en memoria que recibe actualizaciones del hilo del bot y expone
    el estado actual para la futura capa web.

    Todos los métodos push_* son seguros para llamar desde cualquier hilo.
    get_state() devuelve una copia inmutable del estado actual como dict plano,
    lista para serializar a JSON sin ningún procesamiento adicional.
    flush() persiste el estado completo en live_state.json en cada tick del bot,
    de modo que el proceso Streamlit (separado) pueda leerlo sin acceso en memoria.
    """

    SNAPSHOT_FILE   = "state_snapshot.json"
    LIVE_STATE_FILE = "live_state.json"

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._portfolio: Optional[Dict[str, Any]] = None
        self._regime:    Optional[Dict[str, Any]] = None
        self._signals:   List[Dict[str, Any]]     = []
        self._online:    bool                     = False
        self._updated_at: Optional[str]           = None

    # ── Métodos push (llamados desde el hilo del bot) ─────────────────────────

    def push_portfolio(self, state: Any) -> None:
        """Almacena el último PortfolioState como dict serializable."""
        if state is None:
            return
        payload = {
            "equity":                  state.equity,
            "cash":                    state.cash,
            "daily_pnl":               state.daily_pnl,
            "weekly_pnl":              state.weekly_pnl,
            "peak_equity":             state.peak_equity,
            "drawdown_pct":            state.drawdown_pct,
            "circuit_breaker_status":  state.circuit_breaker_status,
            "daily_trades_count":      state.daily_trades_count,
            "positions_count":         len(state.positions),
            "positions":               state.positions,
        }
        with self._lock:
            self._portfolio  = payload
            self._updated_at = datetime.now(timezone.utc).isoformat()

    def push_regime(
        self,
        state: Any,
        flicker_count: int = 0,
        flicker_window: int = 20,
    ) -> None:
        """Almacena el último RegimeState como dict serializable."""
        if state is None:
            with self._lock:
                self._regime = None
            return
        copy = REGIME_COPY.get(state.label, REGIME_COPY_DEFAULT)
        payload = {
            "label":            state.label,
            "probability":      state.probability,
            "consecutive_bars": state.consecutive_bars,
            "is_confirmed":     state.is_confirmed,
            "flicker_count":    flicker_count,
            "flicker_window":   flicker_window,
            "copy":             copy,
        }
        with self._lock:
            self._regime = payload

    def push_signals(self, signals: List[Any]) -> None:
        """Almacena la lista de señales más recientes."""
        with self._lock:
            self._signals = list(signals)

    def push_connection(self, online: bool) -> None:
        """Actualiza el estado de conexión con la API de eToro."""
        with self._lock:
            self._online = online

    # ── Lectura del estado (llamado desde la futura capa web) ─────────────────

    def get_state(self) -> Dict[str, Any]:
        """
        Retorna una copia del estado completo como dict plano.
        Seguro para serializar a JSON directamente.
        No bloquea el hilo del bot (copia bajo lock mínimo).
        """
        with self._lock:
            return {
                "portfolio":  dict(self._portfolio) if self._portfolio else None,
                "regime":     dict(self._regime)    if self._regime    else None,
                "signals":    list(self._signals),
                "online":     self._online,
                "updated_at": self._updated_at,
            }

    def flush(self) -> None:
        """
        Escribe el estado completo en live_state.json (no bloqueante).
        Llamar desde _refresh_dashboard() en main.py tras cada tick.
        El proceso Streamlit (separado) lee este archivo para mostrar datos en vivo.
        """
        state = self.get_state()
        state["flushed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.LIVE_STATE_FILE, "w", encoding="utf-8") as fh:
                json.dump(state, fh, default=str)
        except OSError:
            pass

    def read_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Lee state_snapshot.json desde disco de forma no bloqueante.
        Retorna None si el archivo no existe o no es válido.
        Uso: la futura API web puede ofrecer /snapshot para historial de sesión.
        """
        path = self.SNAPSHOT_FILE
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def get_regime_copy(self, label: Optional[str]) -> Dict[str, str]:
        """Retorna los metadatos de copia UX para un label de régimen."""
        if label is None:
            return dict(REGIME_COPY_DEFAULT)
        return dict(REGIME_COPY.get(label, REGIME_COPY_DEFAULT))

    def get_cb_copy(self, status: str) -> Dict[str, str]:
        """Retorna los metadatos de copia UX para un estado de Circuit Breaker."""
        return dict(CB_COPY.get(status, CB_COPY["OK"]))
