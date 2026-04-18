"""
monitoring/dashboard.py

Dos responsabilidades en un solo archivo:

  1. Clase `Dashboard` (stub) — importada por main.py para emitir un log
     de una línea por tick. Sin dependencias de Streamlit.

  2. Aplicación Streamlit — lanzar con:
         streamlit run monitoring/dashboard.py
     Lee live_state.json (escrito por DataBridge.flush() cada 30 s) y
     logs/*.log desde disco; se refresca cada 5 segundos automáticamente.
     Si el bot no está corriendo, muestra un botón para iniciarlo.

Secciones del dashboard (espejo de la consola Rich):
  REGIME · PORTFOLIO · POSITIONS · CONCENTRACIÓN
  RECENT SIGNALS · RISK STATUS · ROI TRACKER · SYSTEM · TERMINAL
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de UX
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 560.05
TARGET_EQUITY  = 655.37
START_DATE     = datetime(2026, 4, 16, tzinfo=timezone.utc)
TARGET_DATE    = datetime(2026, 8, 12, tzinfo=timezone.utc)
MONTHLY_RATE   = 0.0466

INSTRUMENT_NAMES: Dict[int, str] = {
    4238:   "VOO · S&P 500",
    14328:  "MELI · MercadoLibre",
    9408:   "PLTR · Palantir",
    6218:   "APP · AppLovin",
    2488:   "DAVE · Dave Inc.",
    100000: "BTC/USD",
}

REGIME_COPY: Dict[str, Dict[str, str]] = {
    "STRONG_BULL": {"title": "⬆️ Impulso alcista fuerte",   "sub": "El bot está operando activamente",         "color": "#10B981"},
    "EUPHORIA":    {"title": "🔶 Zona de euforia",           "sub": "El bot opera con precaución selectiva",    "color": "#F59E0B"},
    "WEAK_BULL":   {"title": "📈 Tendencia alcista moderada","sub": "El bot busca oportunidades con cautela",   "color": "#10B981"},
    "NEUTRAL":     {"title": "➡️ Sin dirección clara",       "sub": "El bot espera una señal más fuerte",       "color": "#7B8BB2"},
    "BEAR":        {"title": "📉 Mercado bajista",            "sub": "El bot reduce la exposición al riesgo",    "color": "#F59E0B"},
    "STRONG_BEAR": {"title": "🔻 Caída fuerte",              "sub": "El bot está protegiendo tu capital",       "color": "#EF4444"},
    "CRASH":       {"title": "🚨 Caída extrema detectada",   "sub": "El bot pausó operaciones automáticamente", "color": "#EF4444"},
}
REGIME_DEFAULT = {"title": "🔍 Analizando el mercado…", "sub": "Recopilando datos históricos", "color": "#448AFF"}

CB_COPY: Dict[str, Dict[str, str]] = {
    "OK":        {"label": "✅ Operación normal",                   "color": "#10B981"},
    "REDUCE_50": {"label": "⚠️ Riesgo reducido al 50%",             "color": "#F59E0B"},
    "HALT":      {"label": "⏸ Pausa de seguridad activada",         "color": "#EF4444"},
    "LOCKED":    {"label": "🔒 Bloqueado — intervención requerida", "color": "#EF4444"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Stub: importado por main.py sin dependencias de Streamlit
# ─────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """Stub no-visual. Emite un log de una línea por llamada a render()."""

    def __init__(self, settings: Optional[dict] = None) -> None:
        pass

    def render(
        self,
        portfolio_state: Any,
        hmm_state: Any = None,
        signals: Optional[List] = None,
    ) -> None:
        if portfolio_state is None:
            return
        regime = hmm_state.label      if hmm_state else "—"
        prob   = hmm_state.probability if hmm_state else 0.0
        logger.info(
            "[Dashboard] equity=$%.2f | régimen=%s (%.0f%%) | DD=%.2f%% "
            "| hoy=%+.2f | pos=%d | CB=%s",
            portfolio_state.equity,
            regime, prob * 100,
            portfolio_state.drawdown_pct * 100,
            portfolio_state.daily_pnl,
            len(portfolio_state.positions),
            portfolio_state.circuit_breaker_status,
        )

    def run_live(self, *args, **kwargs) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de lectura de datos (usados únicamente por la app Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def _load_live_state(path: str = "live_state.json") -> Dict:
    """Lee live_state.json escrito por DataBridge.flush() en cada tick del bot."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _bot_alive(state: Dict, max_age_s: int = 90) -> bool:
    """True si el bot escribió live_state.json hace menos de max_age_s segundos."""
    ts = state.get("flushed_at") or state.get("updated_at", "")
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() < max_age_s
    except Exception:
        return False


def _load_settings(path: str = "config/settings.yaml") -> Dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _load_log_lines(log_path: str, n: int = 40) -> List[Dict]:
    """Lee las últimas n líneas JSON de un archivo de log rotativo."""
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.readlines()
        result: List[Dict] = []
        for line in reversed(raw[-400:]):
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except Exception:
                result.append({"message": line, "level": "INFO", "timestamp": "", "log_type": "main"})
            if len(result) >= n:
                break
        return result  # más reciente primero
    except Exception:
        return []




# ─────────────────────────────────────────────────────────────────────────────
# Aplicación Streamlit  —  Midnight Blue Edition (Stitch design)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(
        page_title="AgentBotTrade · Terminal",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Auto-refresco cada 5 s ────────────────────────────────────────────────
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5_000, key="bot_refresh")
    except ImportError:
        st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)

    state     = _load_live_state(os.path.join(_ROOT, "live_state.json"))
    cfg       = _load_settings(os.path.join(_ROOT, "config/settings.yaml"))
    main_logs = _load_log_lines(os.path.join(_ROOT, "logs/main.log"),   n=50)
    trd_logs  = _load_log_lines(os.path.join(_ROOT, "logs/trades.log"), n=20)

    portfolio = state.get("portfolio") or {}
    regime_d  = state.get("regime")    or {}
    signals_d = state.get("signals")   or []
    alive     = _bot_alive(state)

    equity       = float(portfolio.get("equity",       0) or 0)
    cash         = float(portfolio.get("cash",         0) or 0)
    daily_pnl    = float(portfolio.get("daily_pnl",    0) or 0)
    weekly_pnl   = float(portfolio.get("weekly_pnl",   0) or 0)
    peak_equity  = float(portfolio.get("peak_equity",  0) or 0)
    drawdown_pct = float(portfolio.get("drawdown_pct", 0) or 0)
    cb_status    = portfolio.get("circuit_breaker_status", "OK")
    daily_trades = int(portfolio.get("daily_trades_count", 0))
    positions    = portfolio.get("positions", [])

    regime_label   = regime_d.get("label")
    regime_prob    = float(regime_d.get("probability",      0) or 0)
    consec_bars    = int(regime_d.get("consecutive_bars",   0) or 0)
    is_confirmed   = bool(regime_d.get("is_confirmed",      False))
    flicker_count  = int(regime_d.get("flicker_count",      0) or 0)
    flicker_window = int(regime_d.get("flicker_window",     20) or 20)
    regime_info    = REGIME_COPY.get(regime_label, REGIME_DEFAULT)

    risk_cfg      = cfg.get("risk", {})
    init_equity   = float(risk_cfg.get("initial_equity", INITIAL_EQUITY))
    target_equity = float(risk_cfg.get("target_equity",  TARGET_EQUITY))
    env_label     = cfg.get("broker", {}).get("environment", "").upper()

    if equity == 0:
        equity = init_equity
    since_start    = equity - init_equity
    total_invested = sum(float(p.get("amount", 0) or 0) for p in positions)
    allocation_pct = (total_invested / (total_invested + cash) * 100) if equity > 0 else 0.0
    dd_abs         = abs(drawdown_pct * peak_equity) if peak_equity > 0 else max(0.0, peak_equity - equity)
    progress_pct   = max(0.0, min(100.0,
        (equity - init_equity) / (target_equity - init_equity) * 100
    )) if target_equity > init_equity else 0.0

    now_utc    = datetime.now(timezone.utc)
    last_ts    = state.get("flushed_at") or state.get("updated_at") or ""
    last_time  = last_ts[11:19] if len(last_ts) >= 19 else "—"
    flushed    = state.get("flushed_at") or "—"
    poll_ts    = flushed[11:19] if len(flushed) >= 19 else "—"
    api_ok     = state.get("online", False)

    daily_halt_pct  = float(risk_cfg.get("daily_dd_halt",    0.03))
    weekly_halt_pct = float(risk_cfg.get("weekly_dd_halt",   0.07))
    peak_dd_pct     = float(risk_cfg.get("max_dd_from_peak", 0.10))
    daily_limit     = init_equity * daily_halt_pct
    weekly_limit    = init_equity * weekly_halt_pct
    peak_limit      = init_equity * peak_dd_pct
    daily_dd_abs    = max(0.0, -daily_pnl)
    weekly_dd_abs   = max(0.0, -weekly_pnl)
    peak_dd_abs     = dd_abs

    days_total     = (TARGET_DATE - START_DATE).days
    days_elapsed   = max(0, (now_utc - START_DATE).days)
    days_left      = max(0, days_total - days_elapsed)
    months_elapsed = days_elapsed / 30.44
    target_today   = init_equity * ((1 + MONTHLY_RATE) ** months_elapsed)
    on_track       = equity >= target_today
    prob_bar       = int(regime_prob * 100)

    # ── Helpers de color ─────────────────────────────────────────────────────
    def _pnl_c(v: float) -> str:
        return "#5ad7e8" if v >= 0 else "#ffb4ab"

    def _dd_bar_pct(val: float, limit: float) -> int:
        return min(100, int(val / limit * 100)) if limit > 0 else 0

    def _dd_bar_color(pct: int) -> str:
        if pct < 50: return "#5ad7e8"
        if pct < 80: return "#ffb787"
        return "#ffb4ab"

    daily_color  = _pnl_c(daily_pnl)
    weekly_color = _pnl_c(weekly_pnl)
    since_color  = _pnl_c(since_start)
    daily_sign   = "+" if daily_pnl  >= 0 else ""
    weekly_sign  = "+" if weekly_pnl >= 0 else ""
    since_sign   = "+" if since_start >= 0 else ""
    daily_pct_d  = (daily_pnl / init_equity * 100) if init_equity > 0 else 0.0
    since_pct    = (since_start / init_equity * 100) if init_equity > 0 else 0.0

    dd_dp = _dd_bar_pct(daily_dd_abs, daily_limit)
    dd_wp = _dd_bar_pct(weekly_dd_abs, weekly_limit)
    dd_pp = _dd_bar_pct(peak_dd_abs,  peak_limit)
    dd_dc = _dd_bar_color(dd_dp)
    dd_wc = _dd_bar_color(dd_wp)
    dd_pc = _dd_bar_color(dd_pp)

    cb_status_color = {"OK": "#5ad7e8", "REDUCE_50": "#ffb787", "HALT": "#ffb4ab", "LOCKED": "#ffb4ab"}.get(cb_status, "#bcc9cb")
    cb_label_clean  = CB_COPY.get(cb_status, CB_COPY["OK"])["label"].replace("✅ ", "").replace("⚠️ ", "").replace("⏸ ", "").replace("🔒 ", "")

    flicker_color = "#ffb4ab" if flicker_count > 3 else "#bcc9cb"
    conf_text     = "Confirmado" if is_confirmed else "En validación"
    conf_color    = "#5ad7e8" if is_confirmed else "#bcc9cb"
    api_color     = "#5ad7e8" if api_ok else "#ffb4ab"
    api_label     = "Online" if api_ok else "Offline"
    roi_color     = "#5ad7e8" if on_track else "#ffb787"

    live_dot_color = "#5ad7e8" if alive else "#ffb4ab"
    live_label     = "En vivo" if alive else "Sin datos"
    env_color      = "#ffb4ab" if env_label == "REAL" else "#5ad7e8"
    env_bg         = "rgba(255,180,171,0.1)" if env_label == "REAL" else "rgba(90,215,232,0.1)"
    env_border     = "rgba(255,180,171,0.3)" if env_label == "REAL" else "rgba(90,215,232,0.3)"

    # Regime title split for gradient
    raw_title = regime_info.get("title", "Analizando…")
    for emoji in ("⬆️ ", "🔶 ", "📈 ", "➡️ ", "📉 ", "🔻 ", "🚨 ", "🔍 "):
        raw_title = raw_title.replace(emoji, "")
    parts = raw_title.rsplit(" ", 1)
    regime_prefix    = parts[0] if len(parts) == 2 else ""
    regime_highlight = parts[-1]

    n_pos       = len(positions)
    n_pos_label = f"Posiciones Activas ({n_pos})"

    # ── Construir HTML de filas de posiciones ─────────────────────────────────
    pos_rows_html = ""
    for pos in positions:
        instr_id   = int(pos.get("instrumentId") or pos.get("instrumentID") or 0)
        pos_id_str = str(pos.get("positionId") or pos.get("positionID", ""))[:10]
        open_rate  = float(pos.get("openRate",      0) or 0)
        curr_price = float(pos.get("current_price", 0) or open_rate)
        sl_rate    = float(pos.get("stopLossRate",  0) or 0)
        no_stop    = bool(pos.get("isNoStopLoss", False))
        holding    = int(pos.get("holding_days",    0) or 0)
        amount     = float(pos.get("amount",        0) or 0)
        upnl       = float(pos.get("unrealized_pnl",0) or 0)
        pnl_pct    = (upnl / amount * 100) if amount > 0 else 0.0
        stop_ok    = not (no_stop or sl_rate <= 0.001)
        entry_reg  = pos.get("entry_regime", "—")
        pname      = INSTRUMENT_NAMES.get(instr_id, f"#{instr_id}").split("·")[0].strip()
        pnl_sign   = "+" if pnl_pct >= 0 else ""
        pnl_c      = "#5ad7e8" if pnl_pct >= 0 else "#ffb4ab"
        stop_icon  = "shield" if stop_ok else "warning"
        stop_ic    = "#bcc9cb" if stop_ok else "#ffb4ab"
        pos_rows_html += (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);" '
            f'onmouseover="this.style.background=\'rgba(255,255,255,0.03)\'" '
            f'onmouseout="this.style.background=\'transparent\'">'
            f'<td style="padding:12px 16px;font-weight:700;color:#dee3e4;">{pname}</td>'
            f'<td style="padding:12px 16px;color:#bcc9cb;font-size:11px;">{pos_id_str}</td>'
            f'<td style="padding:12px 16px;text-align:right;">${open_rate:,.2f}</td>'
            f'<td style="padding:12px 16px;text-align:right;">${curr_price:,.2f}</td>'
            f'<td style="padding:12px 16px;text-align:right;color:{pnl_c};">{pnl_sign}{pnl_pct:.2f}%</td>'
            f'<td style="padding:12px 16px;text-align:center;"><span class="msym" style="color:{stop_ic};">{stop_icon}</span></td>'
            f'<td style="padding:12px 16px;text-align:center;">{holding}</td>'
            f'<td style="padding:12px 16px;"><span style="padding:3px 8px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:4px;font-size:10px;color:#bcc9cb;">{entry_reg}</span></td>'
            f'</tr>\n'
        )
    if not pos_rows_html:
        pos_rows_html = '<tr><td colspan="8" style="padding:24px;text-align:center;color:#bcc9cb;font-size:13px;">El bot está buscando la oportunidad perfecta.</td></tr>'

    # ── Construir HTML de badges de concentración ─────────────────────────────
    conc_html = ""
    if positions and total_invested > 0:
        conc_items: dict = {}
        for pos in positions:
            iid = int(pos.get("instrumentId") or pos.get("instrumentID") or 0)
            amt = float(pos.get("amount", 0) or 0)
            conc_items[iid] = conc_items.get(iid, 0) + amt
        for iid, amt in sorted(conc_items.items(), key=lambda x: x[1], reverse=True):
            pct   = amt / total_invested * 100
            cname = INSTRUMENT_NAMES.get(iid, f"#{iid}").split("·")[0].strip()
            if pct > 20:
                bc, bb, bbd = "#ffb4ab", "rgba(255,180,171,0.1)", "rgba(255,180,171,0.3)"
            elif pct > 15:
                bc, bb, bbd = "#ffb787", "rgba(255,183,135,0.1)", "rgba(255,183,135,0.3)"
            else:
                bc, bb, bbd = "#5ad7e8", "rgba(90,215,232,0.1)", "rgba(90,215,232,0.3)"
            warn = " ⚠" if pct > 20 else ""
            conc_html += (
                f'<span style="padding:4px 12px;background:{bb};border:1px solid {bbd};'
                f'border-radius:8px;font-size:11px;font-family:\'Roboto Mono\',monospace;'
                f'color:{bc};">{cname} {pct:.1f}%{warn}</span> '
            )
    if not conc_html:
        conc_html = '<span style="color:#bcc9cb;font-size:12px;">Sin posiciones activas</span>'

    # ── Construir HTML de señales ─────────────────────────────────────────────
    signals_html = ""
    if signals_d:
        for s in signals_d[:6]:
            iid    = int(s.get("instrument_id", 0) or 0)
            sname  = INSTRUMENT_NAMES.get(iid, f"#{iid}").split("·")[0].strip()
            action = s.get("action", "—")
            ac     = "#5ad7e8" if action == "BUY" else ("#ffb4ab" if action == "SELL" else "#bcc9cb")
            susd   = f"${float(s.get('position_size_usd', 0)):,.2f}"
            strat  = s.get("strategy_name", "—")[:22]
            qual   = s.get("quality_score", "—")
            signals_html += (
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
                f'<td style="padding:10px 14px;font-weight:700;color:#dee3e4;">{sname}</td>'
                f'<td style="padding:10px 14px;color:{ac};font-weight:600;">{action}</td>'
                f'<td style="padding:10px 14px;text-align:right;">{susd}</td>'
                f'<td style="padding:10px 14px;color:#bcc9cb;font-size:11px;">{strat}</td>'
                f'<td style="padding:10px 14px;text-align:center;color:#5ad7e8;">{qual}</td>'
                f'</tr>\n'
            )
    elif trd_logs:
        for entry in trd_logs[:5]:
            iid   = entry.get("instrument_id")
            sname = INSTRUMENT_NAMES.get(int(iid), f"#{iid}").split("·")[0].strip() if iid else "—"
            amt   = entry.get("amount_usd")
            ts_r  = entry.get("timestamp", "")
            ts_s  = ts_r[11:19] if len(ts_r) >= 19 else "—"
            msg   = entry.get("message", "")[:55]
            signals_html += (
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
                f'<td style="padding:10px 14px;color:#bcc9cb;font-size:11px;">{ts_s}</td>'
                f'<td style="padding:10px 14px;font-weight:700;color:#dee3e4;">{sname}</td>'
                f'<td colspan="2" style="padding:10px 14px;color:#bcc9cb;font-size:11px;">{msg}</td>'
                f'<td style="padding:10px 14px;text-align:right;">{f"${float(amt):,.2f}" if amt else "—"}</td>'
                f'</tr>\n'
            )
    if not signals_html:
        signals_html = '<tr><td colspan="5" style="padding:20px;text-align:center;color:#bcc9cb;">Sin señales recientes</td></tr>'

    # ── Construir HTML del terminal ───────────────────────────────────────────
    _LVL_COLOR = {"INFO": "#4ADE80", "WARNING": "#FFD700", "ERROR": "#ffb4ab", "DEBUG": "#5ad7e8", "CRITICAL": "#ffb4ab"}
    terminal_html = ""
    for entry in main_logs[:25]:
        ts_r  = entry.get("timestamp", "")
        ts_s  = ts_r[11:19] if len(ts_r) >= 19 else "—:—:—"
        level = entry.get("level", "INFO")
        lc    = _LVL_COLOR.get(level, "#4ADE80")
        msg   = entry.get("message", "").replace("<", "&lt;").replace(">", "&gt;")[:130]
        mod   = (entry.get("logger") or "").split(".")[-1][:14]
        terminal_html += (
            f'<div style="line-height:1.65;">'
            f'<span style="color:#3d494b;">[{ts_s}]</span> '
            f'<span style="color:{lc};">›</span> '
            f'<span style="color:#5ad7e8;">{mod}:</span> '
            f'<span style="color:#a8b5b7;">{msg}</span></div>\n'
        )
    if not terminal_html:
        terminal_html = '<div style="color:#3d494b;font-style:italic;">Sin registros en esta sesión.</div>'

    # ── Bot launcher (Streamlit nativo — antes del bloque HTML) ───────────────
    if not alive:
        col_w, col_b, _ = st.columns([4, 1, 1])
        with col_w:
            st.markdown(
                f'<div style="background:rgba(255,183,135,0.06);border:1px solid rgba(255,183,135,0.2);'
                f'border-radius:10px;padding:10px 16px;font-size:12px;color:#ffb787;margin-bottom:4px;">'
                f'⚠️ Bot detenido · último snapshot: {last_time} · Para modo real: '
                f'<code style="color:#5ad7e8;background:transparent;">python main.py</code></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            if "bot_proc" not in st.session_state:
                st.session_state["bot_proc"] = None
            if st.button("▶ Dry-Run", type="primary"):
                try:
                    proc = subprocess.Popen(
                        [sys.executable, os.path.join(_ROOT, "main.py"), "--dry-run"],
                        cwd=_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    st.session_state["bot_proc"] = proc.pid
                    st.success(f"PID {proc.pid}")
                except Exception as _e:
                    st.error(str(_e))

    # ── CSS injection (separada para evitar conflictos de renderizado) ──────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=Roboto+Mono:wght@400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

:root {
  --pr: #5ad7e8; --pr-dim: #07a8b8; --sc: #a3ced5;
  --er: #ffb4ab; --wa: #ffb787;
  --os: #dee3e4; --osv: #bcc9cb;
  --bg: #070F2B; --card: #0B0E14;
  --sf: rgba(255,255,255,0.08); --ol: rgba(255,255,255,0.1);
  --fm: 'Roboto Mono', monospace; --fb: 'Inter', sans-serif;
}
html, body, [class*="css"], .stApp, .block-container,
section[data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--os) !important;
  font-family: var(--fb);
}
.block-container {
  padding: 72px 1.5rem 60px !important;
  max-width: 1640px !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display:none !important; visibility:hidden !important;
}
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:rgba(90,215,232,0.2); border-radius:2px; }
.msym {
  font-family: 'Material Symbols Outlined';
  font-weight: normal; font-style: normal;
  display: inline-block; vertical-align: middle; line-height: 1;
  font-variation-settings: 'FILL' 0, 'wght' 300;
  font-size: 18px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
.gc {
  background: var(--card);
  background-image: linear-gradient(135deg,rgba(255,255,255,0.05) 0%,transparent 100%);
  border: 1px solid var(--ol);
  border-radius: 12px;
  backdrop-filter: blur(12px);
}
.gt {
  background: linear-gradient(to right,var(--pr),var(--pr-dim));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.stButton > button {
  background: rgba(90,215,232,0.08) !important;
  color: #5ad7e8 !important;
  border: 1px solid rgba(90,215,232,0.25) !important;
  border-radius: 8px !important;
  font-family: var(--fm) !important;
  font-size: 12px !important; font-weight: 600 !important;
  padding: 6px 16px !important; transition: all 0.2s !important;
}
.stButton > button:hover {
  background: rgba(90,215,232,0.18) !important;
  box-shadow: 0 0 10px rgba(90,215,232,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Renderizado principal — usa st.html() en Streamlit ≥ 1.31 ─────────────
    _dash_html = f"""
<!-- ═══ TOP NAV ═══════════════════════════════════════════════════════════ -->
<nav style="position:fixed;top:0;left:0;width:100%;z-index:9999;
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 24px;background:rgba(7,15,43,0.88);
  backdrop-filter:blur(14px);border-bottom:1px solid rgba(255,255,255,0.08);
  box-shadow:0 0 15px rgba(90,215,232,0.05);">
  <div style="display:flex;align-items:center;gap:14px;">
    <span style="font-size:16px;font-weight:900;color:#e1e2eb;letter-spacing:-0.3px;">AgentBotTrade</span>
    <div style="display:flex;align-items:center;gap:6px;padding:3px 10px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:20px;">
      <span style="width:6px;height:6px;border-radius:50%;background:{live_dot_color};display:inline-block;animation:pulse 2s infinite;"></span>
      <span style="font-size:10px;font-family:var(--fm);color:var(--osv);text-transform:uppercase;letter-spacing:1px;">{live_label}</span>
    </div>
    <div style="padding:3px 10px;background:{env_bg};border:1px solid {env_border};border-radius:20px;">
      <span style="font-size:10px;font-family:var(--fm);color:{env_color};text-transform:uppercase;letter-spacing:1px;">{env_label or 'DRY-RUN'}</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:20px;">
    <span style="font-size:10px;font-family:var(--fm);color:var(--osv);">
      SYS UTC: {now_utc.strftime('%H:%M:%S')} &nbsp;·&nbsp; ÚLTIMO: {last_time}
    </span>
    <span class="msym" style="color:var(--osv);">sensors</span>
    <span class="msym" style="color:var(--osv);">schedule</span>
    <span class="msym" style="color:var(--osv);">settings</span>
  </div>
</nav>

<!-- ═══ HMM HERO ══════════════════════════════════════════════════════════ -->
<section class="gc" style="position:relative;overflow:hidden;padding:28px 32px;
  display:flex;align-items:center;justify-content:space-between;
  gap:24px;flex-wrap:wrap;margin-bottom:20px;
  box-shadow:0 0 20px rgba(90,215,232,0.08);">
  <div style="position:absolute;inset:0;background:linear-gradient(to right,rgba(90,215,232,0.07),transparent);pointer-events:none;"></div>
  <div style="position:relative;z-index:1;display:flex;flex-direction:column;gap:6px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span class="msym" style="font-size:26px;color:var(--pr);">psychology</span>
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:2px;">Modelo Oculto de Markov (HMM)</span>
    </div>
    <h1 style="font-size:38px;font-weight:900;color:var(--os);line-height:1.1;margin:4px 0 2px;">
      {regime_prefix} <span class="gt">{regime_highlight}</span>
    </h1>
    <div style="font-size:12px;color:var(--osv);">{regime_info.get('sub', '')}</div>
  </div>
  <div style="position:relative;z-index:1;display:flex;gap:28px;flex-wrap:wrap;">
    <div style="display:flex;flex-direction:column;gap:3px;">
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Confianza</span>
      <span style="font-size:26px;font-weight:700;font-family:var(--fm);color:var(--pr);text-shadow:0 0 10px rgba(90,215,232,0.45);">{prob_bar}%</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:3px;">
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Estabilidad</span>
      <span style="font-size:20px;font-weight:600;font-family:var(--fm);color:var(--os);">{consec_bars} barras</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:3px;">
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Flicker</span>
      <span style="font-size:20px;font-weight:600;font-family:var(--fm);color:{flicker_color};">{flicker_count}/{flicker_window}</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:6px;">
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Estado</span>
      <div style="padding:4px 12px;background:rgba(90,215,232,0.07);border:1px solid rgba(90,215,232,0.22);border-radius:20px;display:flex;align-items:center;gap:6px;">
        <span style="width:6px;height:6px;border-radius:50%;background:{conf_color};display:inline-block;animation:pulse 2s infinite;"></span>
        <span style="font-size:10px;font-family:var(--fm);color:{conf_color};text-transform:uppercase;letter-spacing:1px;">{conf_text}</span>
      </div>
    </div>
  </div>
</section>

<!-- ═══ MAIN GRID ══════════════════════════════════════════════════════════ -->
<div style="display:grid;grid-template-columns:minmax(0,3fr) minmax(0,9fr);gap:18px;margin-bottom:18px;">

  <!-- LEFT COLUMN -->
  <div style="display:flex;flex-direction:column;gap:14px;">

    <div class="gc" style="padding:20px;position:relative;overflow:hidden;">
      <div style="position:absolute;right:-10px;top:-10px;width:72px;height:72px;background:rgba(90,215,232,0.05);border-radius:50%;filter:blur(18px);"></div>
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Equity Actual</span>
      <div style="font-size:30px;font-weight:700;font-family:var(--fm);color:var(--os);margin:5px 0 3px;">${equity:,.2f}</div>
      <div style="font-size:11px;font-family:var(--fm);color:var(--osv);">Obj: <span style="color:var(--pr);">${target_equity:,.2f}</span></div>
    </div>

    <div class="gc" style="padding:16px 20px;">
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">P&amp;L Hoy</span>
      <div style="display:flex;align-items:baseline;gap:8px;margin-top:5px;">
        <span style="font-size:21px;font-weight:700;font-family:var(--fm);color:{daily_color};">{daily_sign}${abs(daily_pnl):,.2f}</span>
        <span style="font-size:12px;font-family:var(--fm);color:{daily_color};opacity:0.7;">({daily_sign}{abs(daily_pct_d):.2f}%)</span>
      </div>
    </div>

    <div class="gc" style="padding:16px 20px;">
      <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">P&amp;L Desde Inicio</span>
      <div style="display:flex;align-items:baseline;gap:8px;margin-top:5px;">
        <span style="font-size:21px;font-weight:700;font-family:var(--fm);color:{since_color};">{since_sign}${abs(since_start):,.2f}</span>
        <span style="font-size:12px;font-family:var(--fm);color:{since_color};opacity:0.7;">({since_sign}{abs(since_pct):.2f}%)</span>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
      <div class="gc" style="padding:14px;">
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Cash Libre</span>
        <div style="font-size:17px;font-weight:700;font-family:var(--fm);color:var(--os);margin-top:4px;">${cash:,.2f}</div>
      </div>
      <div class="gc" style="padding:14px;">
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Leverage</span>
        <div style="font-size:17px;font-weight:700;font-family:var(--fm);color:var(--os);margin-top:4px;">1.0x</div>
      </div>
    </div>

    <!-- Drawdown & Riesgo -->
    <div class="gc" style="padding:18px 20px;display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;align-items:center;gap:7px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.08);">
        <span class="msym" style="font-size:16px;color:var(--osv);">warning</span>
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Drawdown &amp; Riesgo</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
          <div>
            <div style="font-size:9px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">DD Diario</div>
            <div style="font-size:12px;font-family:var(--fm);color:{dd_dc};">${daily_dd_abs:,.2f} <span style="color:var(--osv);font-size:10px;">/ ${daily_limit:,.2f}</span></div>
          </div>
          <div style="width:52px;height:12px;background:rgba(255,255,255,0.06);border-radius:20px;overflow:hidden;flex-shrink:0;">
            <div style="height:100%;width:{dd_dp}%;background:{dd_dc};opacity:0.55;border-radius:20px;"></div>
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
          <div>
            <div style="font-size:9px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">DD Semanal</div>
            <div style="font-size:12px;font-family:var(--fm);color:{dd_wc};">${weekly_dd_abs:,.2f} <span style="color:var(--osv);font-size:10px;">/ ${weekly_limit:,.2f}</span></div>
          </div>
          <div style="width:52px;height:12px;background:rgba(255,255,255,0.06);border-radius:20px;overflow:hidden;flex-shrink:0;">
            <div style="height:100%;width:{dd_wp}%;background:{dd_wc};opacity:0.55;border-radius:20px;"></div>
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
          <div>
            <div style="font-size:9px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px;">DD Desde Pico</div>
            <div style="font-size:12px;font-family:var(--fm);color:{dd_pc};">${peak_dd_abs:,.2f} <span style="color:var(--osv);font-size:10px;">/ ${peak_limit:,.2f}</span></div>
          </div>
          <div style="width:52px;height:12px;background:rgba(255,255,255,0.06);border-radius:20px;overflow:hidden;flex-shrink:0;">
            <div style="height:100%;width:{dd_pp}%;background:{dd_pc};opacity:0.55;border-radius:20px;"></div>
          </div>
        </div>
      </div>
      <div style="padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Circuit Breaker</span>
        <span style="font-size:12px;font-weight:600;font-family:var(--fm);color:{cb_status_color};">{cb_label_clean}</span>
      </div>
    </div>

  </div><!-- /LEFT -->

  <!-- RIGHT COLUMN -->
  <div style="display:flex;flex-direction:column;gap:14px;">

    <!-- ROI Progress -->
    <div class="gc" style="padding:22px 24px;display:flex;flex-direction:column;gap:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1.5px;">Progreso del Ciclo — ROI Tracker</span>
        <div style="padding:3px 10px;background:rgba(90,215,232,0.07);border:1px solid rgba(90,215,232,0.18);border-radius:4px;font-size:10px;font-family:var(--fm);color:{roi_color};">{'En ritmo' if on_track else 'Por debajo del objetivo'}</div>
      </div>
      <div style="width:100%;height:7px;background:rgba(255,255,255,0.06);border-radius:20px;overflow:hidden;position:relative;">
        <div style="position:absolute;top:0;left:0;height:100%;width:{progress_pct:.1f}%;background:linear-gradient(to right,var(--pr),var(--sc));border-radius:20px;box-shadow:0 0 8px rgba(90,215,232,0.25);transition:width 0.8s;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;font-family:var(--fm);color:var(--osv);">
        <span>Día {days_elapsed} de {days_total} · {progress_pct:.1f}%</span>
        <span>Obj hoy: ${target_today:,.2f} &nbsp;·&nbsp; Meta: ${target_equity:,.2f} &nbsp;·&nbsp; {days_left}d restantes</span>
      </div>
    </div>

    <!-- Positions Table -->
    <div class="gc" style="overflow:hidden;flex-grow:1;">
      <div style="padding:16px 20px;border-bottom:1px solid rgba(255,255,255,0.08);display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:11px;font-weight:600;color:var(--os);text-transform:uppercase;letter-spacing:1px;">{n_pos_label}</span>
        <span style="font-size:10px;font-weight:500;color:var(--osv);">Asignación: {allocation_pct:.1f}%</span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:var(--fm);color:var(--os);">
          <thead>
            <tr style="background:rgba(255,255,255,0.04);font-size:10px;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">
              <th style="padding:10px 16px;font-weight:500;text-align:left;">InstrID</th>
              <th style="padding:10px 16px;font-weight:500;text-align:left;">PosID</th>
              <th style="padding:10px 16px;font-weight:500;text-align:right;">Apertura</th>
              <th style="padding:10px 16px;font-weight:500;text-align:right;">Actual</th>
              <th style="padding:10px 16px;font-weight:500;text-align:right;">P&amp;L %</th>
              <th style="padding:10px 16px;font-weight:500;text-align:center;">Stop</th>
              <th style="padding:10px 16px;font-weight:500;text-align:center;">Días</th>
              <th style="padding:10px 16px;font-weight:500;">Régimen</th>
            </tr>
          </thead>
          <tbody>{pos_rows_html}</tbody>
        </table>
      </div>
    </div>

    <!-- Bottom row: Concentration + System -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
      <div class="gc" style="padding:18px 20px;display:flex;flex-direction:column;gap:10px;">
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Concentración</span>
        <div style="display:flex;flex-wrap:wrap;gap:7px;">{conc_html}</div>
      </div>
      <div class="gc" style="padding:18px 20px;display:flex;flex-direction:column;gap:8px;">
        <span style="font-size:10px;font-weight:500;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">Estado del Sistema</span>
        <div style="display:flex;flex-direction:column;gap:7px;font-size:12px;font-family:var(--fm);">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:var(--osv);">eToro API</span>
            <span style="color:{api_color};display:flex;align-items:center;gap:4px;">
              <span style="width:6px;height:6px;border-radius:50%;background:{api_color};display:inline-block;"></span> {api_label}
            </span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:var(--osv);">Polling</span>
            <span style="color:var(--os);">30s · {poll_ts}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:var(--osv);">HMM</span>
            <span style="color:var(--pr);">{regime_label or '—'}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:var(--osv);">Ops hoy</span>
            <span style="color:var(--os);">{daily_trades}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:var(--osv);">Pico equity</span>
            <span style="color:var(--os);">${peak_equity:,.2f}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Signals -->
    <div class="gc" style="overflow:hidden;">
      <div style="padding:14px 20px;border-bottom:1px solid rgba(255,255,255,0.08);">
        <span style="font-size:10px;font-weight:600;color:var(--os);text-transform:uppercase;letter-spacing:1.5px;">Señales Recientes</span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:var(--fm);color:var(--os);">
          <thead>
            <tr style="background:rgba(255,255,255,0.04);font-size:10px;color:var(--osv);text-transform:uppercase;letter-spacing:1px;">
              <th style="padding:9px 14px;font-weight:500;text-align:left;">Instrumento</th>
              <th style="padding:9px 14px;font-weight:500;text-align:left;">Acción</th>
              <th style="padding:9px 14px;font-weight:500;text-align:right;">Tamaño USD</th>
              <th style="padding:9px 14px;font-weight:500;text-align:left;">Estrategia</th>
              <th style="padding:9px 14px;font-weight:500;text-align:center;">Calidad</th>
            </tr>
          </thead>
          <tbody>{signals_html}</tbody>
        </table>
      </div>
    </div>

  </div><!-- /RIGHT -->

</div><!-- /MAIN GRID -->

<!-- ═══ TERMINAL ══════════════════════════════════════════════════════════ -->
<div class="gc" style="margin-bottom:44px;overflow:hidden;">
  <div style="padding:12px 20px;border-bottom:1px solid rgba(255,255,255,0.08);display:flex;align-items:center;gap:8px;">
    <span class="msym" style="font-size:16px;color:var(--osv);">terminal</span>
    <span style="font-size:10px;font-weight:600;color:var(--os);text-transform:uppercase;letter-spacing:1.5px;">Terminal — Log del Sistema</span>
  </div>
  <div style="padding:14px 20px;background:#080E1F;font-family:var(--fm);font-size:11px;max-height:260px;overflow-y:auto;line-height:1.6;">
    {terminal_html}
  </div>
</div>

<!-- ═══ FOOTER ════════════════════════════════════════════════════════════ -->
<footer style="position:fixed;bottom:0;left:0;width:100%;z-index:9999;
  display:flex;justify-content:space-between;align-items:center;
  padding:5px 24px;border-top:1px solid rgba(255,255,255,0.08);
  background:rgba(7,15,43,0.88);backdrop-filter:blur(14px);">
  <span style="font-family:var(--fm);font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#a3b1e8;">
    AgentBotTrade Engine v1.0 · Heartbeat 5s
  </span>
  <div style="display:flex;gap:18px;">
    <span style="font-family:var(--fm);font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#a3b1e8;">live_state.json</span>
    <span style="font-family:var(--fm);font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#a3b1e8;">CB: {cb_status}</span>
  </div>
</footer>
"""

    # st.html() (Streamlit ≥ 1.31) renderiza HTML puro sin procesar markdown.
    # En versiones anteriores usamos st.markdown con unsafe_allow_html=True.
    if hasattr(st, "html"):
        st.html(_dash_html)
    else:
        st.markdown(_dash_html, unsafe_allow_html=True)
