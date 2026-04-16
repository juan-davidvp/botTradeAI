"""
monitoring/dashboard.py
Fase 8 — Dashboard Rich en Terminal (refresco cada 5 segundos con Rich Live).

Paneles:
  REGIME       : estado, estabilidad (barras), flicker count
  PORTFOLIO    : equity vs objetivo, progreso ROI, diario, cash, asignación
  POSITIONS    : tabla con P&L%, stop, Stop%ok (✅/⚠️), días
  CONCENTRACIÓN: pesos actuales vs límite 20%
  RECENT SIGNALS: últimas señales (hora, instrID, acción, motivo)
  RISK STATUS  : DD diario / semanal / desde pico (USD) con semáforos
  ROI TRACKER  : curva objetivo compuesta 4.66%/mes
  SYSTEM       : latencia API, polling, HMM age, entorno
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live   import Live
from rich.panel  import Panel
from rich.table  import Table
from rich.text   import Text

from core.risk_manager import PortfolioState

logger  = logging.getLogger(__name__)
console = Console()

INITIAL_EQUITY = 546.14
TARGET_EQUITY  = 655.37
TARGET_DAYS    = 120
START_DATE     = datetime(2026, 4, 15, tzinfo=timezone.utc)
MONTHLY_RATE   = 0.0466

# Umbrales de riesgo en USD (sobre $546.14)
DD_DAILY_REDUCE  =  10.92   # 2%
DD_DAILY_HALT    =  16.38   # 3%
DD_WEEKLY_REDUCE =  27.31   # 5%
DD_WEEKLY_HALT   =  38.23   # 7%
DD_PEAK_LOCK     =  54.61   # 10%


class Dashboard:
    """
    Dashboard Rich que refresca cada 5 segundos usando Rich Live.

    Parámetros
    ----------
    settings : dict  Sección completa de settings.yaml.
    """

    def __init__(self, settings: Optional[Dict] = None):
        self.settings        = settings or {}
        self._last_signals   : List[Dict] = []
        self._api_latency    : float      = 0.0
        self._poll_ts        : Optional[datetime] = None
        self._hmm_trained_ts : Optional[datetime] = None

    # ------------------------------------------------------------------
    # Actualización de estado (llamado por el loop principal)
    # ------------------------------------------------------------------

    def update(
        self,
        portfolio_state: Optional[PortfolioState],
        hmm_state: Optional[Any],
        signals: List[Dict],
        api_latency_ms: float = 0.0,
        hmm_trained_ts: Optional[datetime] = None,
    ) -> None:
        """Actualiza el estado interno sin renderizar."""
        if signals:
            self._last_signals = signals[-5:]
        self._api_latency    = api_latency_ms
        self._poll_ts        = datetime.now(timezone.utc)
        if hmm_trained_ts:
            self._hmm_trained_ts = hmm_trained_ts

        self._portfolio_state = portfolio_state
        self._hmm_state       = hmm_state

    # ------------------------------------------------------------------
    # Renderizado único (sin Live — para el ciclo principal de 30s)
    # ------------------------------------------------------------------

    def render(
        self,
        portfolio_state: Optional[PortfolioState],
        hmm_state: Optional[Any],
        signals: List[Dict],
        api_latency_ms: float = 0.0,
        hmm_trained_ts: Optional[datetime] = None,
    ) -> None:
        """Imprime el dashboard completo una sola vez en consola."""
        self.update(portfolio_state, hmm_state, signals, api_latency_ms, hmm_trained_ts)
        console.clear()
        self._print_all(portfolio_state, hmm_state)

    # ------------------------------------------------------------------
    # Modo Live — refresco cada 5 segundos (bloqueante)
    # ------------------------------------------------------------------

    def run_live(
        self,
        portfolio_state: Optional[PortfolioState],
        hmm_state: Optional[Any],
        signals: List[Dict],
        api_latency_ms: float = 0.0,
        hmm_trained_ts: Optional[datetime] = None,
        refresh_seconds: int = 5,
    ) -> None:
        """
        Inicia Rich Live con refresco cada `refresh_seconds` segundos.
        Bloqueante — usar en hilo separado.
        """
        self.update(portfolio_state, hmm_state, signals, api_latency_ms, hmm_trained_ts)
        with Live(
            self._build_renderable(),
            console=console,
            refresh_per_second=1 / refresh_seconds,
            screen=True,
        ) as live:
            while True:
                import time as _time
                _time.sleep(refresh_seconds)
                live.update(self._build_renderable())

    # ------------------------------------------------------------------
    # Composición del renderable completo
    # ------------------------------------------------------------------

    def _build_renderable(self):
        ps  = getattr(self, "_portfolio_state", None)
        hmm = getattr(self, "_hmm_state",       None)

        layout = Layout()
        panels = [
            self._header(),
            self._regime_panel(hmm),
            self._portfolio_panel(ps),
            self._positions_table(ps),
            self._concentration_panel(ps),
            self._signals_panel(),
            self._risk_panel(ps),
            self._roi_panel(ps),
            self._system_panel(),
        ]
        # Apilamos verticalmente usando una tabla de una columna
        tbl = Table.grid()
        for p in panels:
            tbl.add_row(p)
        return tbl

    def _print_all(self, ps, hmm) -> None:
        console.print(self._header())
        console.print(self._regime_panel(hmm))
        console.print(self._portfolio_panel(ps))
        console.print(self._positions_table(ps))
        console.print(self._concentration_panel(ps))
        console.print(self._signals_panel())
        console.print(self._risk_panel(ps))
        console.print(self._roi_panel(ps))
        console.print(self._system_panel())

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> Text:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        env = self.settings.get("broker", {}).get("environment", "?").upper()
        color = "red" if env == "REAL" else "green"
        t = Text()
        t.append("▶  regime-trader  ", style="bold cyan")
        t.append(f"[{env}]", style=f"bold {color}")
        t.append(f"   {now}", style="dim")
        return t

    # ------------------------------------------------------------------
    # REGIME: estado | estabilidad | flicker
    # ------------------------------------------------------------------

    def _regime_panel(self, state: Optional[Any]) -> Panel:
        if state is None:
            return Panel("[dim]Sin datos de régimen[/dim]", title="REGIME")

        label    = state.label
        prob     = state.probability
        bars     = state.consecutive_bars
        confirmed = "✅" if state.is_confirmed else "⏳"
        flicker_count = getattr(state, "flicker_count", 0)
        flicker_window = self.settings.get("hmm", {}).get("flicker_window", 20)

        color = {
            "BULL": "green", "STRONG_BULL": "bright_green",
            "NEUTRAL": "yellow", "BEAR": "red",
            "STRONG_BEAR": "bright_red", "CRASH": "bold red",
            "EUPHORIA": "bright_cyan",
        }.get(label, "white")

        content = (
            f"Estado: [{color}]{label} {prob:.0%}[/{color}]  "
            f"{confirmed}  |  "
            f"Estabilidad: {bars} barras  |  "
            f"Flicker: {flicker_count}/{flicker_window}  |  "
            f"Confirmado: {'Sí' if state.is_confirmed else 'No'}"
        )
        return Panel(content, title="[bold]REGIME[/bold]", border_style=color)

    # ------------------------------------------------------------------
    # PORTFOLIO
    # ------------------------------------------------------------------

    def _portfolio_panel(self, state: Optional[PortfolioState]) -> Panel:
        if state is None:
            return Panel("[dim]Sin datos de portafolio[/dim]", title="PORTFOLIO")

        eq        = state.equity
        cash      = state.cash
        invested  = eq - cash
        alloc     = invested / eq * 100 if eq > 0 else 0
        progress  = (eq - INITIAL_EQUITY) / (TARGET_EQUITY - INITIAL_EQUITY) * 100
        since_bot = eq - INITIAL_EQUITY
        daily     = state.daily_pnl

        prog_color    = "green" if progress >= 0 else "red"
        since_sign    = "+" if since_bot >= 0 else ""
        daily_sign    = "+" if daily >= 0 else ""
        daily_color   = "green" if daily >= 0 else "red"
        since_color   = "green" if since_bot >= 0 else "red"

        t = Table.grid(padding=1)
        t.add_row(
            f"Equity: [bold]${eq:.2f}[/bold]",
            f"Objetivo: ${TARGET_EQUITY:.2f} (+20%)",
            f"Progreso: [{prog_color}]{progress:.1f}% / 20.0%[/{prog_color}]",
        )
        t.add_row(
            f"Diario: [{daily_color}]{daily_sign}${daily:.2f} ({daily_sign}{daily / eq * 100:.2f}%)[/{daily_color}]",
            f"Desde inicio bot: [{since_color}]{since_sign}${since_bot:.2f}[/{since_color}]",
            f"Cash libre: ${cash:.2f}  |  Asignación: {alloc:.1f}%  |  Leverage: 1.0x",
        )
        return Panel(t, title="[bold]PORTFOLIO[/bold]", border_style="cyan")

    # ------------------------------------------------------------------
    # POSITIONS
    # ------------------------------------------------------------------

    def _positions_table(self, state: Optional[PortfolioState]) -> Panel:
        tbl = Table(show_lines=True, title=None)
        for col in ["InstrID", "PositionID", "Apertura", "Actual", "P&L%", "Stop", "Stop%ok", "Días"]:
            tbl.add_column(col, justify="right")

        if state is None or not state.positions:
            tbl.add_row(*["—"] * 8)
            return Panel(tbl, title="[bold]POSITIONS[/bold]")

        for pos in state.positions:
            instr_id   = str(pos.get("instrumentID", ""))
            pos_id     = str(pos.get("positionID", ""))
            open_rate  = float(pos.get("openRate", 0))
            current    = float(pos.get("current_price", open_rate))
            sl_rate    = float(pos.get("stopLossRate", 0))
            no_stop    = bool(pos.get("isNoStopLoss", False))
            holding    = int(pos.get("holding_days", 0))
            pnl_pct    = (current - open_rate) / open_rate * 100 if open_rate > 0 else 0

            pnl_color  = "green" if pnl_pct >= 0 else "red"

            # Stop%ok: ✅ si stopLossRate >= open_rate * 0.90
            #          ⚠️ si isNoStopLoss=True o stopLossRate <= 0.001
            if no_stop or sl_rate <= 0.001:
                stop_badge = "[red]SIN STOP ⚠️[/red]"
            elif sl_rate >= open_rate * 0.90:
                stop_badge = "✅"
            else:
                stop_badge = "[yellow]⚠️ bajo[/yellow]"

            current_str = f"${current:.2f}" if current != open_rate else "—"

            tbl.add_row(
                instr_id,
                pos_id,
                f"${open_rate:.2f}",
                current_str,
                f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]",
                f"${sl_rate:.2f}",
                stop_badge,
                f"{holding}d",
            )

        return Panel(tbl, title="[bold]POSITIONS[/bold]", border_style="blue")

    # ------------------------------------------------------------------
    # CONCENTRACIÓN
    # ------------------------------------------------------------------

    def _concentration_panel(self, state: Optional[PortfolioState]) -> Panel:
        if state is None or not state.positions:
            return Panel("[dim]Sin posiciones[/dim]", title="CONCENTRACIÓN")

        eq    = state.equity
        parts = []
        for pos in sorted(
            state.positions,
            key=lambda p: float(p.get("amount", 0)),
            reverse=True,
        ):
            instr  = pos.get("instrumentID", "?")
            amount = float(pos.get("amount", 0))
            weight = amount / eq * 100 if eq > 0 else 0
            badge  = " [red][⚠️ MAX 20%][/red]" if weight > 20 else ""
            parts.append(f"{instr}: [bold]{weight:.1f}%[/bold]{badge}")

        return Panel("  |  ".join(parts), title="[bold]CONCENTRACIÓN[/bold]", border_style="yellow")

    # ------------------------------------------------------------------
    # RECENT SIGNALS
    # ------------------------------------------------------------------

    def _signals_panel(self) -> Panel:
        if not self._last_signals:
            return Panel("[dim]Sin señales recientes[/dim]", title="RECENT SIGNALS")

        tbl = Table.grid(padding=1)
        for sig in self._last_signals:
            tbl.add_row(
                sig.get("time", "—"),
                str(sig.get("symbol", "—")),
                sig.get("action", "—"),
                sig.get("reason", "—"),
            )
        return Panel(tbl, title="[bold]RECENT SIGNALS[/bold]", border_style="magenta")

    # ------------------------------------------------------------------
    # RISK STATUS
    # ------------------------------------------------------------------

    def _risk_panel(self, state: Optional[PortfolioState]) -> Panel:
        if state is None:
            return Panel("[dim]Sin datos de riesgo[/dim]", title="RISK STATUS")

        eq        = state.equity
        dd_daily  = abs(state.daily_pnl)  if state.daily_pnl  < 0 else 0.0
        dd_weekly = abs(state.weekly_pnl) if state.weekly_pnl < 0 else 0.0
        dd_peak   = abs(state.drawdown_pct * eq) if state.drawdown_pct < 0 else 0.0

        def badge(val: float, limit: float, label: str) -> str:
            if val < limit * 0.67:
                color = "green"
                icon  = "✅"
            elif val < limit:
                color = "yellow"
                icon  = "⚠️"
            else:
                color = "red"
                icon  = "🔴"
            return f"[{color}]${val:.2f} / ${limit:.2f}[/{color}] {icon}"

        t = Table.grid(padding=2)
        t.add_row(
            f"DD Diario:     {badge(dd_daily,  DD_DAILY_HALT,   '3%')} (3%)",
            f"DD Semanal:    {badge(dd_weekly, DD_WEEKLY_HALT,  '7%')} (7%)",
            f"DD Desde Pico: {badge(dd_peak,   DD_PEAK_LOCK,   '10%')} (10%)",
        )
        cb       = state.circuit_breaker_status
        cb_color = {
            "OK": "green", "REDUCE_50": "yellow",
            "HALT": "red", "LOCKED": "bold red",
        }.get(cb, "white")
        t.add_row(f"Circuit Breaker: [{cb_color}]{cb}[/{cb_color}]", "", "")
        return Panel(t, title="[bold]RISK STATUS[/bold]", border_style="red")

    # ------------------------------------------------------------------
    # ROI TRACKER
    # ------------------------------------------------------------------

    def _roi_panel(self, state: Optional[PortfolioState]) -> Panel:
        now          = datetime.now(timezone.utc)
        elapsed      = max(0, (now - START_DATE).days)
        n_months     = elapsed / 30
        target_today = INITIAL_EQUITY * ((1 + MONTHLY_RATE) ** n_months)

        eq       = state.equity if state else INITIAL_EQUITY
        on_track = eq >= target_today
        badge    = "✅ En ritmo" if on_track else "⚠️ Por debajo"
        color    = "green" if on_track else "yellow"

        content = (
            f"Días transcurridos: {elapsed}/{TARGET_DAYS}  |  "
            f"Equity actual: [bold]${eq:.2f}[/bold]  |  "
            f"Objetivo hoy: ${target_today:.2f}  |  "
            f"[{color}]{badge}[/{color}]"
        )
        return Panel(content, title="[bold]ROI TRACKER[/bold]", border_style="green")

    # ------------------------------------------------------------------
    # SYSTEM
    # ------------------------------------------------------------------

    def _system_panel(self) -> Panel:
        env     = self.settings.get("broker", {}).get("environment", "?").upper()
        env_badge = f"[red]{env} ⚠️[/red]" if env == "REAL" else f"[green]{env}[/green]"
        poll_ts = self._poll_ts.strftime("%H:%M:%S") if self._poll_ts else "—"
        latency = f"{self._api_latency:.0f}ms" if self._api_latency > 0 else "—"

        # Antigüedad del último entrenamiento HMM
        if self._hmm_trained_ts:
            hmm_age_days = (datetime.now(timezone.utc) - self._hmm_trained_ts).days
            hmm_str = f"{hmm_age_days}d ago"
        else:
            hmm_str = "—"

        content = (
            f"eToro API ✅ (latencia: {latency})  |  "
            f"Polling: 30s (último: {poll_ts})  |  "
            f"HMM: {hmm_str}  |  "
            f"Entorno: {env_badge}"
        )
        return Panel(content, title="[bold]SYSTEM[/bold]", border_style="dim")
