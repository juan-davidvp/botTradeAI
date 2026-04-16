"""
monitoring/ui_manager.py
AgentBotTrade — GUI Dashboard (PySide6)

Arquitectura modular:
  DataBridge   → QObject con Qt signals para actualizaciones thread-safe
  RegimeCard   → Estado del mercado en lenguaje natural
  EquityCard   → Balance actual + barra de progreso ROI
  RiskCard     → Drawdown + Circuit Breaker
  PositionsCard→ Tabla de posiciones abiertas
  SystemCard   → Estado API + polling + HMM
  DashboardApp → QMainWindow que compone todos los cards
"""

from __future__ import annotations

import sys
import queue
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from PySide6.QtCore import (
    Qt, QObject, QTimer, Signal, Slot
)
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QPalette, QPainter,
    QLinearGradient
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QProgressBar, QScrollArea,
    QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget
)

# ── Constantes financieras (sincronizadas con dashboard.py) ──────────────────
INITIAL_EQUITY  = 546.14
TARGET_EQUITY   = 655.37
TARGET_DAYS     = 120
START_DATE      = datetime(2026, 4, 15, tzinfo=timezone.utc)
MONTHLY_RATE    = 0.0466

DD_DAILY_HALT   = 16.38   # 3%
DD_WEEKLY_HALT  = 38.23   # 7%
DD_PEAK_LOCK    = 54.61   # 10%

# ── UX Writing: traducción de regímenes HMM a lenguaje natural ───────────────
REGIME_COPY: Dict[str, Dict[str, str]] = {
    "STRONG_BULL": {
        "title":    "Impulso alcista fuerte",
        "subtitle": "El bot está operando activamente",
        "badge_id": "regimeBadgeBull",
        "icon":     "▲▲",
    },
    "EUPHORIA": {
        "title":    "Mercado en zona de euforia",
        "subtitle": "El bot opera con precaución selectiva",
        "badge_id": "regimeBadgeBull",
        "icon":     "◆",
    },
    "WEAK_BULL": {
        "title":    "Tendencia alcista moderada",
        "subtitle": "El bot busca oportunidades con cautela",
        "badge_id": "regimeBadgeBull",
        "icon":     "▲",
    },
    "NEUTRAL": {
        "title":    "Mercado sin dirección clara",
        "subtitle": "El bot espera una señal más fuerte",
        "badge_id": "regimeBadgeNeutral",
        "icon":     "—",
    },
    "BEAR": {
        "title":    "Mercado bajista",
        "subtitle": "El bot reduce la exposición al riesgo",
        "badge_id": "regimeBadgeBear",
        "icon":     "▼",
    },
    "STRONG_BEAR": {
        "title":    "Caída fuerte del mercado",
        "subtitle": "El bot está protegiendo tu capital",
        "badge_id": "regimeBadgeBear",
        "icon":     "▼▼",
    },
    "CRASH": {
        "title":    "Caída extrema detectada",
        "subtitle": "El bot ha pausado operaciones automáticamente",
        "badge_id": "regimeBadgeCrash",
        "icon":     "⬛",
    },
}

REGIME_COPY_DEFAULT = {
    "title":    "Analizando el mercado…",
    "subtitle": "El bot está recopilando datos históricos",
    "badge_id": "regimeBadge",
    "icon":     "○",
}

# ── UX Writing: Circuit Breaker ──────────────────────────────────────────────
CB_COPY = {
    "OK":        ("cbOk",   "Operación normal"),
    "REDUCE_50": ("cbWarn", "Riesgo reducido al 50% preventivamente"),
    "HALT":      ("cbHalt", "Pausa de seguridad activada"),
    "LOCKED":    ("cbHalt", "Bloqueado — requiere intervención manual"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DataBridge: puente thread-safe entre el bot y la GUI
# ─────────────────────────────────────────────────────────────────────────────

class DataBridge(QObject):
    """
    QObject que expone Qt signals para actualizar la GUI desde el hilo del bot.
    Uso:
        bridge = DataBridge()
        bridge.portfolio_updated.emit(portfolio_state)
        bridge.regime_updated.emit(regime_state)
    """
    portfolio_updated = Signal(object)   # PortfolioState
    regime_updated    = Signal(object)   # RegimeState | None
    signals_updated   = Signal(list)     # List[Dict]
    connection_status = Signal(bool)     # True = API online

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    @Slot(object)
    def push_portfolio(self, state: Any) -> None:
        self.portfolio_updated.emit(state)

    @Slot(object)
    def push_regime(self, state: Any) -> None:
        self.regime_updated.emit(state)

    @Slot(list)
    def push_signals(self, signals: List[Dict]) -> None:
        self.signals_updated.emit(signals)

    @Slot(bool)
    def push_connection(self, online: bool) -> None:
        self.connection_status.emit(online)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de layout
# ─────────────────────────────────────────────────────────────────────────────

def _make_card(object_name: str = "card") -> QFrame:
    frame = QFrame()
    frame.setObjectName(object_name)
    frame.setFrameShape(QFrame.NoFrame)
    return frame


def _label(
    text: str,
    object_name: str = "",
    alignment: Qt.AlignmentFlag = Qt.AlignLeft,
) -> QLabel:
    lbl = QLabel(text)
    if object_name:
        lbl.setObjectName(object_name)
    lbl.setAlignment(alignment)
    lbl.setWordWrap(False)
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setObjectName("separator")
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    return line


def _fmt_usd(value: float, sign: bool = False) -> str:
    prefix = "+" if sign and value >= 0 else ""
    return f"{prefix}${abs(value):,.2f}" if value < 0 else f"{prefix}${value:,.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# GradientHeader: banda superior con degradado azul → morado
# ─────────────────────────────────────────────────────────────────────────────

class GradientHeader(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        title = _label("AgentBotTrade")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color: #E8EEFF; background: transparent;")

        env_badge = _label("● REAL")
        env_badge.setStyleSheet(
            "color: #EF4444; font-size: 11px; font-weight: 700;"
            "background: transparent;"
        )

        self._clock = _label("", alignment=Qt.AlignRight)
        self._clock.setStyleSheet(
            "color: #7B8BB2; font-size: 11px; background: transparent;"
        )

        layout.addWidget(title)
        layout.addWidget(env_badge)
        layout.addStretch()
        layout.addWidget(self._clock)

        timer = QTimer(self)
        timer.timeout.connect(self._tick_clock)
        timer.start(1000)
        self._tick_clock()

    def _tick_clock(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
        self._clock.setText(now)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0.0, QColor("#0D1B3E"))
        gradient.setColorAt(0.6, QColor("#1A1040"))
        gradient.setColorAt(1.0, QColor("#2D0A5C"))
        painter.fillRect(self.rect(), gradient)


# ─────────────────────────────────────────────────────────────────────────────
# RegimeCard: estado del mercado en lenguaje natural
# ─────────────────────────────────────────────────────────────────────────────

class RegimeCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        self._section_title = _label("ESTADO DEL MERCADO", "sectionTitle")
        layout.addWidget(self._section_title)

        row = QHBoxLayout()
        row.setSpacing(12)

        self._icon_label = _label("○", alignment=Qt.AlignVCenter)
        self._icon_label.setStyleSheet(
            "font-size: 22px; color: #448AFF; background: transparent;"
        )
        self._icon_label.setFixedWidth(32)

        text_col = QVBoxLayout()
        text_col.setSpacing(3)
        self._title_label = _label("Analizando el mercado…", "metricMedium")
        self._subtitle_label = _label(
            "El bot está recopilando datos históricos", "metricSmall"
        )
        text_col.addWidget(self._title_label)
        text_col.addWidget(self._subtitle_label)

        badge_col = QVBoxLayout()
        badge_col.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self._badge = _label("SIN DATOS", "regimeBadge", Qt.AlignCenter)
        self._prob_label = _label("", "metricSmall", Qt.AlignCenter)
        badge_col.addWidget(self._badge)
        badge_col.addWidget(self._prob_label)

        row.addWidget(self._icon_label)
        row.addLayout(text_col, stretch=1)
        row.addLayout(badge_col)

        layout.addLayout(row)

        detail_row = QHBoxLayout()
        detail_row.setSpacing(20)
        self._stability_label = _label("Estabilidad: —", "metricSmall")
        self._confirmed_label = _label("Estado: esperando datos", "metricSmall")
        detail_row.addWidget(self._stability_label)
        detail_row.addStretch()
        detail_row.addWidget(self._confirmed_label)
        layout.addLayout(detail_row)

    @Slot(object)
    def update_regime(self, state: Any) -> None:
        if state is None:
            self._apply_copy(REGIME_COPY_DEFAULT, label="", prob=0.0)
            self._stability_label.setText("Estabilidad: —")
            self._confirmed_label.setText("Estado: esperando datos")
            return

        copy = REGIME_COPY.get(state.label, REGIME_COPY_DEFAULT)
        self._apply_copy(copy, label=state.label, prob=state.probability)
        bars = state.consecutive_bars
        self._stability_label.setText(
            f"Estabilidad: {bars} {'día' if bars == 1 else 'días'} consecutivos"
        )
        confirmed_text = (
            "✓ Señal confirmada — el bot puede operar"
            if state.is_confirmed
            else "⏳ Señal en validación…"
        )
        self._confirmed_label.setText(confirmed_text)

    def _apply_copy(self, copy: Dict, label: str, prob: float) -> None:
        self._icon_label.setText(copy["icon"])
        self._title_label.setText(copy["title"])
        self._subtitle_label.setText(copy["subtitle"])
        self._badge.setObjectName(copy["badge_id"])
        badge_text = label if label else "SIN DATOS"
        self._badge.setText(badge_text)
        self._badge.style().unpolish(self._badge)
        self._badge.style().polish(self._badge)
        if prob > 0:
            self._prob_label.setText(f"Confianza: {prob:.0%}")
        else:
            self._prob_label.setText("")


# ─────────────────────────────────────────────────────────────────────────────
# EquityCard: balance + progreso hacia el objetivo
# ─────────────────────────────────────────────────────────────────────────────

class EquityCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(_label("TU BALANCE", "sectionTitle"))

        top_row = QHBoxLayout()
        top_row.setSpacing(32)

        # Balance principal
        equity_col = QVBoxLayout()
        equity_col.setSpacing(2)
        self._equity_label = _label("$546.14", "metricLarge")
        self._equity_sub = _label("Balance actual", "metricSmall")
        equity_col.addWidget(self._equity_label)
        equity_col.addWidget(self._equity_sub)

        # Cambio diario
        daily_col = QVBoxLayout()
        daily_col.setSpacing(2)
        self._daily_label = _label("+$0.00", "valueNeutral")
        self._daily_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._daily_sub = _label("Hoy", "metricSmall")
        daily_col.addWidget(self._daily_label)
        daily_col.addWidget(self._daily_sub)

        # Desde inicio
        since_col = QVBoxLayout()
        since_col.setSpacing(2)
        self._since_label = _label("+$0.00", "valueNeutral")
        self._since_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._since_sub = _label("Desde inicio", "metricSmall")
        since_col.addWidget(self._since_label)
        since_col.addWidget(self._since_sub)

        top_row.addLayout(equity_col)
        top_row.addLayout(daily_col)
        top_row.addLayout(since_col)
        top_row.addStretch()
        layout.addLayout(top_row)

        layout.addWidget(_separator())

        # Barra de progreso ROI
        progress_header = QHBoxLayout()
        self._progress_pct_label = _label("Progreso hacia la meta (+20%)", "metricSmall")
        self._target_label = _label(
            f"Objetivo: ${TARGET_EQUITY:,.2f}", "metricSmall", Qt.AlignRight
        )
        progress_header.addWidget(self._progress_pct_label)
        progress_header.addStretch()
        progress_header.addWidget(self._target_label)
        layout.addLayout(progress_header)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("roiProgress")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(10)
        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._progress_bar)

        # ROI tracker diario
        self._roi_label = _label(
            "Días transcurridos: 1 / 120  |  Objetivo hoy: $546.97",
            "metricSmall",
        )
        layout.addWidget(self._roi_label)

    @Slot(object)
    def update_portfolio(self, state: Any) -> None:
        if state is None:
            return

        eq = state.equity
        daily = state.daily_pnl
        since = eq - INITIAL_EQUITY
        progress_pct = max(
            0.0,
            (eq - INITIAL_EQUITY) / (TARGET_EQUITY - INITIAL_EQUITY) * 100,
        )

        self._equity_label.setText(f"${eq:,.2f}")

        self._daily_label.setText(
            f"{'+' if daily >= 0 else ''}{_fmt_usd(daily)}"
        )
        self._daily_label.setStyleSheet(
            f"font-size: 16px; font-weight: 600;"
            f"color: {'#10B981' if daily >= 0 else '#EF4444'};"
        )

        self._since_label.setText(
            f"{'+' if since >= 0 else ''}{_fmt_usd(since)}"
        )
        self._since_label.setStyleSheet(
            f"font-size: 16px; font-weight: 600;"
            f"color: {'#10B981' if since >= 0 else '#EF4444'};"
        )

        self._progress_bar.setValue(int(min(progress_pct, 100)))
        self._progress_pct_label.setText(
            f"Progreso hacia la meta (+20%): {progress_pct:.1f}%"
        )

        now = datetime.now(timezone.utc)
        elapsed = max(0, (now - START_DATE).days)
        n_months = elapsed / 30
        target_today = INITIAL_EQUITY * ((1 + MONTHLY_RATE) ** n_months)
        on_track = eq >= target_today
        track_icon = "✓" if on_track else "↓"
        track_color = "#10B981" if on_track else "#F59E0B"
        self._roi_label.setText(
            f"Día {elapsed} de {TARGET_DAYS}  |  "
            f"Meta de hoy: ${target_today:,.2f}  |  "
            f"<span style='color:{track_color}'>{track_icon} "
            f"{'En ritmo' if on_track else 'Por debajo del ritmo'}</span>"
        )
        self._roi_label.setTextFormat(Qt.RichText)


# ─────────────────────────────────────────────────────────────────────────────
# RiskCard: drawdown + circuit breaker en lenguaje accesible
# ─────────────────────────────────────────────────────────────────────────────

class RiskCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.addWidget(_label("PROTECCIÓN DE CAPITAL", "sectionTitle"))
        header_row.addStretch()
        self._cb_label = _label("Operación normal", "cbOk")
        header_row.addWidget(self._cb_label)
        layout.addLayout(header_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        def _dd_pair(title: str, limit_usd: float, limit_label: str):
            title_lbl = _label(title, "metricSmall")
            val_lbl = _label("$0.00", "valuePositive")
            val_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #10B981;")
            cap_lbl = _label(f"Límite: ${limit_usd:.2f} ({limit_label})", "metricSmall")
            return title_lbl, val_lbl, cap_lbl

        self._dd_daily_title, self._dd_daily_val, self._dd_daily_cap = _dd_pair(
            "Caída de hoy", DD_DAILY_HALT, "3%"
        )
        self._dd_weekly_title, self._dd_weekly_val, self._dd_weekly_cap = _dd_pair(
            "Caída semanal", DD_WEEKLY_HALT, "7%"
        )
        self._dd_peak_title, self._dd_peak_val, self._dd_peak_cap = _dd_pair(
            "Caída máxima histórica", DD_PEAK_LOCK, "10%"
        )

        for col, (title, val, cap) in enumerate([
            (self._dd_daily_title,  self._dd_daily_val,  self._dd_daily_cap),
            (self._dd_weekly_title, self._dd_weekly_val, self._dd_weekly_cap),
            (self._dd_peak_title,   self._dd_peak_val,   self._dd_peak_cap),
        ]):
            grid.addWidget(title, 0, col)
            grid.addWidget(val,   1, col)
            grid.addWidget(cap,   2, col)

        layout.addLayout(grid)

    def _apply_dd(
        self,
        label: QLabel,
        value_usd: float,
        limit_usd: float,
    ) -> None:
        ratio = value_usd / limit_usd if limit_usd > 0 else 0
        if ratio < 0.5:
            color = "#10B981"
            prefix = "Protección activa: "
        elif ratio < 0.85:
            color = "#F59E0B"
            prefix = "Atención: "
        else:
            color = "#EF4444"
            prefix = "Alerta: "
        label.setText(f"{prefix}-${value_usd:.2f}")
        label.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {color};"
        )

    @Slot(object)
    def update_portfolio(self, state: Any) -> None:
        if state is None:
            return

        eq = state.equity
        dd_daily  = abs(state.daily_pnl)  if state.daily_pnl  < 0 else 0.0
        dd_weekly = abs(state.weekly_pnl) if state.weekly_pnl < 0 else 0.0
        dd_peak   = abs(state.drawdown_pct * eq) if state.drawdown_pct < 0 else 0.0

        self._apply_dd(self._dd_daily_val,  dd_daily,  DD_DAILY_HALT)
        self._apply_dd(self._dd_weekly_val, dd_weekly, DD_WEEKLY_HALT)
        self._apply_dd(self._dd_peak_val,   dd_peak,   DD_PEAK_LOCK)

        cb = state.circuit_breaker_status
        cb_id, cb_text = CB_COPY.get(cb, ("cbOk", "Operación normal"))
        self._cb_label.setObjectName(cb_id)
        self._cb_label.setText(cb_text)
        self._cb_label.style().unpolish(self._cb_label)
        self._cb_label.style().polish(self._cb_label)


# ─────────────────────────────────────────────────────────────────────────────
# PositionsCard: tabla de posiciones abiertas
# ─────────────────────────────────────────────────────────────────────────────

POSITION_HEADERS = [
    "Instrumento", "Precio entrada", "Precio actual",
    "Rendimiento", "Stop loss", "Días abierto",
]
POSITION_TOOLTIPS = [
    "ID del instrumento en eToro",
    "Precio al que se abrió la posición",
    "Precio de mercado en tiempo real",
    "Ganancia o pérdida en porcentaje",
    "Precio de cierre automático para limitar pérdidas",
    "Días que lleva abierta la posición",
]

# Instrumentos conocidos (para mostrar nombre amigable)
INSTRUMENT_NAMES: Dict[int, str] = {
    4238:  "VOO (S&P 500)",
    14328: "MELI (MercadoLibre)",
    9408:  "PLTR (Palantir)",
    6218:  "APP (AppLovin)",
    2488:  "DAVE (Dave Inc.)",
    100000: "Posición externa",
}


class PositionsCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(_label("POSICIONES ABIERTAS", "sectionTitle"))
        self._count_label = _label("", "metricSmall", Qt.AlignRight)
        header.addStretch()
        header.addWidget(self._count_label)
        layout.addLayout(header)

        self._table = QTableWidget(0, len(POSITION_HEADERS))
        self._table.setHorizontalHeaderLabels(POSITION_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(160)

        for i, tip in enumerate(POSITION_TOOLTIPS):
            self._table.horizontalHeaderItem(i).setToolTip(tip) if self._table.horizontalHeaderItem(i) else None

        layout.addWidget(self._table)

    @Slot(object)
    def update_portfolio(self, state: Any) -> None:
        if state is None or not state.positions:
            self._table.setRowCount(0)
            self._count_label.setText("Sin posiciones abiertas")
            return

        positions = state.positions
        self._count_label.setText(f"{len(positions)} posición{'es' if len(positions) != 1 else ''}")
        self._table.setRowCount(len(positions))

        for row, pos in enumerate(positions):
            instr_id  = int(pos.get("instrumentID", 0))
            open_rate = float(pos.get("openRate", 0))
            # current_price viene de closeRate (endpoint /real/pnl) via _enrich_positions
            current   = float(pos.get("current_price", 0))
            sl_rate   = float(pos.get("stopLossRate", 0))
            no_stop   = bool(pos.get("isNoStopLoss", False))
            holding   = int(pos.get("holding_days", 0))

            # unrealized_pnl viene del campo pnL del endpoint (ya incluye fees)
            pnl_usd   = float(pos.get("unrealized_pnl", 0))
            init_amt  = float(pos.get("initialAmountInDollars", open_rate))
            pnl_pct   = (pnl_usd / init_amt * 100) if init_amt > 0 else 0.0

            name = INSTRUMENT_NAMES.get(instr_id, f"#{instr_id}")

            if no_stop or sl_rate <= 0.001:
                stop_text  = "⚠ Sin protección"
                stop_color = QColor("#EF4444")
            else:
                stop_text  = f"${sl_rate:,.2f}"
                stop_color = QColor("#10B981") if sl_rate >= open_rate * 0.90 else QColor("#F59E0B")

            pnl_color    = QColor("#10B981") if pnl_pct >= 0 else QColor("#EF4444")
            pnl_text     = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%  (${pnl_usd:+.2f})"
            current_text = f"${current:,.2f}" if current > 0 else "—"

            cells = [
                (name,         QColor("#E8EEFF")),
                (f"${open_rate:,.2f}", QColor("#7B8BB2")),
                (current_text, QColor("#E8EEFF")),
                (pnl_text,     pnl_color),
                (stop_text,    stop_color),
                (f"{holding}d", QColor("#7B8BB2")),
            ]

            for col, (text, color) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(color)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._table.setItem(row, col, item)

        self._table.resizeColumnsToContents()


# ─────────────────────────────────────────────────────────────────────────────
# SystemCard: estado de conexión, polling y modelo
# ─────────────────────────────────────────────────────────────────────────────

class SystemCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(32)

        layout.addWidget(_label("SISTEMA", "sectionTitle"))

        self._api_dot   = _label("●", "statusDot")
        self._api_dot.setStyleSheet("color: #10B981; font-size: 12px;")
        self._api_label = _label("API eToro conectada", "statusText")

        self._poll_label  = _label("Actualización: —", "statusText")
        self._model_label = _label("Modelo: —", "statusText")

        for widget in [self._api_dot, self._api_label,
                       self._poll_label, self._model_label]:
            layout.addWidget(widget)

        layout.addStretch()

    @Slot(bool)
    def update_connection(self, online: bool) -> None:
        if online:
            self._api_dot.setStyleSheet("color: #10B981; font-size: 12px;")
            self._api_label.setText("API eToro conectada")
        else:
            self._api_dot.setStyleSheet("color: #EF4444; font-size: 12px;")
            self._api_label.setText("API eToro sin conexión")

    def update_poll_ts(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._poll_label.setText(f"Última lectura: {ts} UTC")

    def update_hmm_age(self, trained_ts: Optional[datetime]) -> None:
        if trained_ts is None:
            self._model_label.setText("Modelo: calculando…")
            return
        days = (datetime.now(timezone.utc) - trained_ts).days
        self._model_label.setText(
            f"Modelo: actualizado hace {days} día{'s' if days != 1 else ''}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DashboardApp: ventana principal
# ─────────────────────────────────────────────────────────────────────────────

class DashboardApp(QMainWindow):
    """
    Ventana principal del bot. Recibe datos a través de DataBridge
    (hilo seguro) o a través de una Queue para uso standalone.

    Uso mínimo:
        app    = QApplication(sys.argv)
        bridge = DataBridge()
        win    = DashboardApp(bridge)
        win.show()
        sys.exit(app.exec())
    """

    def __init__(
        self,
        bridge: Optional[DataBridge] = None,
        data_queue: Optional[queue.Queue] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge or DataBridge(self)
        self._data_queue = data_queue

        self._setup_window()
        self._build_ui()
        self._connect_signals()

        if self._data_queue is not None:
            self._queue_timer = QTimer(self)
            self._queue_timer.timeout.connect(self._drain_queue)
            self._queue_timer.start(500)

    # ── Window setup ─────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("AgentBotTrade — Dashboard")
        self.resize(1100, 780)
        self.setMinimumSize(800, 600)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        self._header = GradientHeader()
        root.addWidget(self._header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("centralWidget")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        # ── Fila 1: Régimen + Equity (2 columnas) ─────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        self.regime_card = RegimeCard()
        self.regime_card.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )

        self.equity_card = EquityCard()
        self.equity_card.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )

        row1.addWidget(self.regime_card, stretch=1)
        row1.addWidget(self.equity_card, stretch=2)
        content_layout.addLayout(row1)

        # ── Fila 2: Risk (ancho completo) ─────────────────────────────────
        self.risk_card = RiskCard()
        content_layout.addWidget(self.risk_card)

        # ── Fila 3: Posiciones (ancho completo) ───────────────────────────
        self.positions_card = PositionsCard()
        content_layout.addWidget(self.positions_card)

        content_layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

        # Footer con estado del sistema
        self.system_card = SystemCard()
        self.system_card.setFixedHeight(44)
        root.addWidget(self.system_card)

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        b = self._bridge
        b.portfolio_updated.connect(self.equity_card.update_portfolio)
        b.portfolio_updated.connect(self.risk_card.update_portfolio)
        b.portfolio_updated.connect(self.positions_card.update_portfolio)
        b.portfolio_updated.connect(self._on_portfolio_update)
        b.regime_updated.connect(self.regime_card.update_regime)
        b.connection_status.connect(self.system_card.update_connection)

    @Slot(object)
    def _on_portfolio_update(self, state: Any) -> None:
        self.system_card.update_poll_ts()

    # ── Queue drain (usado cuando el bot escribe en una Queue) ────────────────

    def _drain_queue(self) -> None:
        if self._data_queue is None:
            return
        try:
            while True:
                msg = self._data_queue.get_nowait()
                msg_type = msg.get("type")
                payload  = msg.get("payload")
                if msg_type == "portfolio":
                    self._bridge.push_portfolio(payload)
                elif msg_type == "regime":
                    self._bridge.push_regime(payload)
                elif msg_type == "signals":
                    self._bridge.push_signals(payload)
                elif msg_type == "connection":
                    self._bridge.push_connection(payload)
                elif msg_type == "hmm_age":
                    self.system_card.update_hmm_age(payload)
        except queue.Empty:
            pass

    # ── Stylesheet loader ─────────────────────────────────────────────────────

    @staticmethod
    def load_stylesheet(path: str = "monitoring/styles.qss") -> str:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint standalone (para testear sin el bot)
# ─────────────────────────────────────────────────────────────────────────────

def _demo_mode() -> None:
    """Lanza la GUI con datos simulados para ver el look & feel."""
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class FakePortfolio:
        equity: float = 589.42
        cash: float = 38.61
        buying_power: float = 38.61
        positions: List = field(default_factory=lambda: [
            {
                "instrumentID": 4238, "positionID": "A001",
                "openRate": 637.10, "current_price": 648.30, "closeRate": 648.30,
                "initialAmountInDollars": 270.00, "unrealized_pnl": 4.73, "pnL": 4.73,
                "unitsBaseValueDollars": 274.73,
                "stopLossRate": 0.0, "isNoStopLoss": True, "holding_days": 3,
            },
            {
                "instrumentID": 9408, "positionID": "A002",
                "openRate": 136.68, "current_price": 141.20, "closeRate": 141.20,
                "initialAmountInDollars": 60.00, "unrealized_pnl": 1.98, "pnL": 1.98,
                "unitsBaseValueDollars": 61.98,
                "stopLossRate": 122.00, "isNoStopLoss": False, "holding_days": 5,
            },
        ])
        daily_pnl: float = 4.32
        weekly_pnl: float = -2.10
        peak_equity: float = 591.00
        drawdown_pct: float = -0.003
        circuit_breaker_status: str = "OK"
        flicker_rate: float = 0.1
        daily_trades_count: int = 2
        last_order_by_instrument: Dict = field(default_factory=dict)

    @dataclass
    class FakeRegime:
        label: str = "STRONG_BULL"
        probability: float = 0.82
        consecutive_bars: int = 4
        is_confirmed: bool = True
        flicker_count: int = 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    bridge = DataBridge()
    win = DashboardApp(bridge)
    qss = DashboardApp.load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    win.show()

    # Empujar datos demo con un pequeño delay
    QTimer.singleShot(300, lambda: bridge.push_portfolio(FakePortfolio()))
    QTimer.singleShot(400, lambda: bridge.push_regime(FakeRegime()))
    QTimer.singleShot(500, lambda: bridge.push_connection(True))

    sys.exit(app.exec())


if __name__ == "__main__":
    _demo_mode()
