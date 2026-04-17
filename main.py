"""
main.py
Fase 7 — Orquestador Central del Sistema regime-trader (eToro Edition).

Startup   : Carga config → verifica cuenta → confirma entorno → stops urgentes
            → HMM → RiskManager → PositionTracker → polling loop.
Main loop : Polling 30s → precios → portafolio → features (1×día) →
            HMM forward → estrategia → riesgo → órdenes → dashboard.
Shutdown  : SIGINT/SIGTERM → snapshot → resumen de sesión.

CLI flags:
  --dry-run       Pipeline completo sin enviar órdenes reales.
  --backtest      Backtester walk-forward.
  --train-only    Entrena HMM y sale.
  --stress-test   Pruebas de estrés.
  --compare       Backtest + benchmarks.
  --dashboard     Muestra dashboard de instancia en ejecución.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, date, timezone
from typing import Dict, List, Optional

import yaml

# ── Módulos del proyecto ────────────────────────────────────────────────────
from broker.etoro_client  import EToroClient, EToroAPIError
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.hmm_engine       import HMMEngine
from core.risk_manager     import RiskManager, PortfolioState
from core.signal_generator import SignalGenerator
from data.feature_engineering import build_features
from data.market_data      import MarketData
from monitoring.logger     import setup_logging
from monitoring.dashboard  import Dashboard
from monitoring.alerts     import AlertManager

# GUI opcional — no falla si PySide6 no está instalado
try:
    from PySide6.QtWidgets import QApplication
    from monitoring.ui_manager import DataBridge, DashboardApp
    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────────────────────
EXPECTED_CID     = 34044505
POLL_SECONDS     = 30
MARKET_OPEN_H    = 9
MARKET_OPEN_M    = 30
MARKET_CLOSE_H   = 16
MARKET_CLOSE_M   = 0
RETRAIN_DAYS     = 7
SESSION_LOG_FILE = "logs/session_summary.json"


# ============================================================================
# Utilidades
# ============================================================================

def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_market_open(instrument_ids: List[int]) -> bool:
    """
    eToro no expone endpoint de horario de mercado.
    Regla: acciones USA → lunes-viernes 09:30-16:00 ET.
    Crypto → 24/7 (instrumentos con ID conocido de cripto).
    Simplificación conservadora: si algún instrumento es crypto, siempre True.
    """
    CRYPTO_IDS = {1,2,3,100,101}   # ampliar según mapa de instrumentos eToro
    if any(i in CRYPTO_IDS for i in instrument_ids):
        return True
    now    = datetime.now()
    wd     = now.weekday()          # 0=lunes, 4=viernes
    hhmm   = now.hour * 60 + now.minute
    open_  = MARKET_OPEN_H  * 60 + MARKET_OPEN_M
    close_ = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
    return 0 <= wd <= 4 and open_ <= hhmm < close_


def is_market_close_bar() -> bool:
    """True si estamos dentro de los 5 minutos posteriores al cierre (16:00 ET)."""
    now   = datetime.now()
    hhmm  = now.hour * 60 + now.minute
    close = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
    return close <= hhmm < close + 5


def is_monday_open() -> bool:
    """True si es lunes en horario de apertura → re-entrenamiento semanal."""
    now  = datetime.now()
    hhmm = now.hour * 60 + now.minute
    return now.weekday() == 0 and hhmm < MARKET_OPEN_H * 60 + MARKET_OPEN_M + 10


# ============================================================================
# STARTUP
# ============================================================================

def startup(settings: dict, dry_run: bool = False) -> dict:
    """
    Secuencia de arranque completa (10 pasos).
    Retorna dict con todas las instancias inicializadas o sys.exit en fallo.
    """
    logger.info("=" * 60)
    logger.info("regime-trader STARTUP | %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # ── Paso 1: Validar variables de entorno ────────────────────────────────
    for var in ("ETORO_API_KEY", "ETORO_USER_KEY"):
        if not os.environ.get(var):
            logger.critical("Variable de entorno faltante: %s — ABORTANDO", var)
            sys.exit(1)

    # ── Paso 2: Crear cliente eToro y verificar cuenta ──────────────────────
    environment = settings["broker"]["environment"]
    client = EToroClient(environment=environment)

    if not client.health_check():
        logger.critical("Health check fallido — CID inválido o credenciales expiradas")
        sys.exit(1)

    # ── Paso 3: Confirmación de entorno real (ya dentro de EToroClient) ─────
    # EToroClient.__init__ solicita "CONFIRMO" si environment == "real"

    # ── Paso 4: Verificar horario de mercado ────────────────────────────────
    instrument_ids: List[int] = settings["broker"]["active_instruments"]
    if not is_market_open(instrument_ids):
        logger.info("Mercado cerrado. El sistema iniciará polling pero no generará señales.")

    # ── Paso 5: ACCIÓN URGENTE — posiciones sin stop loss ──────────────────
    instrument_symbols = settings.get("broker", {}).get("instrument_symbols", {})
    market_data      = MarketData(client, instrument_symbols=instrument_symbols)
    order_executor   = OrderExecutor(client, dry_run=dry_run)
    risk_settings    = settings.get("risk", {})
    risk_manager     = RiskManager(settings=risk_settings)
    urgent_stops_cfg = risk_settings.get("urgent_stops", [])

    try:
        portfolio_raw = client.get_portfolio()
        raw_positions = (
            portfolio_raw.get("clientPortfolio", {}).get("positions", [])
        )
        alerts = risk_manager.check_urgent_stops(raw_positions, urgent_stops_cfg)
        if alerts:
            logger.warning(
                "⚠️  %d POSICIÓN(ES) SIN STOP LOSS EFECTIVO — requieren acción inmediata",
                len(alerts),
            )
            for a in alerts:
                logger.warning(
                    "   posID=%s | instrID=%s | stop_actual=%.4f | stop_requerido=%s",
                    a["position_id"], a["instrument_id"],
                    a["current_stop"], a["required_stop"],
                )
    except EToroAPIError as exc:
        logger.error("No se pudo obtener el portafolio en startup: %s", exc)

    # ── Paso 6: Modelo HMM ──────────────────────────────────────────────────
    hmm_cfg  = settings.get("hmm", {})
    
    # Extraemos retrain_every_days para la lógica del loop en vivo/dry-run
    retrain_every_days = hmm_cfg.get("retrain_every_days", 7)
    
    hmm      = HMMEngine(
        n_candidates   = hmm_cfg.get("n_candidates", [3, 4]),
        n_init         = hmm_cfg.get("n_init", 10),
        covariance_type= hmm_cfg.get("covariance_type", "diag"),
        stability_bars = hmm_cfg.get("stability_bars", 3),
        flicker_window = hmm_cfg.get("flicker_window", 20),
        flicker_threshold= hmm_cfg.get("flicker_threshold", 4),
        min_confidence = hmm_cfg.get("min_confidence", 0.55),
        model_path     = hmm_cfg.get("model_path", "models/hmm_model.pkl"),
        min_train_bars = hmm_cfg.get("min_train_bars", 126),
    )

    loaded = hmm.load()
    if not loaded or hmm.is_stale(max_age_days=RETRAIN_DAYS):
        logger.info("Entrenando HMM (modelo no encontrado o desactualizado)...")
        primary_id = instrument_ids[0]
        df = market_data.get_historical_candles(primary_id, count=756)
        if df.empty or len(df) < hmm_cfg.get("min_train_bars", 252):
            logger.critical("Datos históricos insuficientes para entrenar HMM — ABORTANDO")
            sys.exit(1)
        features = build_features(df)
        hmm.train(features.values)
        hmm.save()

    # ── Paso 7: Inicializar RiskManager con portafolio actual ───────────────
    # (risk_manager ya creado en paso 5 con settings calibrados)

    # ── Paso 8: PositionTracker ─────────────────────────────────────────────
    alert_mgr = AlertManager(settings=settings.get("monitoring", {}))

    tracker = PositionTracker(
        client          = client,
        on_state_update = lambda s: risk_manager.update_circuit_breaker(s.equity),
        on_stop_alert   = lambda a: alert_mgr.send_stop_loss_alert(a),
    )
    tracker.start()
    tracker.load_snapshot()

    # ── Paso 9: SignalGenerator — usar equity en vivo del tracker ──────────
    _ps    = tracker.get_portfolio_state()
    equity = _ps.equity if (_ps and _ps.equity > 0) else risk_settings.get("initial_equity", 560.05)
    sig_gen = SignalGenerator(hmm_engine=hmm, equity=equity)

    # ── Paso 10: Dashboard y mensaje de inicio ──────────────────────────────
    dashboard = Dashboard(settings=settings)
    portfolio_state = tracker.get_portfolio_state()
    if portfolio_state:
        dashboard.render(portfolio_state, hmm_state=None, signals=[])

    logger.info(
        "System online — eToro %s | equity: $%.2f | instrumentos: %s",
        environment.upper(), equity, instrument_ids,
    )

    return {
        "client":          client,
        "market_data":     market_data,
        "hmm":             hmm,
        "risk_manager":    risk_manager,
        "tracker":         tracker,
        "order_executor":  order_executor,
        "signal_generator":sig_gen,
        "dashboard":       dashboard,
        "alert_manager":   alert_mgr,
        "settings":        settings,
        "instrument_ids":  instrument_ids,
        "dry_run":         dry_run,
    }


# ============================================================================
# MAIN LOOP
# ============================================================================

class MainLoop:
    """
    Bucle principal de polling cada 30 segundos.

    Cada tick:
      1. Polling de precios.
      2. Polling de portafolio.
      3. Circuit breaker update.
      4. [1×día al cierre] Features + HMM prediction.
      5. Estabilidad de régimen y flicker check.
      6. Señales de estrategia.
      7. Validación de riesgo y ejecución.
      8. Re-entrenamiento semanal (lunes).
      9. Refrescar dashboard.
    """

    def __init__(self, ctx: dict, bridge: Optional[object] = None):
        self.ctx           = ctx
        self.client        : EToroClient     = ctx["client"]
        self.market_data   : MarketData      = ctx["market_data"]
        self.hmm           : HMMEngine       = ctx["hmm"]
        self.risk_manager  : RiskManager     = ctx["risk_manager"]
        self.tracker       : PositionTracker = ctx["tracker"]
        self.executor      : OrderExecutor   = ctx["order_executor"]
        self.sig_gen       : SignalGenerator = ctx["signal_generator"]
        self.dashboard     : Dashboard       = ctx["dashboard"]
        self.alert_mgr     : AlertManager    = ctx["alert_manager"]
        self.settings      : dict            = ctx["settings"]
        self.instrument_ids: List[int]       = ctx["instrument_ids"]
        self.dry_run       : bool            = ctx["dry_run"]

        # DataBridge para la GUI PySide6 (None si no hay GUI activa)
        self._bridge = bridge

        self._running          : bool             = True
        self._last_regime_date : Optional[date]   = None
        self._last_retrain_week: Optional[int]    = None
        self._regime_state                        = None
        self._bars_cache       : Dict[int, object]= {}
        self._consecutive_api_fails: int          = 0

        # Registrar manejadores de señal UNIX
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        # Estadísticas de sesión
        self._session_start  = datetime.now(timezone.utc)
        self._session_trades = 0
        self._session_pnl    = 0.0
        _init_state          = self.tracker.get_portfolio_state()
        self._equity_start   = (
            _init_state.equity if (_init_state and _init_state.equity > 0)
            else self.settings["risk"].get("initial_equity", 560.05)
        )

    def run(self) -> None:
        logger.info("[MainLoop] Iniciando bucle de polling cada %ds", POLL_SECONDS)
        while self._running:
            tick_start = time.time()
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("[MainLoop] Error no controlado en tick: %s", exc, exc_info=True)
                self.alert_mgr.send_system_error(str(exc))
                self._save_state()

            elapsed  = time.time() - tick_start
            sleep_for = max(0.0, POLL_SECONDS - elapsed)
            time.sleep(sleep_for)

    def _tick(self) -> None:
        today = date.today()

        # ── 1. Polling de precios ────────────────────────────────────────────
        rates = self._poll_prices()
        if rates:
            self.tracker.update_prices(
                {str(k): v for k, v in rates.items()}
            )
            self._consecutive_api_fails = 0
        else:
            self._consecutive_api_fails += 1
            if self._consecutive_api_fails >= 3:
                logger.error("[MainLoop] 3 fallos consecutivos de API — alertando")
                self.alert_mgr.send_api_failure(self._consecutive_api_fails)
            return

        # ── 2. Polling de portafolio ─────────────────────────────────────────
        portfolio_state = self.tracker.get_portfolio_state()
        if portfolio_state is None:
            return

        # ── 3. Circuit breaker ───────────────────────────────────────────────
        cb_status = self.risk_manager.update_circuit_breaker(
            portfolio_state.equity,
            regime=self._regime_state.label if self._regime_state else "UNKNOWN",
        )
        if cb_status in ("HALT", "LOCKED"):
            logger.warning("[MainLoop] Circuit breaker %s — sin señales este tick", cb_status)
            self._refresh_dashboard(portfolio_state)
            return

        # ── 4. Features + HMM (startup inmediato + actualización al cierre) ───
        if self._regime_state is None or (self._last_regime_date != today and is_market_close_bar()):
            self._compute_daily_regime()
            self._last_regime_date = today

        # ── 5. Re-entrenamiento semanal (lunes al inicio) ────────────────────
        current_week = today.isocalendar()[1]
        if is_monday_open() and self._last_retrain_week != current_week:
            self._retrain_hmm()
            self._last_retrain_week = current_week

        # ── 6. Generar y ejecutar señales ────────────────────────────────────
        if self._regime_state and is_market_open(self.instrument_ids):
            self._process_signals(portfolio_state, rates)

        # ── 7. Refrescar dashboard ────────────────────────────────────────────
        self._refresh_dashboard(portfolio_state)

    # ------------------------------------------------------------------
    # Polling de precios
    # ------------------------------------------------------------------

    def _poll_prices(self) -> Optional[Dict]:
        try:
            return self.market_data.get_current_rates(self.instrument_ids)
        except EToroAPIError as exc:
            logger.error("[MainLoop] Fallo polling precios: %s", exc)
            return None
        except Exception as exc:
            logger.error("[MainLoop] Error inesperado polling precios: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Features + HMM (barra diaria)
    # ------------------------------------------------------------------

    def _compute_daily_regime(self) -> None:
        """Calcula features y predice régimen con el algoritmo Forward."""
        primary_id = self.instrument_ids[0]
        try:
            df = self.market_data.get_historical_candles(primary_id, count=400)
            if df.empty or len(df) < 60:
                logger.warning("[MainLoop] Datos insuficientes para régimen diario")
                return

            features = build_features(df)
            if features.empty:
                return

            self._bars_cache[primary_id] = df

            # Algoritmo Forward (sin look-ahead)
            regime_state = self.hmm.predict_regime_filtered(features.values[-1])
            old_label    = self._regime_state.label if self._regime_state else None
            self._regime_state = regime_state

            if old_label and old_label != regime_state.label:
                self.alert_mgr.send_regime_change(old_label, regime_state.label, regime_state.probability)

            if self.hmm.is_flickering():
                self.alert_mgr.send_flicker_alert(self.hmm._regime_history)

            logger.info(
                "[MainLoop] Régimen diario: %s (%.1f%%) | confirmado=%s | barras=%d | flicker=%s",
                regime_state.label, regime_state.probability * 100,
                regime_state.is_confirmed, regime_state.consecutive_bars,
                self.hmm.is_flickering(),
            )

        except Exception as exc:
            logger.error("[MainLoop] Error en cálculo de régimen: %s", exc, exc_info=True)
            # HMM fallido → mantener régimen anterior

    # ------------------------------------------------------------------
    # Re-entrenamiento semanal
    # ------------------------------------------------------------------

    def _retrain_hmm(self) -> None:
        logger.info("[MainLoop] Re-entrenamiento semanal del HMM...")
        primary_id = self.instrument_ids[0]
        try:
            df = self.market_data.get_historical_candles(primary_id, count=756, use_cache=False)
            if len(df) < 252:
                logger.warning("[MainLoop] Datos insuficientes para re-entrenamiento")
                return
            features = build_features(df)
            self.hmm.train(features.values)
            self.hmm.save()
            self.alert_mgr.send_hmm_retrained(self.hmm.n_regimes, self.hmm.bic_score)
            logger.info("[MainLoop] HMM re-entrenado: n=%d BIC=%.2f", self.hmm.n_regimes, self.hmm.bic_score)
        except Exception as exc:
            logger.error("[MainLoop] Error en re-entrenamiento HMM: %s", exc)

    # ------------------------------------------------------------------
    # Generar y ejecutar señales
    # ------------------------------------------------------------------

    def _process_signals(self, portfolio_state: PortfolioState, rates: Dict) -> None:
        """
        Para cada instrumento:
          1. Obtener bars.
          2. Generar señal (estrategia + MACD + Donchian).
          3. Validar con RiskManager.
          4. Ejecutar si aprobada.
        """
        for instr_id in self.instrument_ids:
            bars = self._bars_cache.get(instr_id)
            if bars is None or len(bars) < 60:
                continue

            spread = self.market_data.calculate_spread_pct(instr_id)

            signal = self.sig_gen.generate(
                symbol=str(instr_id),
                bars=bars,
                regime_state=self._regime_state,
            )
            if signal is None:
                continue

            decision = self.risk_manager.validate_signal(
                signal          = signal,
                portfolio_state = portfolio_state,
                current_spread_pct = spread,
            )

            if not decision.approved:
                logger.info(
                    "[MainLoop] Señal rechazada instrID=%d: %s",
                    instr_id, decision.rejection_reason,
                )
                continue

            final_signal = decision.modified_signal
            logger.info(
                "[MainLoop] Señal APROBADA instrID=%d | size=$%.2f | stop=%.4f | %s",
                instr_id, final_signal.position_size_usd,
                final_signal.stop_loss, final_signal.strategy_name,
            )

            result = self.executor.submit_order(final_signal)
            if result.success:
                self.tracker.record_order(str(instr_id))
                if self._regime_state:
                    self.tracker.record_entry_regime(
                        str(result.position_id), self._regime_state.label
                    )
                self._session_trades += 1
                logger.info(
                    "[MainLoop] ORDEN EJECUTADA | trade_id=%s | posID=%s | instrID=%d",
                    result.trade_id, result.position_id, instr_id,
                )
            else:
                logger.error(
                    "[MainLoop] ORDEN FALLIDA | instrID=%d | error=%s",
                    instr_id, result.error,
                )
                self.alert_mgr.send_system_error(f"Orden fallida instrID={instr_id}: {result.error}")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _refresh_dashboard(self, portfolio_state: PortfolioState) -> None:
        # GUI PySide6: push thread-safe via DataBridge signals
        if self._bridge is not None:
            try:
                self._bridge.push_portfolio(portfolio_state)
                self._bridge.push_regime(self._regime_state)
                self._bridge.push_connection(True)
            except Exception as exc:
                logger.debug("[MainLoop] Error enviando datos a GUI: %s", exc)
        else:
            # Fallback: Rich terminal dashboard
            try:
                self.dashboard.render(
                    portfolio_state = portfolio_state,
                    hmm_state       = self._regime_state,
                    signals         = [],
                )
            except Exception as exc:
                logger.debug("[MainLoop] Error en dashboard: %s", exc)

    # ------------------------------------------------------------------
    # Guardar estado
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            self.tracker.save_snapshot()
        except Exception as exc:
            logger.error("[MainLoop] Error guardando snapshot: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info("[MainLoop] Señal %d recibida — iniciando shutdown seguro...", signum)
        self._running = False
        self._shutdown()
        # Cerrar Qt si está activo (thread-safe: QApplication.quit es reentrant)
        if _GUI_AVAILABLE:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def _shutdown(self) -> None:
        logger.info("[MainLoop] SHUTDOWN — NO se cierran posiciones (stops activos en broker)")

        # Guardar snapshot
        self._save_state()

        # Resumen de sesión
        duration = datetime.now(timezone.utc) - self._session_start
        state    = self.tracker.get_portfolio_state()
        final_eq = state.equity if state else self._equity_start
        pnl      = final_eq - self._equity_start

        logger.info("=" * 60)
        logger.info("RESUMEN DE SESIÓN")
        logger.info("  Duración     : %s", str(duration).split(".")[0])
        logger.info("  Operaciones  : %d", self._session_trades)
        logger.info("  P&L sesión   : $%.2f", pnl)
        logger.info("  Equity final : $%.2f", final_eq)
        logger.info("  Snapshot     : state_snapshot.json")
        logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="regime-trader — eToro Algorithmic Trading System"
    )
    p.add_argument("--dry-run",     action="store_true", help="Pipeline completo sin órdenes reales")
    p.add_argument("--backtest",    action="store_true", help="Ejecutar backtester walk-forward")
    p.add_argument("--train-only",  action="store_true", help="Entrenar HMM y salir")
    p.add_argument("--stress-test", action="store_true", help="Ejecutar pruebas de estrés")
    p.add_argument("--compare",     action="store_true", help="Backtest + comparativa de benchmarks")
    p.add_argument("--dashboard",   action="store_true", help="Mostrar dashboard (instancia ya activa)")
    p.add_argument("--config",      default="config/settings.yaml", help="Ruta a settings.yaml")
    p.add_argument("--symbols",     nargs="+", type=int, help="Filtrar instrumentos (IDs eToro)")
    p.add_argument("--start",       help="Fecha inicio backtest (YYYY-MM-DD)")
    p.add_argument("--end",         help="Fecha fin backtest (YYYY-MM-DD)")
    return p.parse_args()


def run_backtest(settings: dict, compare: bool = False, symbols: Optional[List[int]] = None) -> None:
    from backtest.backtester  import WalkForwardBacktester
    from backtest.performance import PerformanceCalculator

    logger.info("[CLI] Iniciando backtest walk-forward...")
    client      = EToroClient(environment="demo")
    instrument_symbols = settings.get("broker", {}).get("instrument_symbols", {})
    market_data = MarketData(client, instrument_symbols=instrument_symbols)

    instrument_ids = symbols or settings["broker"]["active_instruments"]
    bars_by_symbol = {}
    for iid in instrument_ids:
        df = market_data.get_historical_candles(iid, count=1260)
        if not df.empty:
            bars_by_symbol[str(iid)] = df

    if not bars_by_symbol:
        logger.error("No se pudieron obtener datos históricos para el backtest.")
        return

    backtester = WalkForwardBacktester(
        initial_capital=settings["backtest"]["initial_capital"],
        commission=settings["backtest"]["commission_per_trade"],
        slippage_pct=settings["backtest"]["slippage_pct"],
    )
    result = backtester.run(bars_by_symbol)

    calc    = PerformanceCalculator()
    metrics = calc.compute(result.equity_curve, result.trade_log, label="regime-trader")
    regime_stats  = calc.regime_breakdown(result.equity_curve, result.trade_log)
    conf_buckets  = calc.confidence_buckets(result.trade_log)

    benchmarks = []
    if compare:
        primary_prices = list(bars_by_symbol.values())[0]["close"]
        benchmarks.append(calc.benchmark_buyhold(primary_prices))
        benchmarks.append(calc.benchmark_sma200(primary_prices))
        bm_rand, _ = calc.benchmark_random(primary_prices, rebalance_freq=21)
        benchmarks.append(bm_rand)

    calc.print_summary(metrics, benchmarks, regime_stats, conf_buckets)
    calc.save_csv(result.equity_curve, result.trade_log, result.regime_history, [metrics] + benchmarks)

    validation = "✅ PASA" if result.passes_validation else "❌ NO PASA"
    logger.info("Validación OOS (≥4.66%%/mes): %s", validation)


def run_stress_test(settings: dict) -> None:
    from backtest.stress_test import StressTester
    import pandas as pd

    logger.info("[CLI] Ejecutando pruebas de estrés...")
    client      = EToroClient(environment="demo")
    instrument_symbols = settings.get("broker", {}).get("instrument_symbols", {})
    market_data = MarketData(client, instrument_symbols=instrument_symbols)
    primary_id  = settings["broker"]["active_instruments"][0]

    df  = market_data.get_historical_candles(primary_id, count=756)
    if df.empty:
        logger.error("Sin datos para stress test.")
        return

    import ta
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    tester    = StressTester()
    eq_dummy  = pd.DataFrame({"equity": df["close"] / df["close"].iloc[0] * 546.14,
                               "regime": "NEUTRAL"})

    def mock_strategy(label: str) -> float:
        mapping = {"BULL": 0.95, "NEUTRAL": 0.60, "BEAR": 0.30, "CRASH": 0.10}
        return mapping.get(label, 0.60)

    tester.run_all(
        equity_curve  = eq_dummy,
        trade_log     = pd.DataFrame(),
        price_series  = df["close"],
        atr_series    = atr,
        allocations   = {str(primary_id): 0.92},
        strategy_fn   = mock_strategy,
    )


def run_train_only(settings: dict) -> None:
    logger.info("[CLI] Modo train-only — entrenando HMM y saliendo...")
    client      = EToroClient(environment="demo")
    instrument_symbols = settings.get("broker", {}).get("instrument_symbols", {})
    market_data = MarketData(client, instrument_symbols=instrument_symbols)
    hmm_cfg     = settings.get("hmm", {})
    primary_id  = settings["broker"]["active_instruments"][0]

    df = market_data.get_historical_candles(primary_id, count=756)
    if df.empty or len(df) < 252:
        logger.error("Datos insuficientes para entrenar HMM.")
        return

    features = build_features(df)
    hmm = HMMEngine(
        n_candidates    = hmm_cfg.get("n_candidates", [3, 4, 5, 6, 7]),
        n_init          = hmm_cfg.get("n_init", 10),
        covariance_type = hmm_cfg.get("covariance_type", "full"),
        model_path      = hmm_cfg.get("model_path", "models/hmm_model.pkl"),
    )
    hmm.train(features.values)
    hmm.save()
    logger.info("HMM entrenado: n_regimes=%d BIC=%.2f — guardado en %s",
                hmm.n_regimes, hmm.bic_score, hmm.model_path)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    args     = parse_args()
    settings = load_settings(args.config)

    setup_logging(settings.get("monitoring", {}))
    logger.info("regime-trader iniciando | config=%s", args.config)

    if args.train_only:
        run_train_only(settings)
        return

    if args.stress_test:
        run_stress_test(settings)
        return

    if args.backtest or args.compare:
        run_backtest(settings, compare=args.compare, symbols=args.symbols)
        return

    if args.dashboard:
        logger.info("--dashboard: conectar a instancia activa (ver state_snapshot.json)")
        return

    # ── Modo live / dry-run ──────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=== DRY RUN — sin órdenes reales ===")

    ctx  = startup(settings, dry_run=args.dry_run)
    loop = MainLoop(ctx)
    loop.run()


if __name__ == "__main__":
    main()
