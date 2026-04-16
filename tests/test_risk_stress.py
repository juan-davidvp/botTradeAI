"""
tests/test_risk_stress.py
Fase 9d — Estrés de Riesgo calibrado a $546.14.

Verifica:
  1. Señal de $200 se capea a $109.23 (20% de $546.14)
  2. Señal sin stopLossRate es rechazada
  3. Circuit breaker se activa al simular P&L de -$16.39 (>3% diario)
  4. Bloqueo de duplicados: mismo instrumentID en < 60s rechazado
"""

import time
import pytest
from datetime import datetime

from core.risk_manager import (
    CircuitBreaker,
    RiskManager,
    PortfolioState,
    EQUITY_BASE,
    MAX_POSITION_USD,
    MIN_POSITION_USD,
    CB_DAILY_HALT_PCT,
)
from core.regime_strategies import Signal, Direction, TechnicalConfirmation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EQUITY = 546.14

def _make_portfolio(
    equity: float = EQUITY,
    cash: float = 100.0,
    positions: list = None,
    daily_pnl: float = 0.0,
    daily_trades: int = 0,
    last_order_ts: dict = None,
) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=cash,
        buying_power=cash,
        positions=positions or [],
        daily_pnl=daily_pnl,
        weekly_pnl=0.0,
        peak_equity=equity,
        drawdown_pct=0.0,
        circuit_breaker_status="OK",
        flicker_rate=0.0,
        daily_trades_count=daily_trades,
        last_order_by_instrument=last_order_ts or {},
    )


def _make_signal(
    symbol: str = "4238",
    size_usd: float = 109.23,
    stop_loss: float = 605.24,
    entry_price: float = 637.10,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=Direction.LONG,
        confidence=0.75,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=None,
        position_size_pct=size_usd / EQUITY,
        position_size_usd=size_usd,
        leverage=1.0,
        regime_id=1,
        regime_name="BULL",
        regime_probability=0.75,
        reasoning="Test signal",
        strategy_name="TestStrategy",
        technical_confirmation=TechnicalConfirmation.STRONG,
    )


# ---------------------------------------------------------------------------
# 1. Señal de $200 debe caparse a $109.23
# ---------------------------------------------------------------------------

class TestPositionCap:
    def test_200_usd_signal_capped_to_max(self):
        rm      = RiskManager({"initial_equity": EQUITY})
        signal  = _make_signal(size_usd=200.0)
        ps      = _make_portfolio(cash=250.0)

        decision = rm.validate_signal(signal, ps)

        assert decision.approved, f"Señal rechazada inesperadamente: {decision.rejection_reason}"
        final = decision.modified_signal.position_size_usd
        assert final <= MAX_POSITION_USD + 0.01, (
            f"Tamaño final ${final:.2f} supera el límite de ${MAX_POSITION_USD:.2f}"
        )

    def test_200_usd_capped_modification_logged(self):
        rm      = RiskManager({"initial_equity": EQUITY})
        signal  = _make_signal(size_usd=200.0)
        ps      = _make_portfolio(cash=250.0)

        decision = rm.validate_signal(signal, ps)
        mods = " ".join(decision.modifications_list)
        assert "CAPPED" in mods or "REDUCIDO" in mods or "LIMITADO" in mods, (
            f"No se registró modificación de cap. Modificaciones: {decision.modifications_list}"
        )

    def test_100_usd_signal_passes_through(self):
        """Señal dentro del límite (≤$109.23) debe ser aprobada."""
        rm     = RiskManager({"initial_equity": EQUITY})
        signal = _make_signal(size_usd=109.23)
        ps     = _make_portfolio(cash=150.0)

        decision = rm.validate_signal(signal, ps)
        assert decision.approved, (
            f"Señal de $109.23 rechazada inesperadamente: {decision.rejection_reason}"
        )
        # El tamaño final nunca debe superar el límite del 20%
        if decision.modified_signal:
            assert decision.modified_signal.position_size_usd <= MAX_POSITION_USD + 0.01


# ---------------------------------------------------------------------------
# 2. Señal sin stop loss debe ser rechazada
# ---------------------------------------------------------------------------

class TestStopLossRequired:
    def test_no_stop_loss_rejected(self):
        rm      = RiskManager({"initial_equity": EQUITY})
        signal  = _make_signal(stop_loss=0.0)   # sin stop
        ps      = _make_portfolio(cash=150.0)

        decision = rm.validate_signal(signal, ps)
        assert not decision.approved
        assert "STOP" in decision.rejection_reason.upper(), (
            f"El motivo de rechazo no menciona STOP: {decision.rejection_reason}"
        )

    def test_stop_loss_below_threshold_rejected(self):
        rm     = RiskManager({"initial_equity": EQUITY})
        signal = _make_signal(stop_loss=0.0005)  # <= 0.001
        ps     = _make_portfolio(cash=150.0)

        decision = rm.validate_signal(signal, ps)
        assert not decision.approved

    def test_valid_stop_loss_approved(self):
        rm     = RiskManager({"initial_equity": EQUITY})
        signal = _make_signal(stop_loss=605.24, size_usd=109.23)
        ps     = _make_portfolio(cash=150.0)

        decision = rm.validate_signal(signal, ps)
        assert decision.approved, decision.rejection_reason


# ---------------------------------------------------------------------------
# 3. Circuit breaker — activación por DD diario > 3% ($16.39)
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_daily_halt_triggered_at_3pct(self):
        cb = CircuitBreaker(base_equity=EQUITY)
        # Simular pérdida de $16.39 (3.0% de $546.14)
        loss_usd    = EQUITY * CB_DAILY_HALT_PCT + 0.01   # justo por encima
        equity_now  = EQUITY - loss_usd
        status      = cb.update(equity_now, regime="BULL")

        assert status in ("HALT", "LOCKED"), (
            f"Se esperaba HALT o LOCKED, se obtuvo: {status}"
        )

    def test_daily_halt_blocks_trading(self):
        cb = CircuitBreaker(base_equity=EQUITY)
        cb.update(EQUITY - 16.39, regime="BULL")   # activa HALT
        assert cb.size_multiplier() == 0.0

    def test_reduce_50_at_2pct(self):
        cb = CircuitBreaker(base_equity=EQUITY)
        cb.update(EQUITY - 10.93, regime="BULL")   # 2.0%+ → REDUCE_50
        assert cb.size_multiplier() == 0.50

    def test_ok_under_2pct(self):
        cb = CircuitBreaker(base_equity=EQUITY)
        cb.update(EQUITY - 5.00, regime="BULL")    # < 2% → OK
        assert cb.size_multiplier() == 1.0

    def test_circuit_breaker_blocks_signal(self):
        """Cuando el CB está en HALT, validate_signal debe rechazar."""
        import os
        # Limpiar lock file por si existe de otro test
        if os.path.exists("trading_halted.lock"):
            os.remove("trading_halted.lock")

        rm     = RiskManager({"initial_equity": EQUITY})
        # Simular HALT manualmente
        rm.circuit_breaker._day_start_eq = EQUITY
        rm.circuit_breaker.update(EQUITY - 16.39, regime="BULL")

        signal   = _make_signal(size_usd=109.23)
        ps       = _make_portfolio(cash=150.0)
        decision = rm.validate_signal(signal, ps)

        assert not decision.approved
        assert "CIRCUIT" in decision.rejection_reason.upper() or "HALT" in decision.rejection_reason.upper()

    def test_peak_dd_creates_lock_file(self, tmp_path, monkeypatch):
        """Peak DD > 10% genera trading_halted.lock."""
        import core.risk_manager as rm_module
        lock_path = str(tmp_path / "trading_halted.lock")
        monkeypatch.setattr(rm_module, "LOCK_FILE", lock_path)

        cb = CircuitBreaker(base_equity=EQUITY)
        # DD > 10%
        cb.update(EQUITY - (EQUITY * 0.11), regime="BEAR")
        assert cb.status == "LOCKED"
        assert (tmp_path / "trading_halted.lock").exists()


# ---------------------------------------------------------------------------
# 4. Bloqueo de duplicados — mismo instrumento en < 60s
# ---------------------------------------------------------------------------

class TestDuplicateBlock:
    def test_same_instrument_within_60s_rejected(self):
        rm     = RiskManager({"initial_equity": EQUITY})
        signal = _make_signal(symbol="4238", size_usd=109.23)

        # Registrar una orden reciente para el instrumento
        ps = _make_portfolio(
            cash=250.0,
            last_order_ts={"4238": time.time()},   # ahora mismo
        )
        decision = rm.validate_signal(signal, ps)

        assert not decision.approved
        assert "DUPLICAD" in decision.rejection_reason.upper() or "DUP" in decision.rejection_reason.upper()

    def test_same_instrument_after_60s_allowed(self):
        rm     = RiskManager({"initial_equity": EQUITY})
        signal = _make_signal(symbol="4238", size_usd=109.23)

        # Registrar orden de hace 61 segundos
        ps = _make_portfolio(
            cash=250.0,
            last_order_ts={"4238": time.time() - 61},
        )
        decision = rm.validate_signal(signal, ps)

        # No debe rechazar por duplicado (puede rechazarse por otras razones)
        if not decision.approved:
            assert "DUPLICAD" not in decision.rejection_reason.upper()

    def test_different_instrument_not_blocked(self):
        rm = RiskManager({"initial_equity": EQUITY})

        signal_a = _make_signal(symbol="4238", size_usd=109.23)
        signal_b = _make_signal(symbol="9408",  size_usd=109.23, entry_price=136.68, stop_loss=129.85)

        ps = _make_portfolio(
            cash=250.0,
            last_order_ts={"4238": time.time()},   # solo 4238 bloqueado
        )
        decision_b = rm.validate_signal(signal_b, ps)
        # 9408 no está bloqueado — no debe rechazarse por duplicado
        if not decision_b.approved:
            assert "DUPLICAD" not in decision_b.rejection_reason.upper()
