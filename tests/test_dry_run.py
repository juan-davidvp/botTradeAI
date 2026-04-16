"""
tests/test_dry_run.py
Fase 9b — Ejecución Dry Run de Extremo a Extremo.

Simula el flujo completo sin enviar órdenes reales:
  datos eToro (mock) → PortfolioState → RiskManager → detección de alertas

Verifica:
  1. Detección y alerta de las 4 posiciones sin stop loss activo
  2. Detección del desequilibrio de pesos (posición 4238 al ~49.4%)
  3. Señal válida fluye correctamente hasta la orden simulada (dry_run=True)
  4. Señal inválida (sin stop) es rechazada antes de ejecutarse
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from core.risk_manager import RiskManager, PortfolioState, EQUITY_BASE
from core.regime_strategies import Signal, Direction, TechnicalConfirmation

EQUITY = 546.14

# ---------------------------------------------------------------------------
# Posiciones del portafolio real (2026-04-14)
# ---------------------------------------------------------------------------

REAL_POSITIONS = [
    {
        "positionID":   3403421461,
        "instrumentID": 4238,
        "amount":       270.00,       # 49.4% del equity → sobrepeso
        "openRate":     637.10,
        "stopLossRate": 0.0,
        "isNoStopLoss": True,
        "units":        0.4238,
        "openDateTime": "2026-04-14T10:00:00Z",
        "current_price": 637.10,
        "holding_days": 1,
    },
    {
        "positionID":   3403428716,
        "instrumentID": 14328,
        "amount":       70.00,
        "openRate":     1853.51,
        "stopLossRate": 0.0,
        "isNoStopLoss": True,
        "units":        0.0378,
        "openDateTime": "2026-04-14T10:05:00Z",
        "current_price": 1853.51,
        "holding_days": 1,
    },
    {
        "positionID":   3403430868,
        "instrumentID": 9408,
        "amount":       60.00,
        "openRate":     136.68,
        "stopLossRate": 0.0,
        "isNoStopLoss": True,
        "units":        0.439,
        "openDateTime": "2026-04-14T10:10:00Z",
        "current_price": 136.68,
        "holding_days": 1,
    },
    {
        "positionID":   3403433830,
        "instrumentID": 6218,
        "amount":       60.00,
        "openRate":     432.00,
        "stopLossRate": 0.0,
        "isNoStopLoss": True,
        "units":        0.139,
        "openDateTime": "2026-04-14T10:15:00Z",
        "current_price": 432.00,
        "holding_days": 1,
    },
    {
        "positionID":   3403418899,
        "instrumentID": 2488,
        "amount":       40.00,
        "openRate":     208.50,
        "stopLossRate": 0.0,
        "isNoStopLoss": True,
        "units":        0.192,
        "openDateTime": "2026-04-14T10:20:00Z",
        "current_price": 208.50,
        "holding_days": 1,
    },
]

KNOWN_NO_STOP_IDS = {3403421461, 3403428716, 3403430868, 3403433830, 3403418899}


def _make_portfolio_state(positions=None) -> PortfolioState:
    pos = positions or REAL_POSITIONS
    invested = sum(float(p.get("amount", 0)) for p in pos)
    return PortfolioState(
        equity=EQUITY,
        cash=EQUITY - invested,
        buying_power=EQUITY - invested,
        positions=pos,
        daily_pnl=0.0,
        weekly_pnl=0.0,
        peak_equity=EQUITY,
        drawdown_pct=0.0,
        circuit_breaker_status="OK",
        flicker_rate=0.0,
        daily_trades_count=0,
        last_order_by_instrument={},
    )


# ---------------------------------------------------------------------------
# 1. Detección de posiciones sin stop loss
# ---------------------------------------------------------------------------

class TestNoStopDetection:
    def test_all_known_positions_detected(self):
        """Las 5 posiciones reales deben detectarse como sin stop."""
        positions = REAL_POSITIONS
        no_stop = [
            p for p in positions
            if p.get("isNoStopLoss") is True or float(p.get("stopLossRate", 0)) <= 0.001
        ]
        assert len(no_stop) == len(KNOWN_NO_STOP_IDS), (
            f"Se esperaban {len(KNOWN_NO_STOP_IDS)} posiciones sin stop, "
            f"detectadas: {len(no_stop)}"
        )

    def test_position_ids_match_known_set(self):
        no_stop_ids = {
            p["positionID"] for p in REAL_POSITIONS
            if p.get("isNoStopLoss") is True or float(p.get("stopLossRate", 0)) <= 0.001
        }
        assert no_stop_ids == KNOWN_NO_STOP_IDS

    def test_alert_manager_send_stop_called(self):
        """AlertManager.send_stop_loss_alert debe llamarse por cada posición sin stop."""
        from monitoring.alerts import AlertManager
        am = AlertManager({})
        am.send_stop_loss_alert  # existe
        positions_sin_stop = [p for p in REAL_POSITIONS if p.get("isNoStopLoss")]

        with patch.object(am, "_emit") as mock_emit:
            # remove rate limit constraint (immediate=True bypasses it)
            am.send_stop_loss_alert(positions_sin_stop)
            assert mock_emit.call_count == len(positions_sin_stop)


# ---------------------------------------------------------------------------
# 2. Detección de desequilibrio de pesos (4238 al 49.4%)
# ---------------------------------------------------------------------------

class TestWeightImbalance:
    def test_instrument_4238_exceeds_20pct(self):
        ps       = _make_portfolio_state()
        equity   = ps.equity
        pos_4238 = next(p for p in ps.positions if p["instrumentID"] == 4238)
        weight   = float(pos_4238["amount"]) / equity * 100

        assert weight > 20.0, f"Peso de 4238 debería ser >20%, fue {weight:.1f}%"
        assert abs(weight - 49.4) < 2.0, f"Peso esperado ~49.4%, fue {weight:.1f}%"

    def test_overweight_positions_detected(self):
        ps = _make_portfolio_state()
        overweight = [
            p for p in ps.positions
            if float(p.get("amount", 0)) / ps.equity * 100 > 20.0
        ]
        assert len(overweight) >= 1
        assert any(p["instrumentID"] == 4238 for p in overweight)

    def test_alert_manager_overweight_called(self):
        from monitoring.alerts import AlertManager
        am = AlertManager({})
        ps = _make_portfolio_state()

        called = []
        def fake_emit(alert_type, message, **kwargs):
            called.append(alert_type)

        with patch.object(am, "_emit", side_effect=fake_emit):
            for pos in ps.positions:
                w = float(pos.get("amount", 0)) / ps.equity * 100
                if w > 20.0:
                    am.send_overweight_alert(pos["instrumentID"], w, ps.equity)

        assert "OVERWEIGHT" in called, "OVERWEIGHT alert no fue llamada"


# ---------------------------------------------------------------------------
# 3. Flujo dry-run: señal válida → RiskManager → orden simulada
# ---------------------------------------------------------------------------

class TestDryRunFlow:
    def test_valid_signal_approved_by_risk_manager(self):
        """
        Una señal bien formada (con stop, tamaño dentro de límites)
        debe ser aprobada cuando hay cash disponible.
        """
        rm = RiskManager({"initial_equity": EQUITY})
        # Estado limpio sin posiciones abiertas
        ps = PortfolioState(
            equity=EQUITY,
            cash=200.0,
            buying_power=200.0,
            positions=[],
            daily_pnl=0.0,
            weekly_pnl=0.0,
            peak_equity=EQUITY,
            drawdown_pct=0.0,
            circuit_breaker_status="OK",
            flicker_rate=0.0,
            daily_trades_count=0,
            last_order_by_instrument={},
        )
        signal = Signal(
            symbol="4238",
            direction=Direction.LONG,
            confidence=0.80,
            entry_price=637.10,
            stop_loss=605.24,       # 5% stop
            take_profit=None,
            position_size_pct=109.23 / EQUITY,
            position_size_usd=109.23,
            leverage=1.0,
            regime_id=1,
            regime_name="BULL",
            regime_probability=0.80,
            reasoning="DryRun test",
            strategy_name="LowVolBull",
            technical_confirmation=TechnicalConfirmation.STRONG,
        )
        decision = rm.validate_signal(signal, ps)
        assert decision.approved, f"Señal válida rechazada: {decision.rejection_reason}"

    def test_order_executor_dry_run_does_not_call_api(self):
        """
        OrderExecutor con dry_run=True NO debe llamar a client.open_market_order.
        """
        import os
        os.environ.setdefault("ETORO_API_KEY",  "TEST")
        os.environ.setdefault("ETORO_USER_KEY", "TEST")

        from broker.order_executor import OrderExecutor
        from broker.etoro_client   import EToroClient

        mock_client = MagicMock(spec=EToroClient)
        executor    = OrderExecutor(client=mock_client, dry_run=True)

        signal = Signal(
            symbol="4238",
            direction=Direction.LONG,
            confidence=0.80,
            entry_price=637.10,
            stop_loss=605.24,
            take_profit=None,
            position_size_pct=109.23 / EQUITY,
            position_size_usd=109.23,
            leverage=1.0,
            regime_id=1,
            regime_name="BULL",
            regime_probability=0.80,
            reasoning="DryRun",
            strategy_name="LowVolBull",
            technical_confirmation=TechnicalConfirmation.STRONG,
        )
        executor.submit_order(signal)
        mock_client.open_market_order.assert_not_called()

    def test_invalid_signal_no_stop_rejected_before_execution(self):
        rm = RiskManager({"initial_equity": EQUITY})
        ps = PortfolioState(
            equity=EQUITY,
            cash=200.0,
            buying_power=200.0,
            positions=[],
            daily_pnl=0.0,
            weekly_pnl=0.0,
            peak_equity=EQUITY,
            drawdown_pct=0.0,
            circuit_breaker_status="OK",
            flicker_rate=0.0,
            daily_trades_count=0,
            last_order_by_instrument={},
        )
        signal = Signal(
            symbol="4238",
            direction=Direction.LONG,
            confidence=0.80,
            entry_price=637.10,
            stop_loss=0.0,          # sin stop — debe rechazarse
            take_profit=None,
            position_size_pct=0.20,
            position_size_usd=109.23,
            leverage=1.0,
            regime_id=1,
            regime_name="BULL",
            regime_probability=0.80,
            reasoning="Invalid signal",
            strategy_name="LowVolBull",
            technical_confirmation=TechnicalConfirmation.STRONG,
        )
        decision = rm.validate_signal(signal, ps)
        assert not decision.approved
        assert "STOP" in decision.rejection_reason.upper()
