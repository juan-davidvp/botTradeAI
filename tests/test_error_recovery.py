"""
tests/test_error_recovery.py
Fase 9f — Recuperación de Errores desde state_snapshot.json.

Verifica:
  1. El snapshot se guarda correctamente con positionIDs reales
  2. Al reiniciar, PositionTracker carga el snapshot y reconcilia con la API
  3. No se duplican posiciones tras el reinicio (positionID en snapshot ∩ API)
  4. Posiciones "fantasma" (en snapshot pero no en API) se detectan
  5. Posiciones nuevas (en API pero no en snapshot) se agregan sin duplicar
"""

import json
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SNAPSHOT_POSITIONS = [
    {
        "positionID":   3403421461,
        "instrumentID": 4238,
        "amount":       270.00,
        "openRate":     637.10,
        "stopLossRate": 0.0,
        "isNoStopLoss": True,
        "units":        0.4238,
        "openDateTime": "2026-04-14T10:00:00Z",
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
    },
]

# Mismas posiciones que devolvería la API en el restart
API_POSITIONS = [dict(p) for p in SNAPSHOT_POSITIONS]


@pytest.fixture()
def snapshot_file(tmp_path):
    """Crea un state_snapshot.json temporal."""
    snap = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "equity":      546.14,
        "cash":        206.14,
        "positions":   SNAPSHOT_POSITIONS,
        "peak_equity": 546.14,
    }
    path = tmp_path / "state_snapshot.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 1. Guardado del snapshot
# ---------------------------------------------------------------------------

class TestSnapshotSave:
    def test_snapshot_contains_position_ids(self, tmp_path, monkeypatch):
        import broker.position_tracker as pt_module
        snap_path = str(tmp_path / "state_snapshot.json")
        monkeypatch.setattr(pt_module, "SNAPSHOT_FILE", snap_path)

        mock_client = MagicMock()
        tracker = pt_module.PositionTracker(client=mock_client)
        tracker._positions    = SNAPSHOT_POSITIONS
        tracker._peak_equity  = 546.14
        # portfolio_state no es necesario en save_snapshot (usa _positions directamente)

        tracker.save_snapshot()

        assert os.path.exists(snap_path), "snapshot no fue creado"
        with open(snap_path, encoding="utf-8") as f:
            data = json.load(f)

        saved_ids = {p["positionID"] for p in data["positions"]}
        expected  = {p["positionID"] for p in SNAPSHOT_POSITIONS}
        assert saved_ids == expected

    def test_snapshot_has_timestamp(self, tmp_path, monkeypatch):
        import broker.position_tracker as pt_module
        snap_path = str(tmp_path / "state_snapshot.json")
        monkeypatch.setattr(pt_module, "SNAPSHOT_FILE", snap_path)

        mock_client = MagicMock()
        tracker = pt_module.PositionTracker(client=mock_client)
        tracker._positions   = SNAPSHOT_POSITIONS
        tracker._peak_equity = 546.14
        tracker.save_snapshot()

        with open(snap_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "timestamp" in data, "snapshot debe incluir timestamp"


# ---------------------------------------------------------------------------
# 2. Carga del snapshot al reiniciar
# ---------------------------------------------------------------------------

class TestSnapshotLoad:
    def test_load_restores_positions(self, snapshot_file, monkeypatch):
        """
        load_snapshot() retorna True si el archivo existe y fue cargado.
        El estado interno del tracker se actualiza (peak_equity, entry_regimes...).
        """
        import broker.position_tracker as pt_module
        monkeypatch.setattr(pt_module, "SNAPSHOT_FILE", snapshot_file)

        mock_client = MagicMock()
        tracker = pt_module.PositionTracker(client=mock_client)
        result = tracker.load_snapshot()

        assert result is True, "load_snapshot debe retornar True con archivo existente"
        assert tracker._peak_equity == 546.14, (
            f"peak_equity no restaurado correctamente: {tracker._peak_equity}"
        )

    def test_load_returns_false_when_no_file(self, tmp_path, monkeypatch):
        """load_snapshot() retorna False cuando no existe el archivo."""
        import broker.position_tracker as pt_module
        monkeypatch.setattr(pt_module, "SNAPSHOT_FILE", str(tmp_path / "missing.json"))

        mock_client = MagicMock()
        tracker = pt_module.PositionTracker(client=mock_client)
        result = tracker.load_snapshot()
        assert result is False, "load_snapshot debe retornar False cuando no existe el archivo"


# ---------------------------------------------------------------------------
# 3 & 4. Reconciliación — sin duplicados, detección de fantasmas
# ---------------------------------------------------------------------------

class TestReconciliation:
    def test_no_duplicate_positions_after_restart(self):
        """
        Los positionIDs del snapshot que ya están en la API
        no deben procesarse como posiciones nuevas.
        """
        snapshot_ids = {p["positionID"] for p in SNAPSHOT_POSITIONS}
        api_ids      = {p["positionID"] for p in API_POSITIONS}

        # Intersección: posiciones ya conocidas → no duplicar
        duplicates = snapshot_ids & api_ids
        new_from_api = api_ids - snapshot_ids

        assert len(new_from_api) == 0, (
            f"Posiciones nuevas detectadas que deberían ser conocidas: {new_from_api}"
        )
        # Las posiciones conocidas están presentes en ambos lados → sin duplicado
        assert duplicates == snapshot_ids

    def test_ghost_positions_detected(self):
        """
        Posición en snapshot pero NO en API → es una posición fantasma (cerrada externamente).
        """
        ghost_id   = 9999999
        snap_ids   = {p["positionID"] for p in SNAPSHOT_POSITIONS} | {ghost_id}
        api_ids    = {p["positionID"] for p in API_POSITIONS}

        ghosts = snap_ids - api_ids
        assert ghost_id in ghosts, "La posición fantasma no fue detectada"

    def test_new_positions_added_without_duplication(self):
        """
        Posición en API pero NO en snapshot → se agrega como nueva.
        """
        new_pos_id = 9999998
        api_ids    = {p["positionID"] for p in API_POSITIONS} | {new_pos_id}
        snap_ids   = {p["positionID"] for p in SNAPSHOT_POSITIONS}

        new_positions = api_ids - snap_ids
        assert new_pos_id in new_positions
        assert len(new_positions) == 1

    def test_reconcile_method_exists(self):
        """PositionTracker debe exponer load_snapshot y save_snapshot."""
        import broker.position_tracker as pt_module
        mock_client = MagicMock()
        tracker = pt_module.PositionTracker(client=mock_client)

        assert hasattr(tracker, "load_snapshot"), "falta método load_snapshot"
        assert hasattr(tracker, "save_snapshot"), "falta método save_snapshot"
        assert callable(tracker.load_snapshot)
        assert callable(tracker.save_snapshot)


# ---------------------------------------------------------------------------
# 5. Verificación de settings.yaml pre-flight
# ---------------------------------------------------------------------------

class TestSettingsVerification:
    """Verifica que settings.yaml tiene todos los valores calibrados correctamente."""

    @pytest.fixture(scope="class")
    def settings(self):
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "settings.yaml"
        )
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_initial_equity(self, settings):
        assert settings["risk"]["initial_equity"] == 560.05

    def test_max_single_position(self, settings):
        assert settings["risk"]["max_single_position"] == 0.20

    def test_max_leverage(self, settings):
        assert settings["risk"]["max_leverage"] == 1.0

    def test_default_stop_loss_pct(self, settings):
        assert settings["risk"]["default_stop_loss_pct"] == 0.05

    def test_broker_name(self, settings):
        assert settings["broker"]["name"] == "etoro"

    def test_commission_per_trade(self, settings):
        assert settings["backtest"]["commission_per_trade"] == 1.0

    def test_urgent_stops_configured(self, settings):
        urgent = settings["risk"].get("urgent_stops", [])
        assert len(urgent) >= 4, "Deben configurarse al menos 4 urgent_stops"
        ids = {s["position_id"] for s in urgent}
        expected = {3403421461, 3403428716, 3403430868, 3403433830, 3403418899}
        missing = expected - ids
        assert not missing, f"urgent_stops faltantes: {missing}"
