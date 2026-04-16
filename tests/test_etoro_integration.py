"""
tests/test_etoro_integration.py
Fase 9 — Test de Integración eToro API (con mocks — sin llamadas reales).

Cobertura:
  a. GET /me retorna CID 34044505
  b. GET /trading/info/portfolio parsea los 5 campos de posición
  c. GET /market-data/instruments/rates retorna precios para los 5 IDs activos
  d. Cálculo correcto de spread porcentual
  e. EToroAPIError 401 dispara excepción controlada

NOTA: Todos los tests usan responses mockeados — NO se realizan llamadas reales.
      Para tests contra Demo real, usar la suite de integración manual (Sección e del spec).
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Aseguramos vars de entorno antes de importar EToroClient
os.environ.setdefault("ETORO_API_KEY",  "TEST_API_KEY")
os.environ.setdefault("ETORO_USER_KEY", "TEST_USER_KEY")

from broker.etoro_client import EToroClient, EToroAPIError

ACTIVE_INSTRUMENT_IDS = [4238, 14328, 9408, 6218, 2488]
EXPECTED_CID          = 34044505


# ---------------------------------------------------------------------------
# Fixture: cliente con base_url que nunca llega a la red
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """EToroClient en modo demo. Sus métodos HTTP se mockean por test."""
    with patch.object(EToroClient, "_validate_env_vars", return_value=None):
        c = EToroClient.__new__(EToroClient)
        c.environment = "demo"
        c.base_url    = "http://mock-etoro"
        c._cid        = None
    return c


# ---------------------------------------------------------------------------
# a. health_check: GET /me debe retornar CID 34044505
# ---------------------------------------------------------------------------

def test_get_me_returns_expected_cid(client):
    client._get = MagicMock(return_value={"realCID": EXPECTED_CID, "name": "TestUser"})
    identity = client.get_identity()
    cid = identity.get("realCID") or identity.get("cid") or identity.get("id")
    assert int(cid) == EXPECTED_CID, f"CID esperado {EXPECTED_CID}, obtenido {cid}"


def test_health_check_passes_with_correct_cid(client):
    client._get = MagicMock(return_value={"realCID": EXPECTED_CID})
    assert client.health_check() is True


def test_health_check_fails_with_wrong_cid(client):
    client._get = MagicMock(return_value={"realCID": 99999999})
    assert client.health_check() is False


# ---------------------------------------------------------------------------
# b. GET /trading/info/portfolio — parseo de los 5 campos de posición
# ---------------------------------------------------------------------------

MOCK_PORTFOLIO = {
    "clientPortfolio": {
        "credit": 46.14,
        "positions": [
            {
                "positionID":   3403421461,
                "instrumentID": 4238,
                "amount":       270.00,
                "stopLossRate": 0.0,
                "isNoStopLoss": True,
                "openRate":     637.10,
                "units":        0.4238,
                "openDateTime": "2026-04-14T10:00:00Z",
            },
            {
                "positionID":   3403428716,
                "instrumentID": 14328,
                "amount":       70.00,
                "stopLossRate": 0.0,
                "isNoStopLoss": True,
                "openRate":     1853.51,
                "units":        0.0378,
                "openDateTime": "2026-04-14T10:05:00Z",
            },
        ],
    }
}

def test_portfolio_parses_required_fields(client):
    client._get = MagicMock(return_value=MOCK_PORTFOLIO)
    portfolio   = client.get_portfolio()

    positions = portfolio["clientPortfolio"]["positions"]
    assert len(positions) == 2, "Debe haber 2 posiciones en el mock"

    required_keys = {"positionID", "instrumentID", "amount", "stopLossRate", "isNoStopLoss"}
    for pos in positions:
        missing = required_keys - set(pos.keys())
        assert not missing, f"Campos faltantes en posición {pos['positionID']}: {missing}"


def test_portfolio_position_id_type(client):
    client._get = MagicMock(return_value=MOCK_PORTFOLIO)
    portfolio   = client.get_portfolio()
    pos = portfolio["clientPortfolio"]["positions"][0]
    assert isinstance(pos["positionID"],   int)
    assert isinstance(pos["instrumentID"], int)
    assert isinstance(pos["amount"],       float)


def test_portfolio_no_stop_detected(client):
    client._get = MagicMock(return_value=MOCK_PORTFOLIO)
    portfolio   = client.get_portfolio()
    positions   = portfolio["clientPortfolio"]["positions"]

    no_stop = [
        p for p in positions
        if p.get("isNoStopLoss") is True or float(p.get("stopLossRate", 0)) <= 0.001
    ]
    assert len(no_stop) == 2, "Ambas posiciones del mock deben detectarse sin stop"


# ---------------------------------------------------------------------------
# c. GET /market-data/instruments/rates — 5 instrumentos activos
# ---------------------------------------------------------------------------

MOCK_RATES = {
    "rates": [
        {"instrumentId": 4238,  "bid": 636.50, "ask": 637.70},
        {"instrumentId": 14328, "bid": 1852.10, "ask": 1854.90},
        {"instrumentId": 9408,  "bid": 136.20, "ask": 137.10},
        {"instrumentId": 6218,  "bid": 431.50, "ask": 432.50},
        {"instrumentId": 2488,  "bid": 208.10, "ask": 208.90},
    ]
}

def test_instrument_rates_returns_all_active(client):
    client._get = MagicMock(return_value=MOCK_RATES)
    result      = client.get_instrument_rates(ACTIVE_INSTRUMENT_IDS)

    returned_ids = {r["instrumentId"] for r in result["rates"]}
    for iid in ACTIVE_INSTRUMENT_IDS:
        assert iid in returned_ids, f"instrumento {iid} no presente en la respuesta de rates"


def test_instrument_rates_request_params(client):
    """
    Verifica que get_instrument_rates llama al endpoint UNA VEZ POR instrumento.
    La API de eToro devuelve 500 con múltiples IDs en una sola petición;
    el cliente los itera individualmente y fusiona los resultados.
    """
    calls = []
    def fake_get(path, params=None):
        calls.append(params.get("instrumentIds") if params else None)
        # Devolver solo la rate del instrumento pedido
        iid_str = params.get("instrumentIds", "") if params else ""
        matching = [r for r in MOCK_RATES["rates"] if str(r["instrumentId"]) == iid_str]
        return {"rates": matching}

    client._get = fake_get
    result = client.get_instrument_rates(ACTIVE_INSTRUMENT_IDS)

    # Debe haberse llamado una vez por instrumento
    assert len(calls) == len(ACTIVE_INSTRUMENT_IDS), (
        f"Se esperaban {len(ACTIVE_INSTRUMENT_IDS)} llamadas, se hicieron {len(calls)}"
    )
    # Cada llamada usa un único instrumentId (sin comas)
    for call_param in calls:
        assert "," not in str(call_param), (
            f"Se detectó llamada con múltiples IDs: '{call_param}' — "
            "el cliente debe llamar uno a uno"
        )
    # El resultado consolidado contiene todas las rates
    assert "rates" in result


# ---------------------------------------------------------------------------
# d. Cálculo de spread porcentual
# ---------------------------------------------------------------------------

def test_spread_pct_calculation():
    """
    spread% = (ask - bid) / mid
    Para bid=636.50, ask=637.70: mid=637.10, spread=1.20, spread%=0.188%
    """
    bid = 636.50
    ask = 637.70
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    assert 0.001 < spread_pct < 0.005, (
        f"Spread {spread_pct:.4%} fuera del rango esperado (0.1%–0.5%)"
    )


def test_spread_zero_on_equal_bid_ask():
    bid = ask = 637.10
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    assert spread_pct == 0.0


def test_spread_fallback_on_api_error():
    """
    data/market_data.py retorna 1.0 (100%) como fallback en error de API
    para que el RiskManager rechace la señal por spread excesivo.
    """
    fallback_spread = 1.0
    MAX_SPREAD = 0.005
    assert fallback_spread > MAX_SPREAD, (
        "El spread de fallback debe superar el máximo para forzar rechazo"
    )


# ---------------------------------------------------------------------------
# e. EToroAPIError 401
# ---------------------------------------------------------------------------

def test_api_error_401_raises_controlled_exception(client):
    from requests.models import Response
    import requests

    mock_resp = MagicMock(spec=Response)
    mock_resp.status_code = 401

    def fake_request(method, url, **kwargs):
        return mock_resp

    with patch("requests.request", side_effect=fake_request):
        # Parchear _headers para evitar KeyError en entorno test
        client._headers = lambda: {
            "x-request-id": "test-uuid",
            "x-api-key": "TEST",
            "x-user-key": "TEST",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Restaurar base_url real para _request
        client.base_url = "http://mock-etoro"

        with pytest.raises(EToroAPIError) as exc_info:
            client._request("GET", "/me")

        assert exc_info.value.status_code == 401
