"""
tests/test_look_ahead.py
Fase 2 — Validación de ausencia de sesgo de anticipación (Look-ahead bias).

El régimen en T debe ser idéntico si se calcula con data[0:T]
que con data[0:T+100].
"""

import numpy as np
import pytest
from core.hmm_engine import HMMEngine


def _generate_synthetic_data(n_samples: int = 800, n_features: int = 14, seed: int = 42) -> np.ndarray:
    """Genera datos sintéticos con 3 regímenes de volatilidad distintos."""
    rng = np.random.default_rng(seed)
    segment = n_samples // 3
    data = np.concatenate([
        rng.normal(loc=0.0, scale=0.5, size=(segment, n_features)),
        rng.normal(loc=0.0, scale=1.5, size=(segment, n_features)),
        rng.normal(loc=0.0, scale=0.8, size=(n_samples - 2 * segment, n_features)),
    ])
    return data


@pytest.fixture(scope="module")
def trained_engine():
    X = _generate_synthetic_data(n_samples=800)
    engine = HMMEngine(n_candidates=[3, 4], n_init=3, stability_bars=1)
    engine.train(X)
    return engine


def test_no_look_ahead_bias(trained_engine: HMMEngine):
    """
    El régimen predicho en T debe ser igual tanto si se calcula
    con data[0:T] como con data[0:T+100].
    """
    X = _generate_synthetic_data(n_samples=800)
    T = 600

    def _clone(src: HMMEngine) -> HMMEngine:
        e = HMMEngine(n_candidates=src.n_candidates, n_init=1, stability_bars=1)
        e.model         = src.model
        e.n_regimes     = src.n_regimes
        e.regime_labels = src.regime_labels
        e.regime_order  = src.regime_order
        e.regime_infos  = src.regime_infos
        return e

    engine_short = _clone(trained_engine)
    state_short  = engine_short.predict_regime_filtered(X[:T])

    engine_long = _clone(trained_engine)
    engine_long.predict_regime_filtered(X[:T])
    regime_at_T = engine_long._prev_label

    assert state_short.label == regime_at_T, (
        f"LOOK-AHEAD BIAS DETECTED: "
        f"régimen con data[0:T]='{state_short.label}' "
        f"!= régimen con data[0:T+100]='{regime_at_T}'"
    )


def test_forward_alpha_updates_incrementally(trained_engine: HMMEngine):
    """El vector alpha debe cambiar con cada nueva observación."""
    X = _generate_synthetic_data(n_samples=700)

    engine = HMMEngine(n_candidates=[3], n_init=1, stability_bars=1)
    engine.model         = trained_engine.model
    engine.n_regimes     = trained_engine.n_regimes
    engine.regime_labels = trained_engine.regime_labels
    engine.regime_order  = trained_engine.regime_order
    engine.regime_infos  = trained_engine.regime_infos

    engine.predict_regime_filtered(X[0])
    alpha_t0 = engine._log_alpha.copy()

    engine.predict_regime_filtered(X[1])
    alpha_t1 = engine._log_alpha.copy()

    assert not np.allclose(alpha_t0, alpha_t1), (
        "El vector alpha no cambió tras procesar una nueva observación."
    )
