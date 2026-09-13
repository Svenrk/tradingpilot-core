from __future__ import annotations

import numpy as np
import pytest

from app.modules.ml_strategy.backtest import (
    DecisionThresholds,
    positions_from_probabilities,
    select_thresholds,
    simulate,
)
from app.modules.ml_strategy.features import (
    FEATURE_NAMES,
    WARMUP_BARS,
    BarArrays,
    compute_features,
    wilder_rsi,
)
from app.modules.ml_strategy.labels import CLASS_DOWN, CLASS_UP, LabelConfig, triple_barrier_labels


def make_arrays(n: int = 300, seed: int = 0, drift: float = 0.0) -> BarArrays:
    rng = np.random.default_rng(seed)
    ret = drift + rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(ret))
    opn = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.lognormal(5, 0.5, n)
    return BarArrays(np.arange(n, dtype=np.int64) * 3_600_000, opn, high, low, close, volume)


def test_feature_matrix_shape_and_warmup() -> None:
    arrays = make_arrays()
    matrix, names = compute_features(arrays)
    assert names == FEATURE_NAMES
    assert matrix.shape == (len(arrays), len(FEATURE_NAMES))
    assert np.isnan(matrix[:WARMUP_BARS]).all()
    assert np.isfinite(matrix[WARMUP_BARS:]).all()


def test_features_have_no_lookahead() -> None:
    """Changing future bars must not alter any feature computed for earlier bars."""
    arrays = make_arrays()
    matrix, _ = compute_features(arrays)
    cut = 200
    mutated = make_arrays()
    for field in ("open", "high", "low", "close"):
        getattr(mutated, field)[cut:] *= 1.5
    mutated.volume[cut:] *= 10
    mutated_matrix, _ = compute_features(mutated)
    np.testing.assert_allclose(matrix[:cut], mutated_matrix[:cut], equal_nan=True)


def test_rsi_bounds_and_direction() -> None:
    rising = np.linspace(100, 120, 60)
    falling = np.linspace(120, 100, 60)
    assert wilder_rsi(rising, 14)[-1] > 90
    assert wilder_rsi(falling, 14)[-1] < 10


def test_triple_barrier_labels_uptrend_hits_upper_barrier() -> None:
    arrays = make_arrays(n=200, drift=0.01)
    result = triple_barrier_labels(arrays, LabelConfig(horizon=10))
    valid = result.valid
    assert valid.sum() > 100
    assert (result.labels[valid] == 1).mean() > 0.9
    assert (result.classes[valid][result.labels[valid] == 1] == CLASS_UP).all()
    assert (result.returns[valid][result.labels[valid] == 1] > 0).all()
    assert (result.exit_index[valid] > np.flatnonzero(valid)).all()


def test_triple_barrier_labels_downtrend_hits_lower_barrier() -> None:
    arrays = make_arrays(n=200, drift=-0.01)
    result = triple_barrier_labels(arrays, LabelConfig(horizon=10))
    valid = result.valid
    assert (result.labels[valid] == -1).mean() > 0.9
    assert (result.classes[valid][result.labels[valid] == -1] == CLASS_DOWN).all()


def test_triple_barrier_last_rows_are_unresolved() -> None:
    arrays = make_arrays(n=100)
    result = triple_barrier_labels(arrays, LabelConfig(horizon=10))
    assert not result.valid[-10:].any()


def test_positions_from_probabilities_respects_thresholds() -> None:
    probs = np.array(
        [
            [0.10, 0.10, 0.80],  # confident long
            [0.80, 0.10, 0.10],  # confident short
            [0.40, 0.20, 0.40],  # no edge
            [0.30, 0.30, 0.40],  # below min probability
        ]
    )
    positions = positions_from_probabilities(probs, DecisionThresholds(0.5, 0.1))
    assert positions.tolist() == [1, -1, 0, 0]


def test_simulate_non_overlapping_and_costs() -> None:
    probs = np.tile([0.0, 0.0, 1.0], (10, 1))
    returns = np.full(10, 0.01)
    stats = simulate(probs, returns, DecisionThresholds(0.5, 0.1), cost_bps=0.0, horizon=5)
    assert stats.trades == 2
    assert stats.total_return == pytest.approx(0.02)
    costly = simulate(probs, returns, DecisionThresholds(0.5, 0.1), cost_bps=10.0, horizon=5)
    assert costly.total_return == pytest.approx(0.02 - 2 * 0.002)
    assert costly.hit_rate == 1.0


def test_select_thresholds_prefers_confident_trades() -> None:
    rng = np.random.default_rng(1)
    n = 600
    confidence = rng.uniform(0.34, 0.95, n)
    direction = rng.choice([-1, 1], n)
    # Outcome agrees with the forecast more often when the model is confident.
    agrees = rng.random(n) < confidence
    returns = np.where(agrees, direction, -direction) * 0.01
    probs = np.zeros((n, 3))
    probs[:, 1] = (1 - confidence) / 2
    probs[direction == 1, 2] = confidence[direction == 1]
    probs[direction == 1, 0] = (1 - confidence[direction == 1]) / 2
    probs[direction == -1, 0] = confidence[direction == -1]
    probs[direction == -1, 2] = (1 - confidence[direction == -1]) / 2
    thresholds, stats = select_thresholds(
        probs, returns, horizon=1, bars_per_year=8760, cost_bps=0.0, min_trades=30
    )
    assert thresholds.min_probability > 0.5
    assert stats.trades >= 30
    assert stats.hit_rate > 0.6
