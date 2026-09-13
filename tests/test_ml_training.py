from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from app.modules.ml_strategy.features import BarArrays
from app.modules.ml_strategy.labels import LabelConfig
from app.modules.ml_strategy.model import StrategyModel, model_path
from app.modules.ml_strategy.training import (
    InsufficientDataError,
    TrainingConfig,
    purged_walk_forward_splits,
    train_strategy,
)


def regime_arrays(n: int = 1500, seed: int = 0) -> BarArrays:
    """Momentum regimes with persistent drift: recent returns predict the next move."""
    rng = np.random.default_rng(seed)
    drift = np.empty(n)
    state = 1
    for index in range(n):
        if rng.random() < 0.02:
            state = -state
        drift[index] = state * 0.0015
    ret = drift + rng.normal(0, 0.004, n)
    close = 100 * np.exp(np.cumsum(ret))
    opn = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.lognormal(5, 0.5, n)
    return BarArrays(np.arange(n, dtype=np.int64) * 3_600_000, opn, high, low, close, volume)


FAST_CONFIG = TrainingConfig(
    label=LabelConfig(horizon=8),
    n_folds=3,
    min_train_size=300,
    num_boost_round=200,
    early_stopping_rounds=20,
    min_trades_for_threshold=10,
)


def test_purged_splits_exclude_overlapping_labels() -> None:
    n = 200
    exit_index = np.arange(n) + 10
    splits = list(
        purged_walk_forward_splits(n, exit_index, n_folds=2, embargo=5, min_train_size=100)
    )
    assert len(splits) == 2
    for train_idx, test_idx in splits:
        test_start, test_end = test_idx[0], test_idx[-1]
        before = train_idx[train_idx < test_start]
        after = train_idx[train_idx > test_end]
        assert (exit_index[before] < test_start).all()
        assert (after > test_end + 5).all()
        assert not np.intersect1d(train_idx, test_idx).size


def test_purged_splits_require_enough_rows() -> None:
    with pytest.raises(InsufficientDataError):
        list(
            purged_walk_forward_splits(50, np.arange(50), n_folds=3, embargo=1, min_train_size=100)
        )


def test_train_strategy_learns_regimes_and_roundtrips(tmp_path: Path) -> None:
    arrays = regime_arrays()
    model, report = train_strategy(arrays, symbol="test", timeframe="1h", config=FAST_CONFIG)

    assert report.n_samples > 1000
    assert len(report.folds) == 3
    assert report.oos_accuracy > 0.5
    assert report.oos_backtest.trades >= 10
    assert sum(report.class_balance.values()) == pytest.approx(1.0)
    assert sum(report.feature_importance.values()) == pytest.approx(1.0, abs=1e-3)
    assert model.metadata.symbol == "TEST"
    assert model.metadata.thresholds == report.thresholds

    path = model_path(tmp_path, "TEST", "1h")
    model.save(path)
    loaded = StrategyModel.load(path)
    assert loaded.metadata.to_dict() == model.metadata.to_dict()

    prediction = model.predict_latest(arrays)
    reloaded_prediction = loaded.predict_latest(arrays)
    assert prediction is not None and reloaded_prediction is not None
    np.testing.assert_allclose(prediction.probabilities, reloaded_prediction.probabilities)
    assert prediction.probabilities.sum() == pytest.approx(1.0)


def test_decide_sets_barriers_consistent_with_labels() -> None:
    arrays = regime_arrays(n=900)
    model, _ = train_strategy(arrays, symbol="TEST", timeframe="1h", config=FAST_CONFIG)
    prediction = model.predict_latest(arrays)
    assert prediction is not None
    price = Decimal("100")
    for position in (1, -1):
        prediction.position = position
        decision = model.decide(prediction, price)
        assert decision.action == ("BUY" if position == 1 else "SELL")
        assert decision.stop_loss is not None and decision.take_profit is not None
        risk = abs(price - decision.stop_loss)
        reward = abs(decision.take_profit - price)
        tp_mult = model.metadata.label_config.take_profit_mult
        sl_mult = model.metadata.label_config.stop_loss_mult
        assert float(reward / risk) == pytest.approx(tp_mult / sl_mult, rel=0.05)
    prediction.position = 0
    assert model.decide(prediction, price).action == "HOLD"
    assert model.decide(None, price).action == "HOLD"


def test_predict_latest_requires_warmup() -> None:
    arrays = regime_arrays(n=900)
    model, _ = train_strategy(arrays, symbol="TEST", timeframe="1h", config=FAST_CONFIG)
    short = BarArrays(
        *(getattr(arrays, f)[:20] for f in ("open_time", "open", "high", "low", "close", "volume"))
    )
    assert model.predict_latest(short) is None
