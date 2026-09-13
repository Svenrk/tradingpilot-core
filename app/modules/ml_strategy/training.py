"""Walk-forward training with purging and embargo.

Financial labels overlap in time (a triple-barrier label at bar *t* is only
known at bar *t + h*), so a naive K-fold leaks future information into the
training set. We therefore:

* split chronologically into expanding-window folds,
* **purge** training rows whose label window overlaps the test fold,
* **embargo** a further ``embargo`` bars after the test fold,
* collect out-of-sample probabilities from every fold and select trade
  thresholds only on those, then
* refit a final model on all data with the median early-stopped round count.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from app.modules.ml_strategy.backtest import (
    BARS_PER_YEAR,
    BacktestStats,
    DecisionThresholds,
    select_thresholds,
    simulate,
)
from app.modules.ml_strategy.features import BarArrays, compute_features
from app.modules.ml_strategy.labels import (
    CLASS_NAMES,
    LabelConfig,
    LabelResult,
    triple_barrier_labels,
)
from app.modules.ml_strategy.model import ModelMetadata, StrategyModel, utc_now_iso

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(slots=True)
class TrainingConfig:
    label: LabelConfig = field(default_factory=LabelConfig)
    n_folds: int = 5
    embargo: int | None = None  # defaults to the label horizon
    min_train_size: int = 300
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    cost_bps: float = 5.0
    min_trades_for_threshold: int = 30
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))

    @property
    def effective_embargo(self) -> int:
        return self.label.horizon if self.embargo is None else self.embargo


@dataclass(slots=True)
class FoldReport:
    fold: int
    train_rows: int
    test_rows: int
    best_iteration: int
    log_loss: float
    accuracy: float


@dataclass(slots=True)
class TrainingReport:
    symbol: str
    timeframe: str
    n_bars: int
    n_samples: int
    class_balance: dict[str, float]
    folds: list[FoldReport]
    oos_log_loss: float
    oos_accuracy: float
    oos_balanced_accuracy: float
    thresholds: DecisionThresholds
    oos_backtest: BacktestStats
    baseline_backtest: BacktestStats
    feature_importance: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["thresholds"] = self.thresholds.to_dict()
        data["oos_backtest"] = self.oos_backtest.to_dict()
        data["baseline_backtest"] = self.baseline_backtest.to_dict()
        return data


class InsufficientDataError(ValueError):
    pass


def purged_walk_forward_splits(
    n: int, exit_index: np.ndarray, *, n_folds: int, embargo: int, min_train_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` over expanding chronological windows."""
    if n < min_train_size + n_folds:
        raise InsufficientDataError(
            f"Need at least {min_train_size + n_folds} labelled rows, have {n}"
        )
    test_region = np.arange(min_train_size, n)
    for test_idx in np.array_split(test_region, n_folds):
        if test_idx.size == 0:
            continue
        test_start, test_end = int(test_idx[0]), int(test_idx[-1])
        candidates = np.arange(0, test_start)
        # Purge: drop training rows whose label resolves inside the test window.
        train_idx = candidates[exit_index[candidates] < test_start]
        # Embargo: also allow later rows once they are clear of the test fold.
        after = np.arange(min(test_end + 1 + embargo, n), n)
        train_idx = np.concatenate([train_idx, after]) if after.size else train_idx
        yield train_idx, test_idx


def _log_loss(probabilities: np.ndarray, classes: np.ndarray) -> float:
    clipped = np.clip(probabilities[np.arange(classes.shape[0]), classes], 1e-12, 1.0)
    return float(-np.log(clipped).mean())


def _balanced_accuracy(predicted: np.ndarray, classes: np.ndarray) -> float:
    recalls = []
    for klass in np.unique(classes):
        mask = classes == klass
        recalls.append(float((predicted[mask] == klass).mean()))
    return float(np.mean(recalls))


def _class_weights(classes: np.ndarray) -> np.ndarray:
    counts = np.bincount(classes, minlength=3).astype(float)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    return weights[classes]


def _sample_weights(classes: np.ndarray, returns: np.ndarray) -> np.ndarray:
    # Balance classes and up-weight decisive moves so the model learns the
    # trades that matter for P&L rather than the abundant small ones.
    magnitude = np.abs(returns)
    scaled = 0.5 + magnitude / (np.nanmedian(magnitude) + 1e-12)
    return _class_weights(classes) * np.clip(scaled, 0.5, 3.0)


def build_dataset(arrays: BarArrays, label_config: LabelConfig):
    matrix, names = compute_features(arrays)
    labels: LabelResult = triple_barrier_labels(arrays, label_config)
    valid = labels.valid & np.isfinite(matrix).all(axis=1)
    return matrix, names, labels, valid


def _fit(lightgbm, params, x_train, y_train, w_train, config, x_valid=None, y_valid=None):
    train_set = lightgbm.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=False)
    callbacks = [lightgbm.log_evaluation(period=0)]
    valid_sets = []
    if x_valid is not None:
        valid_sets = [lightgbm.Dataset(x_valid, label=y_valid, reference=train_set)]
        callbacks.append(lightgbm.early_stopping(config.early_stopping_rounds, verbose=False))
    return lightgbm.train(
        params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=valid_sets or None,
        callbacks=callbacks,
    )


def train_strategy(
    arrays: BarArrays, *, symbol: str, timeframe: str, config: TrainingConfig | None = None
) -> tuple[StrategyModel, TrainingReport]:
    from app.modules.ml_strategy.model import _require_lightgbm

    lightgbm = _require_lightgbm()
    config = config or TrainingConfig()
    matrix, names, labels, valid = build_dataset(arrays, config.label)
    x_all = matrix[valid]
    y_all = labels.classes[valid]
    r_all = labels.returns[valid]
    exit_all = labels.exit_index[valid]
    # Map bar-space exit indices into the compacted sample space for purging:
    # the position of the first retained row at or after the label's exit bar.
    exit_compact = np.searchsorted(np.flatnonzero(valid), exit_all)
    n = x_all.shape[0]
    weights = _sample_weights(y_all, r_all)

    oos_probs = np.full((n, 3), np.nan)
    folds: list[FoldReport] = []
    best_rounds: list[int] = []
    params = dict(config.lgbm_params)
    for fold, (train_idx, test_idx) in enumerate(
        purged_walk_forward_splits(
            n,
            exit_compact,
            n_folds=config.n_folds,
            embargo=config.effective_embargo,
            min_train_size=config.min_train_size,
        )
    ):
        if train_idx.size < config.min_train_size // 2:
            continue
        # Inner tail of the training window drives early stopping.
        cut = int(train_idx.size * 0.85)
        inner_train, inner_valid = train_idx[:cut], train_idx[cut:]
        booster = _fit(
            lightgbm,
            params,
            x_all[inner_train],
            y_all[inner_train],
            weights[inner_train],
            config,
            x_all[inner_valid],
            y_all[inner_valid],
        )
        best_iteration = max(int(booster.best_iteration or booster.current_iteration()), 1)
        probs = np.asarray(booster.predict(x_all[test_idx], num_iteration=best_iteration))
        oos_probs[test_idx] = probs
        predicted = probs.argmax(axis=1)
        folds.append(
            FoldReport(
                fold=fold,
                train_rows=int(train_idx.size),
                test_rows=int(test_idx.size),
                best_iteration=best_iteration,
                log_loss=_log_loss(probs, y_all[test_idx]),
                accuracy=float((predicted == y_all[test_idx]).mean()),
            )
        )
        best_rounds.append(best_iteration)
        logger.info(
            "Fold %s: logloss=%.4f acc=%.3f iters=%s",
            fold,
            folds[-1].log_loss,
            folds[-1].accuracy,
            best_iteration,
        )

    if not folds:
        raise InsufficientDataError("No walk-forward folds could be evaluated")

    scored = np.isfinite(oos_probs).all(axis=1)
    probs_oos, y_oos, r_oos = oos_probs[scored], y_all[scored], r_all[scored]
    bars_per_year = BARS_PER_YEAR.get(timeframe, 8_760)
    thresholds, oos_backtest = select_thresholds(
        probs_oos,
        r_oos,
        horizon=config.label.horizon,
        bars_per_year=bars_per_year,
        cost_bps=config.cost_bps,
        min_trades=config.min_trades_for_threshold,
    )
    # Baseline: always trade the argmax direction, to show what thresholding adds.
    baseline = simulate(
        probs_oos,
        r_oos,
        DecisionThresholds(0.0, 0.0),
        cost_bps=config.cost_bps,
        bars_per_year=bars_per_year,
        horizon=config.label.horizon,
    )

    final_rounds = int(np.median(best_rounds))
    final_config = replace(config, num_boost_round=final_rounds)
    booster = _fit(lightgbm, params, x_all, y_all, weights, final_config)
    gain = booster.feature_importance(importance_type="gain")
    total_gain = float(gain.sum()) or 1.0
    importance = {
        name: round(float(value) / total_gain, 6)
        for name, value in sorted(zip(names, gain, strict=True), key=lambda item: -item[1])
    }
    predicted_oos = probs_oos.argmax(axis=1)
    counts = np.bincount(y_all, minlength=3)
    report = TrainingReport(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=len(arrays),
        n_samples=int(n),
        class_balance={name: float(counts[i] / n) for i, name in enumerate(CLASS_NAMES)},
        folds=folds,
        oos_log_loss=_log_loss(probs_oos, y_oos),
        oos_accuracy=float((predicted_oos == y_oos).mean()),
        oos_balanced_accuracy=_balanced_accuracy(predicted_oos, y_oos),
        thresholds=thresholds,
        oos_backtest=oos_backtest,
        baseline_backtest=baseline,
        feature_importance=importance,
    )
    metadata = ModelMetadata(
        symbol=symbol.upper(),
        timeframe=timeframe,
        feature_names=names,
        label_config=config.label,
        thresholds=thresholds,
        trained_at=utc_now_iso(),
        n_samples=int(n),
        metrics={
            "oos_log_loss": report.oos_log_loss,
            "oos_accuracy": report.oos_accuracy,
            "oos_balanced_accuracy": report.oos_balanced_accuracy,
            "oos_backtest": oos_backtest.to_dict(),
            "baseline_backtest": baseline.to_dict(),
            "folds": [asdict(fold) for fold in folds],
            "final_rounds": final_rounds,
        },
        feature_importance=importance,
    )
    return StrategyModel(booster, metadata), report
