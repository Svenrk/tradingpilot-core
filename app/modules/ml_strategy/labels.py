"""Triple-barrier labelling (López de Prado, *Advances in Financial ML*, ch. 3).

For each bar we project a volatility-scaled profit-taking barrier above,
a stop-loss barrier below, and a vertical time barrier ``horizon`` bars
ahead. The label is the first barrier touched: ``+1`` (up), ``-1`` (down)
or ``0`` (timed out). Barriers are evaluated on subsequent bars' highs and
lows so intrabar touches count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.modules.ml_strategy.features import EPS, BarArrays, realized_volatility

CLASS_DOWN, CLASS_FLAT, CLASS_UP = 0, 1, 2
CLASS_NAMES = ("DOWN", "FLAT", "UP")


@dataclass(slots=True, frozen=True)
class LabelConfig:
    horizon: int = 12
    take_profit_mult: float = 2.0
    stop_loss_mult: float = 1.0
    vol_window: int = 20
    min_vol: float = 1e-4

    def barrier_widths(self, vol: float) -> tuple[float, float]:
        scaled = max(float(vol), self.min_vol)
        return self.take_profit_mult * scaled, self.stop_loss_mult * scaled


@dataclass(slots=True)
class LabelResult:
    labels: np.ndarray  # int in {-1, 0, 1}
    classes: np.ndarray  # int in {0, 1, 2} for LightGBM multiclass
    returns: np.ndarray  # realised log return at the barrier touch
    exit_index: np.ndarray  # bar index at which the label resolved
    valid: np.ndarray  # rows with a resolved label


def triple_barrier_labels(arrays: BarArrays, config: LabelConfig) -> LabelResult:
    n = len(arrays)
    close, high, low = arrays.close, arrays.high, arrays.low
    vol = realized_volatility(close, config.vol_window)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.full(n, np.nan)
    exit_index = np.full(n, -1, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    log_close = np.log(close + EPS)

    for index in range(n):
        end = index + config.horizon
        if end >= n or not np.isfinite(vol[index]):
            continue
        up_width, down_width = config.barrier_widths(vol[index])
        entry = close[index]
        upper = entry * np.exp(up_width)
        lower = entry * np.exp(-down_width)
        label, touched = 0, end
        for step in range(index + 1, end + 1):
            hit_up = high[step] >= upper
            hit_down = low[step] <= lower
            if hit_down:
                # Both barriers inside one bar resolve conservatively as the stop.
                label, touched = -1, step
                break
            if hit_up:
                label, touched = 1, step
                break
        labels[index] = label
        if label == 1:
            returns[index] = up_width
        elif label == -1:
            returns[index] = -down_width
        else:
            returns[index] = log_close[end] - log_close[index]
        exit_index[index] = touched
        valid[index] = True

    classes = (labels.astype(np.int64) + 1).astype(np.int64)
    return LabelResult(
        labels=labels, classes=classes, returns=returns, exit_index=exit_index, valid=valid
    )
