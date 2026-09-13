"""Causal feature engineering for OHLCV bars.

Every feature at row ``i`` depends only on bars ``<= i``; the no-lookahead
property is enforced by ``tests/test_ml_features.py``. All functions are pure
numpy so feature computation is identical in training and live inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12
WARMUP_BARS = 64


@dataclass(slots=True)
class BarArrays:
    open_time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.close.shape[0])

    @classmethod
    def from_bars(cls, bars) -> BarArrays:
        return cls(
            open_time=np.array([int(bar.open_time) for bar in bars], dtype=np.int64),
            open=np.array([float(bar.open) for bar in bars], dtype=float),
            high=np.array([float(bar.high) for bar in bars], dtype=float),
            low=np.array([float(bar.low) for bar in bars], dtype=float),
            close=np.array([float(bar.close) for bar in bars], dtype=float),
            volume=np.array([float(bar.volume) for bar in bars], dtype=float),
        )


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if periods < values.shape[0]:
        out[periods:] = values[:-periods] if periods else values
    return out


def _rolling_apply(values: np.ndarray, window: int, func) -> np.ndarray:
    out = np.full(values.shape[0], np.nan)
    if values.shape[0] < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1 :] = func(windows, axis=-1)
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_apply(values, window, np.nanmean)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_apply(values, window, np.nanstd)


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_apply(values, window, np.nanmax)


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_apply(values, window, np.nanmin)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for index in range(1, values.shape[0]):
        out[index] = alpha * values[index] + (1.0 - alpha) * out[index - 1]
    return out


def wilder_rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gains = np.clip(delta, 0.0, None)
    losses = np.clip(-delta, 0.0, None)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if close.shape[0] <= period:
        return np.full_like(close, 50.0)
    avg_gain[period] = gains[1 : period + 1].mean()
    avg_loss[period] = losses[1 : period + 1].mean()
    for index in range(period + 1, close.shape[0]):
        avg_gain[index] = (avg_gain[index - 1] * (period - 1) + gains[index]) / period
        avg_loss[index] = (avg_loss[index - 1] * (period - 1) + losses[index]) / period
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = 50.0
    return rsi


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = _shift(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    return ema(true_range(high, low, close), period)


def realized_volatility(close: np.ndarray, window: int) -> np.ndarray:
    """Rolling std of log returns; used to scale labels, stops and targets."""
    log_returns = np.diff(np.log(close + EPS), prepend=np.nan)
    return rolling_std(log_returns, window)


def _rolling_skew(windows: np.ndarray, axis: int) -> np.ndarray:
    mean = windows.mean(axis=axis, keepdims=True)
    std = windows.std(axis=axis, keepdims=True) + EPS
    return (((windows - mean) / std) ** 3).mean(axis=axis)


def _rolling_kurt(windows: np.ndarray, axis: int) -> np.ndarray:
    mean = windows.mean(axis=axis, keepdims=True)
    std = windows.std(axis=axis, keepdims=True) + EPS
    return (((windows - mean) / std) ** 4).mean(axis=axis) - 3.0


def _rolling_autocorr(windows: np.ndarray, axis: int) -> np.ndarray:
    del axis
    left = windows[:, :-1]
    right = windows[:, 1:]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = (left * right).sum(axis=1)
    denominator = np.sqrt((left**2).sum(axis=1) * (right**2).sum(axis=1)) + EPS
    return numerator / denominator


def compute_features(arrays: BarArrays) -> tuple[np.ndarray, list[str]]:
    """Return ``(matrix, names)``; rows before ``WARMUP_BARS`` contain NaNs."""
    close, high, low, opn = arrays.close, arrays.high, arrays.low, arrays.open
    volume = arrays.volume
    log_close = np.log(close + EPS)
    ret = np.diff(log_close, prepend=np.nan)
    features: dict[str, np.ndarray] = {}

    for lag in (1, 2, 3, 5, 10, 20):
        features[f"ret_{lag}"] = log_close - _shift(log_close, lag)

    vol_10 = rolling_std(ret, 10)
    vol_20 = rolling_std(ret, 20)
    vol_50 = rolling_std(ret, 50)
    features["vol_10"] = vol_10
    features["vol_20"] = vol_20
    features["vol_50"] = vol_50
    features["vol_ratio_10_50"] = vol_10 / (vol_50 + EPS)
    features["ret_1_z"] = ret / (vol_20 + EPS)
    features["ret_5_z"] = features["ret_5"] / (vol_20 * np.sqrt(5.0) + EPS)

    ema_9, ema_21, ema_50 = ema(close, 9), ema(close, 21), ema(close, 50)
    features["close_ema9"] = close / (ema_9 + EPS) - 1.0
    features["close_ema21"] = close / (ema_21 + EPS) - 1.0
    features["close_ema50"] = close / (ema_50 + EPS) - 1.0
    features["ema9_ema21"] = ema_9 / (ema_21 + EPS) - 1.0
    features["ema21_ema50"] = ema_21 / (ema_50 + EPS) - 1.0

    features["rsi_7"] = wilder_rsi(close, 7) / 100.0
    features["rsi_14"] = wilder_rsi(close, 14) / 100.0

    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    features["macd_norm"] = macd_line / (close + EPS)
    features["macd_hist_norm"] = (macd_line - macd_signal) / (close + EPS)

    sma_20 = rolling_mean(close, 20)
    std_20 = rolling_std(close, 20)
    features["bb_pct_b"] = (close - (sma_20 - 2 * std_20)) / (4 * std_20 + EPS)
    features["bb_width"] = (4 * std_20) / (sma_20 + EPS)

    atr_14 = atr(high, low, close, 14)
    features["atr_norm"] = atr_14 / (close + EPS)

    hi_20, lo_20 = rolling_max(high, 20), rolling_min(low, 20)
    hi_50, lo_50 = rolling_max(high, 50), rolling_min(low, 50)
    features["donchian_20"] = (close - lo_20) / (hi_20 - lo_20 + EPS)
    features["dist_high_50"] = close / (hi_50 + EPS) - 1.0
    features["dist_low_50"] = close / (lo_50 + EPS) - 1.0

    candle_range = high - low + EPS
    features["body_ratio"] = (close - opn) / candle_range
    features["upper_wick"] = (high - np.maximum(close, opn)) / candle_range
    features["lower_wick"] = (np.minimum(close, opn) - low) / candle_range
    features["gap"] = opn / (_shift(close, 1) + EPS) - 1.0

    log_volume = np.log1p(np.clip(volume, 0.0, None))
    vol_mean_20 = rolling_mean(log_volume, 20)
    vol_std_20 = rolling_std(log_volume, 20)
    features["volume_z_20"] = (log_volume - vol_mean_20) / (vol_std_20 + EPS)
    features["volume_trend"] = rolling_mean(log_volume, 5) - vol_mean_20

    features["ret_skew_20"] = _rolling_apply(ret, 20, _rolling_skew)
    features["ret_kurt_20"] = _rolling_apply(ret, 20, _rolling_kurt)
    features["ret_autocorr_20"] = _rolling_apply(ret, 21, _rolling_autocorr)
    var_1 = vol_20**2
    var_5 = rolling_std(features["ret_5"], 20) ** 2
    features["variance_ratio_5"] = var_5 / (5.0 * var_1 + EPS)
    features["up_ratio_20"] = _rolling_apply((ret > 0).astype(float), 20, np.nanmean)

    seconds = (arrays.open_time // 1000).astype(np.int64)
    hour = (seconds // 3600) % 24
    dow = (seconds // 86400 + 4) % 7  # 1970-01-01 was a Thursday
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    names = list(features)
    matrix = np.column_stack([features[name] for name in names]).astype(float)
    matrix[:WARMUP_BARS] = np.nan
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


FEATURE_NAMES: list[str] = compute_features(
    BarArrays(
        open_time=np.arange(WARMUP_BARS + 1, dtype=np.int64) * 60_000,
        open=np.ones(WARMUP_BARS + 1),
        high=np.ones(WARMUP_BARS + 1),
        low=np.ones(WARMUP_BARS + 1),
        close=np.ones(WARMUP_BARS + 1),
        volume=np.ones(WARMUP_BARS + 1),
    )
)[1]
