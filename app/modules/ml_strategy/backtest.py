"""Vectorised evaluation of probability forecasts as a trading strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.modules.ml_strategy.labels import CLASS_DOWN, CLASS_UP

BARS_PER_YEAR = {
    "1m": 525_600,
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "12h": 730,
    "1d": 365,
    "1w": 52,
}


@dataclass(slots=True, frozen=True)
class DecisionThresholds:
    """Trade only when the favoured class is confident *and* clearly ahead."""

    min_probability: float = 0.45
    min_edge: float = 0.10

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class BacktestStats:
    trades: int
    long_trades: int
    short_trades: int
    hit_rate: float
    mean_return: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    exposure: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def positions_from_probabilities(
    probabilities: np.ndarray, thresholds: DecisionThresholds
) -> np.ndarray:
    """Map class probabilities to positions in ``{-1, 0, +1}``."""
    p_down = probabilities[:, CLASS_DOWN]
    p_up = probabilities[:, CLASS_UP]
    edge = p_up - p_down
    longs = (p_up >= thresholds.min_probability) & (edge >= thresholds.min_edge)
    shorts = (p_down >= thresholds.min_probability) & (-edge >= thresholds.min_edge)
    positions = np.zeros(probabilities.shape[0], dtype=np.int8)
    positions[longs] = 1
    positions[shorts] = -1
    return positions


def simulate(
    probabilities: np.ndarray,
    barrier_returns: np.ndarray,
    thresholds: DecisionThresholds,
    *,
    cost_bps: float = 5.0,
    bars_per_year: int = 8_760,
    horizon: int = 12,
) -> BacktestStats:
    """Evaluate a non-overlapping position-per-signal strategy.

    Each accepted signal earns the triple-barrier return of that bar (signed by
    direction) minus round-trip costs. Signals are non-overlapping: once a trade
    is open, subsequent signals are ignored until it resolves ``horizon`` bars
    later at most, mirroring how the live pipeline holds one position per symbol.
    """
    positions = positions_from_probabilities(probabilities, thresholds)
    cost = cost_bps / 10_000.0 * 2.0
    n = positions.shape[0]
    pnl = np.zeros(n)
    taken = np.zeros(n, dtype=bool)
    next_free = 0
    for index in range(n):
        if index < next_free or positions[index] == 0 or not np.isfinite(barrier_returns[index]):
            continue
        pnl[index] = positions[index] * barrier_returns[index] - cost
        taken[index] = True
        next_free = index + horizon
    trade_pnl = pnl[taken]
    trades = int(taken.sum())
    if trades == 0:
        return BacktestStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(equity)
    per_bar_std = pnl.std()
    sharpe = float(pnl.mean() / per_bar_std * np.sqrt(bars_per_year)) if per_bar_std > 0 else 0.0
    gross_profit = trade_pnl[trade_pnl > 0].sum()
    gross_loss = -trade_pnl[trade_pnl < 0].sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    return BacktestStats(
        trades=trades,
        long_trades=int((positions[taken] == 1).sum()),
        short_trades=int((positions[taken] == -1).sum()),
        hit_rate=float((trade_pnl > 0).mean()),
        mean_return=float(trade_pnl.mean()),
        total_return=float(equity[-1]),
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        profit_factor=min(profit_factor, 1e6),
        exposure=float(trades * horizon / n),
    )


def select_thresholds(
    probabilities: np.ndarray,
    barrier_returns: np.ndarray,
    *,
    horizon: int,
    bars_per_year: int,
    cost_bps: float,
    min_trades: int = 30,
) -> tuple[DecisionThresholds, BacktestStats]:
    """Grid-search thresholds on out-of-sample forecasts, maximising Sharpe.

    Requiring ``min_trades`` guards against picking an over-tight threshold that
    happened to catch a handful of lucky trades.
    """
    best: tuple[float, DecisionThresholds, BacktestStats] | None = None
    fallback: tuple[int, DecisionThresholds, BacktestStats] | None = None
    for min_probability in np.arange(0.34, 0.76, 0.02):
        for min_edge in np.arange(0.0, 0.41, 0.05):
            thresholds = DecisionThresholds(
                round(float(min_probability), 2), round(float(min_edge), 2)
            )
            stats = simulate(
                probabilities,
                barrier_returns,
                thresholds,
                cost_bps=cost_bps,
                bars_per_year=bars_per_year,
                horizon=horizon,
            )
            if fallback is None or stats.trades > fallback[0]:
                fallback = (stats.trades, thresholds, stats)
            if stats.trades < min_trades:
                continue
            score = stats.sharpe
            if best is None or score > best[0]:
                best = (score, thresholds, stats)
    if best is not None:
        return best[1], best[2]
    assert fallback is not None
    return fallback[1], fallback[2]
