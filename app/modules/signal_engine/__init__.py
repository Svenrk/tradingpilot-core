from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np

from app.core.module import BaseModule, PipelineContext, PipelineStep
from app.models import Signal
from app.money import quantize


@dataclass(slots=True)
class SignalDecision:
    action: Literal["BUY", "SELL", "HOLD"]
    rationale: str
    strength: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None


def ema(values: list[Decimal], period: int) -> float:
    data = np.array([float(value) for value in values], dtype=float)
    alpha = 2 / (period + 1)
    decay = (1 - alpha) ** np.arange(len(data) - 1, -1, -1, dtype=float)
    weights = alpha * decay
    weights[0] = decay[0]
    return float(np.dot(weights, data))


def rsi(values: list[Decimal], period: int = 14) -> float:
    data = np.array([float(value) for value in values], dtype=float)
    diffs = np.diff(data)
    if len(diffs) == 0:
        return 50.0
    gains = np.clip(diffs, 0, None)
    losses = np.clip(-diffs, 0, None)
    avg_gain = gains[-period:].mean() if gains.size else 0.0
    avg_loss = losses[-period:].mean() if losses.size else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values: list[Decimal]) -> float:
    return ema(values, 12) - ema(values, 26)


def atr(values: list[Decimal], period: int = 14) -> Decimal:
    data = np.array([float(value) for value in values], dtype=float)
    diffs = np.abs(np.diff(data))
    average = diffs[-period:].mean() if diffs.size else 0.5
    return quantize(max(average, 0.5))


def evaluate(snapshot) -> SignalDecision:
    closes = snapshot.closes or [snapshot.price]
    fast = ema(closes[-20:], min(9, len(closes)))
    slow = ema(closes[-50:], min(21, len(closes)))
    momentum = macd(closes)
    oscillator = rsi(closes)
    current_price = quantize(snapshot.price)
    band = atr(closes)
    lookback_price = closes[-10] if len(closes) >= 10 else closes[0]
    trend_delta = current_price - lookback_price
    if fast > slow and momentum >= 0 and oscillator >= 50 and trend_delta > band:
        stop_loss = quantize(current_price - band)
        take_profit = quantize(current_price + (band * Decimal("2")))
        return SignalDecision(
            "BUY", "Bullish crossover confirmed", Decimal("0.80"), stop_loss, take_profit
        )
    if fast < slow and momentum <= 0 and oscillator <= 50 and trend_delta < (Decimal("0") - band):
        stop_loss = quantize(current_price + band)
        take_profit = quantize(current_price - (band * Decimal("2")))
        return SignalDecision(
            "SELL", "Bearish crossover confirmed", Decimal("0.80"), stop_loss, take_profit
        )
    return SignalDecision("HOLD", "No aligned edge", Decimal("0.20"))


async def evaluate_signal_step(ctx: PipelineContext) -> None:
    ctx.signal = evaluate(ctx.snapshot)
    signal = Signal(
        symbol=ctx.snapshot.symbol,
        timeframe=ctx.snapshot.timeframe,
        action=ctx.signal.action,
        strength=ctx.signal.strength,
        stop_loss=ctx.signal.stop_loss,
        take_profit=ctx.signal.take_profit,
        rationale=ctx.signal.rationale,
    )
    ctx.session.add(signal)
    await ctx.session.flush()
    ctx.metadata["signal_id"] = signal.id


class SignalEngineModule(BaseModule):
    name = "signal_engine"
    depends_on = ("market_data",)

    def models(self):
        return [Signal]

    def pipeline_steps(self):
        return [PipelineStep(name="signal_evaluate", order=20, handler=evaluate_signal_step)]


module = SignalEngineModule()
