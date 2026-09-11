from decimal import Decimal

from app.modules.market_data import MarketSnapshot
from app.modules.signal_engine import evaluate


def test_signal_engine_generates_buy_signal_for_uptrend() -> None:
    closes = [Decimal("100") + Decimal(index) for index in range(40)]
    snapshot = MarketSnapshot(symbol="BTCUSDT", timeframe="1m", price=closes[-1], closes=closes)
    signal = evaluate(snapshot)
    assert signal.action == "BUY"
    assert signal.stop_loss is not None
    assert signal.take_profit is not None


def test_signal_engine_can_hold_when_rangebound() -> None:
    closes = [Decimal("100"), Decimal("101"), Decimal("100.5"), Decimal("100.9")] * 10
    snapshot = MarketSnapshot(
        symbol="BTCUSDT", timeframe="1m", price=Decimal("100.9"), closes=closes
    )
    signal = evaluate(snapshot)
    assert signal.action in {"HOLD", "SELL"}
