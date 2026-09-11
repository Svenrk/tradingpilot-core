from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.events.bus import MemoryEventBus
from app.core.module import AppContext, PipelineContext, PipelineStep
from app.db import get_session_factory, init_db
from app.models import Signal, TradingViewEvent
from workers.pipeline_worker import process_pipeline_message


class DummyRegistry:
    def __init__(self, steps: list[PipelineStep]) -> None:
        self._steps = steps

    def pipeline_steps(self) -> list[PipelineStep]:
        return self._steps


@pytest.mark.asyncio
async def test_process_pipeline_message_publishes_once_per_dedupe_key(
    test_env, test_db_url: str
) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    event_bus = MemoryEventBus()
    ctx = AppContext(
        settings=get_settings(),
        session_factory=session_factory,
        event_bus=event_bus,
    )
    executions = {"count": 0}

    async def step(pipeline_ctx: PipelineContext) -> None:
        executions["count"] += 1
        pipeline_ctx.signal = type("SignalDecision", (), {"action": "BUY"})()
        pipeline_ctx.risk_decision = type("RiskDecision", (), {"approved": True})()
        pipeline_ctx.order = type("OrderRef", (), {"id": 7})()

    ctx.registry = DummyRegistry([PipelineStep(name="ok", order=10, handler=step)])
    async with session_factory() as session:
        session.add(
            TradingViewEvent(
                webhook_id="demo",
                dedupe_key="evt-1",
                symbol="BTCUSDT",
                timeframe="1m",
                side="BUY",
                price=Decimal("100"),
                payload={"symbol": "BTCUSDT", "timeframe": "1m", "price": "100"},
            )
        )
        await session.commit()

    payload = {"event_id": 1, "dedupe_key": "evt-1"}
    await process_pipeline_message(ctx, payload)
    await process_pipeline_message(ctx, payload)

    message = await event_bus.consume("ui_updates", consumer="test", group="test", timeout=0.01)
    duplicate = await event_bus.consume("ui_updates", consumer="test", group="test", timeout=0.01)
    assert executions["count"] == 1
    assert message == {
        "dedupe_key": "evt-1",
        "signal": "BUY",
        "approved": True,
        "order_id": 7,
    }
    assert duplicate is None


@pytest.mark.asyncio
async def test_process_pipeline_message_rolls_back_failed_steps(test_env, test_db_url: str) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    event_bus = MemoryEventBus()
    ctx = AppContext(
        settings=get_settings(),
        session_factory=session_factory,
        event_bus=event_bus,
    )

    async def failing_step(pipeline_ctx: PipelineContext) -> None:
        pipeline_ctx.session.add(
            Signal(
                symbol="BTCUSDT",
                timeframe="1m",
                action="BUY",
                strength=Decimal("0.8"),
                stop_loss=Decimal("99"),
                take_profit=Decimal("102"),
                rationale="test",
            )
        )
        await pipeline_ctx.session.flush()
        raise RuntimeError("boom")

    ctx.registry = DummyRegistry([PipelineStep(name="boom", order=10, handler=failing_step)])
    async with session_factory() as session:
        session.add(
            TradingViewEvent(
                webhook_id="demo",
                dedupe_key="evt-2",
                symbol="BTCUSDT",
                timeframe="1m",
                side="BUY",
                price=Decimal("100"),
                payload={"symbol": "BTCUSDT", "timeframe": "1m", "price": "100"},
            )
        )
        await session.commit()

    await process_pipeline_message(ctx, {"event_id": 1, "dedupe_key": "evt-2"})

    async with session_factory() as session:
        signal = await session.scalar(select(Signal).where(Signal.symbol == "BTCUSDT"))
    message = await event_bus.consume("ui_updates", consumer="test", group="test", timeout=0.01)
    assert signal is None
    assert message is None


@pytest.mark.asyncio
async def test_process_pipeline_message_publishes_risk_rejection_without_order(
    test_env, test_db_url: str
) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    event_bus = MemoryEventBus()
    ctx = AppContext(
        settings=get_settings(),
        session_factory=session_factory,
        event_bus=event_bus,
    )

    async def rejection_step(pipeline_ctx: PipelineContext) -> None:
        pipeline_ctx.session.add(
            Signal(
                symbol="BTCUSDT",
                timeframe="1m",
                action="SELL",
                strength=Decimal("0.8"),
                stop_loss=Decimal("101"),
                take_profit=Decimal("98"),
                rationale="reject",
            )
        )
        pipeline_ctx.signal = type("SignalDecision", (), {"action": "SELL"})()
        pipeline_ctx.risk_decision = type(
            "RiskDecision",
            (),
            {"approved": False, "reasons": ["Kill switch enabled"]},
        )()

    ctx.registry = DummyRegistry([PipelineStep(name="reject", order=10, handler=rejection_step)])
    async with session_factory() as session:
        session.add(
            TradingViewEvent(
                webhook_id="demo",
                dedupe_key="evt-3",
                symbol="BTCUSDT",
                timeframe="1m",
                side="SELL",
                price=Decimal("100"),
                payload={"symbol": "BTCUSDT", "timeframe": "1m", "price": "100"},
            )
        )
        await session.commit()

    await process_pipeline_message(ctx, {"event_id": 1, "dedupe_key": "evt-3"})

    async with session_factory() as session:
        signal = await session.scalar(select(Signal).where(Signal.rationale == "reject"))
    message = await event_bus.consume("ui_updates", consumer="test", group="test", timeout=0.01)
    assert signal is not None
    assert message == {
        "dedupe_key": "evt-3",
        "signal": "SELL",
        "approved": False,
        "order_id": None,
    }
