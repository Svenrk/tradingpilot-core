from decimal import Decimal

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.events.bus import MemoryEventBus
from app.core.module import AppContext, PipelineContext
from app.db import get_session_factory, init_db
from app.models import Position
from app.modules.execution import ExecutionService, PaperBroker
from app.modules.market_data import MarketSnapshot
from app.modules.settings import SettingsService
from app.security import MemoryStore


@pytest.mark.asyncio
async def test_execution_requires_risk_approval(test_env, test_db_url: str) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    ctx = AppContext(
        settings=get_settings(), session_factory=session_factory, event_bus=MemoryEventBus()
    )
    ctx.state["settings_service"] = SettingsService(MemoryStore(), get_settings())
    async with session_factory() as session:
        pipeline_ctx = PipelineContext(app=ctx, session=session, payload={"dedupe_key": "x"})
        pipeline_ctx.snapshot = MarketSnapshot(
            "BTCUSDT", "1m", Decimal("100"), [Decimal("99"), Decimal("100")]
        )
        pipeline_ctx.signal = type(
            "Signal",
            (),
            {"action": "BUY", "stop_loss": Decimal("99"), "take_profit": Decimal("102")},
        )()
        pipeline_ctx.risk_decision = type("RiskDecision", (), {"approved": False})()
        with pytest.raises(RuntimeError, match="approved risk decision"):
            await ExecutionService(PaperBroker()).execute(pipeline_ctx)


@pytest.mark.asyncio
async def test_paper_broker_execution_creates_position(test_env, test_db_url: str) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    ctx = AppContext(
        settings=get_settings(), session_factory=session_factory, event_bus=MemoryEventBus()
    )
    ctx.state["settings_service"] = SettingsService(MemoryStore(), get_settings())
    async with session_factory() as session:
        pipeline_ctx = PipelineContext(app=ctx, session=session, payload={"dedupe_key": "x"})
        pipeline_ctx.snapshot = MarketSnapshot(
            "BTCUSDT", "1m", Decimal("100"), [Decimal("99"), Decimal("100")]
        )
        pipeline_ctx.metadata["quantity"] = Decimal("2")
        pipeline_ctx.signal = type(
            "Signal",
            (),
            {"action": "BUY", "stop_loss": Decimal("99"), "take_profit": Decimal("102")},
        )()
        pipeline_ctx.risk_decision = type("RiskDecision", (), {"approved": True})()
        order, position = await ExecutionService(PaperBroker()).execute(pipeline_ctx)
        assert order.status == "FILLED"
        assert order.quantity == Decimal("2")
        assert position.symbol == "BTCUSDT"
        assert position.quantity == Decimal("2")


@pytest.mark.asyncio
async def test_execution_accumulates_existing_same_side_position(
    test_env, test_db_url: str
) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    ctx = AppContext(
        settings=get_settings(),
        session_factory=session_factory,
        event_bus=MemoryEventBus(),
    )
    ctx.state["settings_service"] = SettingsService(MemoryStore(), get_settings())
    async with session_factory() as session:
        pipeline_ctx = PipelineContext(app=ctx, session=session, payload={"dedupe_key": "x"})
        pipeline_ctx.snapshot = MarketSnapshot(
            "BTCUSDT",
            "1m",
            Decimal("100"),
            [Decimal("99"), Decimal("100")],
        )
        pipeline_ctx.signal = type(
            "Signal",
            (),
            {"action": "BUY", "stop_loss": Decimal("99"), "take_profit": Decimal("102")},
        )()
        pipeline_ctx.risk_decision = type("RiskDecision", (), {"approved": True})()
        _, position = await ExecutionService(PaperBroker()).execute(pipeline_ctx)
        _, updated_position = await ExecutionService(PaperBroker()).execute(pipeline_ctx)
        assert position.quantity == Decimal("2")
        assert updated_position.quantity == Decimal("2")


@pytest.mark.asyncio
async def test_execution_ignores_hold_signal(test_env, test_db_url: str) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    ctx = AppContext(
        settings=get_settings(),
        session_factory=session_factory,
        event_bus=MemoryEventBus(),
    )
    ctx.state["settings_service"] = SettingsService(MemoryStore(), get_settings())
    async with session_factory() as session:
        pipeline_ctx = PipelineContext(app=ctx, session=session, payload={"dedupe_key": "x"})
        pipeline_ctx.snapshot = MarketSnapshot(
            "BTCUSDT",
            "1m",
            Decimal("100"),
            [Decimal("99"), Decimal("100")],
        )
        pipeline_ctx.signal = type(
            "Signal",
            (),
            {"action": "HOLD", "stop_loss": None, "take_profit": None},
        )()
        pipeline_ctx.risk_decision = type("RiskDecision", (), {"approved": True})()
        result = await ExecutionService(PaperBroker()).execute(pipeline_ctx)
        assert result is None
        position = await session.scalar(select(Position))
        assert position is None
