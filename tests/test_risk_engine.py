from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.config import get_settings
from app.core.events.bus import MemoryEventBus
from app.core.module import AppContext, PipelineContext
from app.db import get_session_factory, init_db
from app.models import Position
from app.modules.market_data import MarketSnapshot
from app.modules.risk_engine import RiskService
from app.modules.settings import RuntimeSettings, SettingsService
from app.security import MemoryStore


@pytest.mark.asyncio
async def test_risk_engine_rejects_when_kill_switch_enabled(test_env, test_db_url: str) -> None:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    store = MemoryStore()
    settings_service = SettingsService(store, get_settings())
    await settings_service.update_runtime_settings(
        RuntimeSettings(
            kill_switch=True,
            mode="paper",
            max_positions=5,
            daily_loss_limit="1000",
            min_reward_risk="1.5",
            exposure_cap="100000",
        )
    )
    ctx = AppContext(
        settings=get_settings(), session_factory=session_factory, event_bus=MemoryEventBus()
    )
    ctx.state["settings_service"] = settings_service
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
        decision = await RiskService().evaluate(pipeline_ctx)
    assert not decision.approved
    assert "Kill switch enabled" in decision.reasons


async def _build_risk_context(
    test_db_url: str, runtime: RuntimeSettings
) -> tuple[AppContext, object]:
    await init_db(test_db_url)
    session_factory = get_session_factory(test_db_url)
    store = MemoryStore()
    settings_service = SettingsService(store, get_settings())
    await settings_service.update_runtime_settings(runtime)
    ctx = AppContext(
        settings=get_settings(),
        session_factory=session_factory,
        event_bus=MemoryEventBus(),
    )
    ctx.state["settings_service"] = settings_service
    return ctx, session_factory


@pytest.mark.asyncio
async def test_risk_engine_rejects_low_reward_to_risk(test_env, test_db_url: str) -> None:
    ctx, session_factory = await _build_risk_context(
        test_db_url,
        RuntimeSettings(min_reward_risk=Decimal("3")),
    )
    async with session_factory() as session:
        pipeline_ctx = PipelineContext(app=ctx, session=session, payload={"dedupe_key": "x"})
        pipeline_ctx.snapshot = MarketSnapshot(
            "BTCUSDT", "1m", Decimal("100"), [Decimal("99"), Decimal("100")]
        )
        pipeline_ctx.signal = type(
            "Signal",
            (),
            {"action": "BUY", "stop_loss": Decimal("99"), "take_profit": Decimal("101")},
        )()
        decision = await RiskService().evaluate(pipeline_ctx)
    assert not decision.approved
    assert "Reward-to-risk below minimum" in decision.reasons


@pytest.mark.asyncio
async def test_risk_engine_rejects_max_positions(test_env, test_db_url: str) -> None:
    ctx, session_factory = await _build_risk_context(
        test_db_url,
        RuntimeSettings(max_positions=1),
    )
    async with session_factory() as session:
        session.add(
            Position(
                symbol="ETHUSDT",
                side="BUY",
                quantity=Decimal("1"),
                entry_price=Decimal("100"),
                status="OPEN",
                realized_pnl=Decimal("0"),
            )
        )
        await session.commit()
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
        decision = await RiskService().evaluate(pipeline_ctx)
    assert not decision.approved
    assert "Max positions reached" in decision.reasons


@pytest.mark.asyncio
async def test_risk_engine_rejects_daily_loss_breach(test_env, test_db_url: str) -> None:
    ctx, session_factory = await _build_risk_context(
        test_db_url,
        RuntimeSettings(daily_loss_limit=Decimal("500")),
    )
    async with session_factory() as session:
        session.add(
            Position(
                symbol="ETHUSDT",
                side="BUY",
                quantity=Decimal("1"),
                entry_price=Decimal("100"),
                status="CLOSED",
                realized_pnl=Decimal("-600"),
                closed_at=datetime.now(UTC),
            )
        )
        await session.commit()
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
        decision = await RiskService().evaluate(pipeline_ctx)
    assert not decision.approved
    assert "Daily loss limit breached" in decision.reasons


@pytest.mark.asyncio
async def test_risk_engine_rejects_exposure_cap(test_env, test_db_url: str) -> None:
    ctx, session_factory = await _build_risk_context(
        test_db_url,
        RuntimeSettings(exposure_cap=Decimal("150")),
    )
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
        decision = await RiskService().evaluate(pipeline_ctx)
    assert not decision.approved
    assert "Exposure cap exceeded" in decision.reasons
