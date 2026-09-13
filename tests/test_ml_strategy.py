from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.core.events.bus import MemoryEventBus
from app.core.module import AppContext, PipelineContext
from app.db import get_session_factory, init_db
from app.modules.market_data import Bar, MarketSnapshot, SnapshotService
from app.modules.ml_strategy import (
    MLStrategyModule,
    ModelRegistry,
    combine_decisions,
    ml_signal_step,
    persist_bars_step,
)
from app.modules.ml_strategy.features import BarArrays
from app.modules.ml_strategy.model import model_path
from app.modules.ml_strategy.store import count_bars, load_bar_arrays, load_bars, upsert_bars
from app.modules.ml_strategy.training import train_strategy
from app.modules.signal_engine import SignalDecision
from tests.test_ml_training import FAST_CONFIG, regime_arrays


def bars_from_arrays(arrays: BarArrays) -> list[Bar]:
    return [
        Bar(
            open_time=int(arrays.open_time[i]),
            open=Decimal(str(round(arrays.open[i], 8))),
            high=Decimal(str(round(arrays.high[i], 8))),
            low=Decimal(str(round(arrays.low[i], 8))),
            close=Decimal(str(round(arrays.close[i], 8))),
            volume=Decimal(str(round(arrays.volume[i], 8))),
        )
        for i in range(len(arrays))
    ]


@pytest.fixture
async def app_ctx(test_env, test_db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TP_ML_MODEL_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    await init_db(test_db_url)
    ctx = AppContext(
        settings=get_settings(),
        session_factory=get_session_factory(test_db_url),
        event_bus=MemoryEventBus(),
    )
    await MLStrategyModule().on_startup(ctx)
    return ctx


async def test_upsert_bars_is_idempotent_and_refreshes(app_ctx: AppContext) -> None:
    bars = bars_from_arrays(regime_arrays(n=50))
    async with app_ctx.session_factory() as session:
        assert await upsert_bars(session, "BTCUSDT", "1h", bars) == 50
        await session.commit()
        assert await count_bars(session, "BTCUSDT", "1h") == 50
        refreshed = [Bar(bars[-1].open_time, *(Decimal("1"),) * 5)]
        await upsert_bars(session, "BTCUSDT", "1h", refreshed)
        await session.commit()
        assert await count_bars(session, "BTCUSDT", "1h") == 50
        loaded = await load_bars(session, "BTCUSDT", "1h", limit=2)
        assert loaded[-1].close == Decimal("1")
        assert loaded[0].open_time < loaded[-1].open_time
        arrays = await load_bar_arrays(session, "BTCUSDT", "1h")
        assert len(arrays) == 50


async def test_snapshot_service_exposes_bars_from_closes_only_provider() -> None:
    class ClosesOnly:
        async def get_closes(self, symbol, timeframe, limit=100):
            return [Decimal("1"), Decimal("2")]

    snapshot = await SnapshotService(ClosesOnly(), None).build_snapshot(
        {"symbol": "X", "timeframe": "1m", "price": "2"}
    )
    assert [bar.close for bar in snapshot.bars] == snapshot.closes


def test_combine_decisions_modes() -> None:
    rule = SignalDecision("BUY", "rule", Decimal("0.8"), Decimal("99"), Decimal("102"))
    ml_buy = SignalDecision("BUY", "ml", Decimal("0.7"), Decimal("98"), Decimal("104"))
    ml_sell = SignalDecision("SELL", "ml", Decimal("0.7"), Decimal("101"), Decimal("96"))
    assert combine_decisions("shadow", rule, ml_sell) is rule
    assert combine_decisions("primary", rule, ml_sell) is ml_sell
    confirmed = combine_decisions("confirm", rule, ml_buy)
    assert confirmed.action == "BUY"
    assert confirmed.take_profit == Decimal("104")
    assert combine_decisions("confirm", rule, ml_sell).action == "HOLD"
    assert combine_decisions("shadow", None, ml_sell) is ml_sell


async def _run_ml_step(app_ctx: AppContext, arrays: BarArrays, rule: SignalDecision):
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timeframe="1h",
        price=Decimal(str(arrays.close[-1])),
        closes=[],
        bars=bars_from_arrays(arrays),
    )
    async with app_ctx.session_factory() as session:
        ctx = PipelineContext(app=app_ctx, session=session, payload={}, snapshot=snapshot)
        ctx.signal = rule
        await persist_bars_step(ctx)
        await ml_signal_step(ctx)
        await session.commit()
        return ctx


async def test_ml_step_without_model_leaves_rule_signal(app_ctx: AppContext) -> None:
    rule = SignalDecision("BUY", "rule", Decimal("0.8"))
    ctx = await _run_ml_step(app_ctx, regime_arrays(n=120), rule)
    assert ctx.signal is rule
    assert "ml_decision" not in ctx.metadata
    async with app_ctx.session_factory() as session:
        assert await count_bars(session, "BTCUSDT", "1h") == 120


async def test_ml_step_shadow_and_primary_modes(
    app_ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = regime_arrays(n=900)
    model, _ = train_strategy(arrays, symbol="BTCUSDT", timeframe="1h", config=FAST_CONFIG)
    model.save(model_path(Path(app_ctx.settings.ml_model_dir), "BTCUSDT", "1h"))
    rule = SignalDecision("HOLD", "rule", Decimal("0.2"))

    ctx = await _run_ml_step(app_ctx, arrays, rule)
    ml_decision = ctx.metadata["ml_decision"]
    assert ctx.signal is rule  # shadow: recorded, not acted on
    assert ml_decision.rationale.startswith("ml:")
    assert set(ctx.metadata["ml_probabilities"]) == {"down", "flat", "up"}
    assert sum(ctx.metadata["ml_probabilities"].values()) == pytest.approx(1.0)

    monkeypatch.setattr(app_ctx.settings, "ml_strategy_mode", "primary")
    ctx = await _run_ml_step(app_ctx, arrays, rule)
    assert ctx.signal is ctx.metadata["ml_decision"]
    assert ctx.signal.action == ml_decision.action

    from sqlalchemy import select

    from app.models import Signal

    async with app_ctx.session_factory() as session:
        rows = (await session.scalars(select(Signal).order_by(Signal.id))).all()
    assert [row.rationale.split("]")[0] for row in rows] == ["[shadow", "[primary"]


async def test_model_registry_hot_reloads(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    assert registry.get("BTCUSDT", "1h") is None
    arrays = regime_arrays(n=900)
    model, _ = train_strategy(arrays, symbol="BTCUSDT", timeframe="1h", config=FAST_CONFIG)
    path = model_path(tmp_path, "BTCUSDT", "1h")
    model.save(path)
    loaded = registry.get("BTCUSDT", "1h")
    assert loaded is not None
    assert registry.get("btcusdt", "1h") is loaded  # cached, case-insensitive
    assert registry.load_all() == ["BTCUSDT_1h"]
    path.write_text("not json")
    import os

    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))
    assert registry.get("BTCUSDT", "1h") is None
    assert registry.describe() == []


async def test_ml_api_endpoints(client, test_db_url: str, tmp_path: Path) -> None:
    settings = get_settings()
    arrays = regime_arrays(n=900)
    model, _ = train_strategy(arrays, symbol="BTCUSDT", timeframe="1h", config=FAST_CONFIG)
    model.save(model_path(Path(settings.ml_model_dir), "BTCUSDT", "1h"))
    async with get_session_factory(test_db_url)() as session:
        await upsert_bars(session, "BTCUSDT", "1h", bars_from_arrays(arrays))
        await session.commit()

    assert (await client.get("/api/v1/ml/models")).status_code == 401
    login = await client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "test-password"}
    )
    assert login.status_code == 200
    models = await client.get("/api/v1/ml/models")
    assert models.status_code == 200
    assert [entry["key"] for entry in models.json()] == ["BTCUSDT_1h"]
    assert "oos_backtest" in models.json()[0]["metrics"]

    prediction = await client.get("/api/v1/ml/predict/btcusdt/1h")
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["action"] in {"BUY", "SELL", "HOLD"}
    assert body["bars_used"] == min(900, settings.ml_inference_bars)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)

    missing = await client.get("/api/v1/ml/predict/ETHUSDT/1h")
    assert missing.status_code == 404

    reload = await client.post(
        "/api/v1/ml/models/reload", headers={"X-CSRF-Token": login.json()["csrf_token"]}
    )
    assert reload.status_code == 200
    assert len(reload.json()) == 1


def test_bar_arrays_roundtrip() -> None:
    arrays = regime_arrays(n=10)
    rebuilt = BarArrays.from_bars(bars_from_arrays(arrays))
    np.testing.assert_allclose(rebuilt.close, arrays.close)
    assert rebuilt.open_time.tolist() == arrays.open_time.tolist()
