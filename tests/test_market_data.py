from __future__ import annotations

from decimal import Decimal

import pytest

from app.modules.market_data import SnapshotService


class FailingProvider:
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        del symbol, timeframe, limit
        raise RuntimeError("primary unavailable")


class StaticProvider:
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        del symbol, timeframe, limit
        return [Decimal("100"), Decimal("101"), Decimal("102")]


@pytest.mark.asyncio
async def test_snapshot_service_uses_fallback_provider() -> None:
    service = SnapshotService(FailingProvider(), StaticProvider())
    snapshot = await service.build_snapshot(
        {"symbol": "BTCUSDT", "timeframe": "1m", "price": "102"}
    )
    assert snapshot.closes == [
        Decimal("100.00000000"),
        Decimal("101.00000000"),
        Decimal("102.00000000"),
    ]
    assert snapshot.price == Decimal("102.00000000")
