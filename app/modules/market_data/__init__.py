from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

import httpx

from app.core.module import AppContext, BaseModule, PipelineContext, PipelineStep
from app.money import quantize, to_decimal


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    timeframe: str
    price: Decimal
    closes: list[Decimal]


class MarketDataProvider(Protocol):
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]: ...


class BinanceMarketDataProvider:
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        params: dict[str, str | int] = {"symbol": symbol, "interval": timeframe, "limit": limit}
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://api.binance.com/api/v3/klines", params=params)
            response.raise_for_status()
        data = response.json()
        return [quantize(row[4]) for row in data]


class SyntheticMarketDataProvider:
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        del symbol, timeframe
        start = Decimal("100")
        return [quantize(start + Decimal(index) / Decimal("10")) for index in range(limit)]


class SnapshotService:
    def __init__(self, primary: MarketDataProvider, fallback: MarketDataProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    async def build_snapshot(self, payload: dict) -> MarketSnapshot:
        symbol = payload["symbol"]
        timeframe = payload["timeframe"]
        price = quantize(payload.get("price", "0"))
        closes = await self._get_closes(symbol, timeframe)
        if not closes:
            closes = [price]
        return MarketSnapshot(
            symbol=symbol, timeframe=timeframe, price=price or closes[-1], closes=closes
        )

    async def _get_closes(self, symbol: str, timeframe: str) -> list[Decimal]:
        try:
            return await self.primary.get_closes(symbol, timeframe)
        except Exception:
            return await self.fallback.get_closes(symbol, timeframe)


async def build_snapshot_step(ctx: PipelineContext) -> None:
    service: SnapshotService = ctx.app.state["market_data_service"]
    merged_payload = dict(ctx.payload)
    if ctx.event is not None:
        merged_payload.update(ctx.event.payload)
    ctx.snapshot = await service.build_snapshot(merged_payload)
    ctx.app.state.setdefault("latest_prices", {})[ctx.snapshot.symbol] = to_decimal(
        ctx.snapshot.price
    )


class MarketDataModule(BaseModule):
    name = "market_data"

    async def on_startup(self, ctx: AppContext) -> None:
        ctx.state["market_data_service"] = SnapshotService(
            BinanceMarketDataProvider(), SyntheticMarketDataProvider()
        )

    def pipeline_steps(self):
        return [PipelineStep(name="market_snapshot", order=10, handler=build_snapshot_step)]


module = MarketDataModule()
