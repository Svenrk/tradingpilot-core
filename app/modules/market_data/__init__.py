from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

import httpx

from app.core.module import AppContext, BaseModule, PipelineContext, PipelineStep
from app.money import quantize, to_decimal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    timeframe: str
    price: Decimal
    closes: list[Decimal]


class MarketDataProvider(Protocol):
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]: ...


class MarketDataProviderError(RuntimeError):
    pass


class BinanceMarketDataProvider:
    """Fetches closes from Binance using a shared, pooled HTTP client."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        params: dict[str, str | int] = {"symbol": symbol, "interval": timeframe, "limit": limit}
        response = await self._get_client().get(
            "https://api.binance.com/api/v3/klines", params=params
        )
        response.raise_for_status()
        data = response.json()
        return [quantize(row[4]) for row in data]

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class SyntheticMarketDataProvider:
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        del symbol, timeframe
        start = Decimal("100")
        return [quantize(start + Decimal(index) / Decimal("10")) for index in range(limit)]


class SnapshotService:
    def __init__(
        self,
        primary: MarketDataProvider,
        fallback: MarketDataProvider | None,
        *,
        cache_ttl_seconds: float = 5.0,
        retries: int = 2,
        retry_backoff_seconds: float = 0.2,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cache_ttl_seconds = cache_ttl_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._cache: dict[tuple[str, str], tuple[float, list[Decimal]]] = {}

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
        cached = self._cache.get((symbol, timeframe))
        if cached is not None and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            return cached[1]
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                closes = await self.primary.get_closes(symbol, timeframe)
                self._cache[(symbol, timeframe)] = (time.monotonic(), closes)
                return closes
            except (httpx.HTTPError, MarketDataProviderError) as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
        if self.fallback is None:
            raise MarketDataProviderError(
                f"Primary market data unavailable for {symbol}/{timeframe}"
            ) from last_error
        logger.warning(
            "Primary market data unavailable for %s/%s, using fallback provider",
            symbol,
            timeframe,
        )
        return await self.fallback.get_closes(symbol, timeframe)


async def build_snapshot_step(ctx: PipelineContext) -> None:
    service: SnapshotService = ctx.app.state["market_data_service"]
    merged_payload = dict(ctx.payload)
    if ctx.event is not None:
        merged_payload.update(ctx.event.payload)
    ctx.snapshot = await service.build_snapshot(merged_payload)
    ctx.metadata["quantity"] = to_decimal(merged_payload.get("quantity", 1))
    price = to_decimal(ctx.snapshot.price)
    ctx.app.state.setdefault("latest_prices", {})[ctx.snapshot.symbol] = price
    store = ctx.app.state.get("store")
    if store is not None:
        await store.set(
            f"price:{ctx.snapshot.symbol}",
            str(price),
            ex=ctx.app.settings.latest_price_ttl_seconds,
        )


class MarketDataModule(BaseModule):
    name = "market_data"

    async def on_startup(self, ctx: AppContext) -> None:
        provider = BinanceMarketDataProvider(httpx.AsyncClient(timeout=5.0))
        ctx.state["market_data_provider"] = provider
        # Never trade live on synthetic data: only wire the synthetic fallback
        # when live trading is disabled.
        fallback = None if ctx.settings.enable_live_trading else SyntheticMarketDataProvider()
        ctx.state["market_data_service"] = SnapshotService(
            provider,
            fallback,
            cache_ttl_seconds=ctx.settings.market_data_cache_ttl_seconds,
        )

    async def on_shutdown(self, ctx: AppContext) -> None:
        provider = ctx.state.get("market_data_provider")
        if provider is not None:
            await provider.close()

    def pipeline_steps(self):
        return [PipelineStep(name="market_snapshot", order=10, handler=build_snapshot_step)]


module = MarketDataModule()
