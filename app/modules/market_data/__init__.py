from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

import httpx

from app.core.module import AppContext, BaseModule, PipelineContext, PipelineStep
from app.money import quantize, to_decimal
from app.observability import metrics

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Bar:
    """A single OHLCV candle. ``open_time`` is epoch milliseconds (UTC)."""

    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    synthetic: bool = False  # flat candle derived from a close; never persisted

    @classmethod
    def from_close(cls, close: Decimal, open_time: int = 0) -> Bar:
        return cls(open_time, close, close, close, close, Decimal("0"), synthetic=True)


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    timeframe: str
    price: Decimal
    closes: list[Decimal]
    bars: list[Bar] = field(default_factory=list)


class MarketDataProvider(Protocol):
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]: ...


class BarProvider(Protocol):
    async def get_bars(self, symbol: str, timeframe: str, limit: int = 100) -> list[Bar]: ...


def parse_binance_klines(data: list[list]) -> list[Bar]:
    return [
        Bar(
            open_time=int(row[0]),
            open=quantize(row[1]),
            high=quantize(row[2]),
            low=quantize(row[3]),
            close=quantize(row[4]),
            volume=quantize(row[5]),
        )
        for row in data
    ]


async def fetch_bars(
    provider: MarketDataProvider, symbol: str, timeframe: str, limit: int
) -> list[Bar]:
    """Fetch OHLCV bars, synthesising flat candles for closes-only providers."""
    getter = getattr(provider, "get_bars", None)
    if getter is not None:
        return await getter(symbol, timeframe, limit)
    closes = await provider.get_closes(symbol, timeframe, limit)
    return [Bar.from_close(close, index) for index, close in enumerate(closes)]


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

    async def get_bars(
        self, symbol: str, timeframe: str, limit: int = 100, *, end_time: int | None = None
    ) -> list[Bar]:
        params: dict[str, str | int] = {"symbol": symbol, "interval": timeframe, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        response = await self._get_client().get(
            "https://api.binance.com/api/v3/klines", params=params
        )
        response.raise_for_status()
        return parse_binance_klines(response.json())

    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        return [bar.close for bar in await self.get_bars(symbol, timeframe, limit)]

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class SyntheticMarketDataProvider:
    async def get_closes(self, symbol: str, timeframe: str, limit: int = 100) -> list[Decimal]:
        del symbol, timeframe
        start = Decimal("100")
        return [quantize(start + Decimal(index) / Decimal("10")) for index in range(limit)]

    async def get_bars(self, symbol: str, timeframe: str, limit: int = 100) -> list[Bar]:
        closes = await self.get_closes(symbol, timeframe, limit)
        step_ms = timeframe_to_ms(timeframe)
        return [Bar.from_close(close, index * step_ms) for index, close in enumerate(closes)]


_TIMEFRAME_UNITS_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def timeframe_to_ms(timeframe: str) -> int:
    """Convert a Binance-style interval (``1m``, ``4h``, ``1d``) to milliseconds."""
    unit = timeframe[-1]
    try:
        return int(timeframe[:-1]) * _TIMEFRAME_UNITS_MS[unit]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


class SnapshotService:
    def __init__(
        self,
        primary: MarketDataProvider,
        fallback: MarketDataProvider | None,
        *,
        cache_ttl_seconds: float = 5.0,
        retries: int = 2,
        retry_backoff_seconds: float = 0.2,
        bar_limit: int = 100,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cache_ttl_seconds = cache_ttl_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.bar_limit = bar_limit
        self._cache: dict[tuple[str, str], tuple[float, list[Bar]]] = {}

    async def build_snapshot(self, payload: dict) -> MarketSnapshot:
        symbol = payload["symbol"]
        timeframe = payload["timeframe"]
        price = quantize(payload.get("price", "0"))
        bars = await self._get_bars(symbol, timeframe)
        if not bars:
            bars = [Bar.from_close(price)]
        closes = [bar.close for bar in bars]
        return MarketSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            price=price or closes[-1],
            closes=closes,
            bars=bars,
        )

    async def _get_bars(self, symbol: str, timeframe: str) -> list[Bar]:
        cached = self._cache.get((symbol, timeframe))
        if cached is not None and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            metrics.increment("market_data.cache_hit")
            return cached[1]
        metrics.increment("market_data.cache_miss")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                bars = await fetch_bars(self.primary, symbol, timeframe, self.bar_limit)
                self._cache[(symbol, timeframe)] = (time.monotonic(), bars)
                return bars
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
        return await fetch_bars(self.fallback, symbol, timeframe, self.bar_limit)


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
            bar_limit=ctx.settings.market_data_bar_limit,
        )

    async def on_shutdown(self, ctx: AppContext) -> None:
        provider = ctx.state.get("market_data_provider")
        if provider is not None:
            await provider.close()

    def pipeline_steps(self):
        return [PipelineStep(name="market_snapshot", order=10, handler=build_snapshot_step)]


module = MarketDataModule()
