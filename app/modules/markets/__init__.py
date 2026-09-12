from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.core.module import AppContext, BaseModule

logger = logging.getLogger(__name__)

router = APIRouter(tags=["markets"])

TOP_LIST_SIZE = 50
_STATIC_DIR = Path(__file__).parent / "static"


class MarketsProviderError(RuntimeError):
    pass


class MarketsProvider(Protocol):
    async def get_tickers(self) -> list[dict[str, Any]]: ...


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_top_lists(tickers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the top 50 tickers by quote volume and by 24h percent change."""
    by_volume = sorted(tickers, key=lambda t: _to_float(t.get("volume")), reverse=True)
    by_change = sorted(tickers, key=lambda t: _to_float(t.get("change_percent")), reverse=True)
    return {
        "top_by_volume": by_volume[:TOP_LIST_SIZE],
        "top_by_change": by_change[:TOP_LIST_SIZE],
    }


class BinanceCryptoMarketsProvider:
    """Fetches 24h ticker statistics for USDT pairs from Binance."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def get_tickers(self) -> list[dict[str, Any]]:
        try:
            response = await self._get_client().get("https://api.binance.com/api/v3/ticker/24hr")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise MarketsProviderError("Binance 24h ticker request failed") from exc
        tickers: list[dict[str, Any]] = []
        for row in data:
            symbol = row.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            tickers.append(
                {
                    "symbol": symbol,
                    "name": symbol.removesuffix("USDT"),
                    "price": _to_float(row.get("lastPrice")),
                    "change_percent": _to_float(row.get("priceChangePercent")),
                    "volume": _to_float(row.get("quoteVolume")),
                }
            )
        return tickers

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class YahooStocksMarketsProvider:
    """Fetches the most active US stocks from the Yahoo Finance screener API."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=10.0, headers={"User-Agent": "Mozilla/5.0 (TradingPilot)"}
            )
        return self._client

    async def _fetch_screener(self, scr_id: str) -> list[dict[str, Any]]:
        response = await self._get_client().get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"scrIds": scr_id, "count": 250},
        )
        response.raise_for_status()
        data = response.json()
        return data["finance"]["result"][0].get("quotes", [])

    async def get_tickers(self) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        try:
            for scr_id in ("most_actives", "day_gainers", "day_losers"):
                for quote in await self._fetch_screener(scr_id):
                    symbol = quote.get("symbol")
                    if not symbol or symbol in merged:
                        continue
                    price = _to_float(quote.get("regularMarketPrice"))
                    volume = _to_float(quote.get("regularMarketVolume"))
                    merged[symbol] = {
                        "symbol": symbol,
                        "name": quote.get("shortName") or symbol,
                        "price": price,
                        "change_percent": _to_float(quote.get("regularMarketChangePercent")),
                        "volume": volume * price,
                    }
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise MarketsProviderError("Yahoo Finance screener request failed") from exc
        return list(merged.values())

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class SyntheticMarketsProvider:
    """Deterministic placeholder data used when the live provider is unavailable."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    async def get_tickers(self) -> list[dict[str, Any]]:
        tickers: list[dict[str, Any]] = []
        for index in range(1, 101):
            symbol = f"{self._prefix}{index:03d}"
            tickers.append(
                {
                    "symbol": symbol,
                    "name": f"Synthetic {symbol}",
                    "price": round(100 + index * 0.5, 2),
                    "change_percent": round((index % 21) - 10 + index * 0.01, 2),
                    "volume": float(1_000_000 * ((index * 37) % 100 + 1)),
                }
            )
        return tickers


class MarketsService:
    """Caches top-50 lists per asset class with a shared TTL."""

    def __init__(
        self,
        primary: MarketsProvider,
        fallback: MarketsProvider | None,
        *,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: tuple[float, dict[str, Any]] | None = None

    async def get_top_lists(self) -> dict[str, Any]:
        if self._cache is not None and (time.monotonic() - self._cache[0]) < self.cache_ttl_seconds:
            return self._cache[1]
        try:
            tickers = await self.primary.get_tickers()
            source = "live"
        except MarketsProviderError:
            if self.fallback is None:
                raise
            logger.warning("Primary markets provider unavailable, using synthetic fallback")
            tickers = await self.fallback.get_tickers()
            source = "synthetic"
        payload = {"source": source, **build_top_lists(tickers)}
        self._cache = (time.monotonic(), payload)
        return payload


def _get_service(request: Request, key: str) -> MarketsService:
    service = request.app.state.ctx.state.get(key)
    if service is None:
        raise HTTPException(status_code=503, detail="Markets module is not ready")
    return service


@router.get("/api/v1/markets/crypto")
async def crypto_markets(request: Request) -> dict[str, Any]:
    try:
        return await _get_service(request, "crypto_markets_service").get_top_lists()
    except MarketsProviderError as exc:
        raise HTTPException(status_code=502, detail="Crypto market data unavailable") from exc


@router.get("/api/v1/markets/stocks")
async def stock_markets(request: Request) -> dict[str, Any]:
    try:
        return await _get_service(request, "stock_markets_service").get_top_lists()
    except MarketsProviderError as exc:
        raise HTTPException(status_code=502, detail="Stock market data unavailable") from exc


@router.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


class MarketsModule(BaseModule):
    name = "markets"

    async def on_startup(self, ctx: AppContext) -> None:
        crypto_provider = BinanceCryptoMarketsProvider()
        stocks_provider = YahooStocksMarketsProvider()
        ctx.state["crypto_markets_provider"] = crypto_provider
        ctx.state["stock_markets_provider"] = stocks_provider
        ctx.state["crypto_markets_service"] = MarketsService(
            crypto_provider, SyntheticMarketsProvider("CRY")
        )
        ctx.state["stock_markets_service"] = MarketsService(
            stocks_provider, SyntheticMarketsProvider("STK")
        )

    async def on_shutdown(self, ctx: AppContext) -> None:
        for key in ("crypto_markets_provider", "stock_markets_provider"):
            provider = ctx.state.get(key)
            if provider is not None:
                await provider.close()

    def routers(self):
        return [router]


module = MarketsModule()
