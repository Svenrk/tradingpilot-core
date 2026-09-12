from __future__ import annotations

from typing import Any

import pytest

from app.modules.markets import (
    TOP_LIST_SIZE,
    MarketsProviderError,
    MarketsService,
    SyntheticMarketsProvider,
    build_top_lists,
)


class FailingProvider:
    async def get_tickers(self) -> list[dict[str, Any]]:
        raise MarketsProviderError("primary unavailable")


class StaticProvider:
    def __init__(self, tickers: list[dict[str, Any]]) -> None:
        self.tickers = tickers
        self.calls = 0

    async def get_tickers(self) -> list[dict[str, Any]]:
        self.calls += 1
        return self.tickers


def _ticker(symbol: str, volume: float, change: float) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": symbol,
        "price": 100.0,
        "change_percent": change,
        "volume": volume,
    }


def test_build_top_lists_sorts_and_limits() -> None:
    tickers = [_ticker(f"S{index}", volume=index, change=-index) for index in range(60)]
    lists = build_top_lists(tickers)
    assert len(lists["top_by_volume"]) == TOP_LIST_SIZE
    assert len(lists["top_by_change"]) == TOP_LIST_SIZE
    assert lists["top_by_volume"][0]["symbol"] == "S59"
    assert lists["top_by_change"][0]["symbol"] == "S0"


@pytest.mark.asyncio
async def test_markets_service_uses_fallback_provider() -> None:
    service = MarketsService(FailingProvider(), SyntheticMarketsProvider("TST"))
    payload = await service.get_top_lists()
    assert payload["source"] == "synthetic"
    assert len(payload["top_by_volume"]) == TOP_LIST_SIZE


@pytest.mark.asyncio
async def test_markets_service_caches_results() -> None:
    provider = StaticProvider([_ticker("BTCUSDT", 10.0, 1.0)])
    service = MarketsService(provider, None, cache_ttl_seconds=60.0)
    first = await service.get_top_lists()
    second = await service.get_top_lists()
    assert provider.calls == 1
    assert first is second
    assert first["source"] == "live"
    assert first["top_by_volume"][0]["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_markets_service_raises_without_fallback() -> None:
    service = MarketsService(FailingProvider(), None)
    with pytest.raises(MarketsProviderError):
        await service.get_top_lists()


@pytest.mark.asyncio
async def test_markets_endpoints_and_dashboard(client, monkeypatch: pytest.MonkeyPatch) -> None:
    tickers = [_ticker(f"C{index}USDT", volume=index * 10.0, change=index / 2) for index in range(5)]

    async def fake_get_tickers(self) -> list[dict[str, Any]]:
        return tickers

    from app.modules.markets import BinanceCryptoMarketsProvider, YahooStocksMarketsProvider

    monkeypatch.setattr(BinanceCryptoMarketsProvider, "get_tickers", fake_get_tickers)
    monkeypatch.setattr(YahooStocksMarketsProvider, "get_tickers", fake_get_tickers)

    for path in ("/api/v1/markets/crypto", "/api/v1/markets/stocks"):
        response = await client.get(path)
        assert response.status_code == 200
        payload = response.json()
        assert payload["source"] == "live"
        assert payload["top_by_volume"][0]["symbol"] == "C4USDT"
        assert payload["top_by_change"][0]["symbol"] == "C4USDT"

    page = await client.get("/dashboard")
    assert page.status_code == 200
    assert "TradingPilot Markets" in page.text
