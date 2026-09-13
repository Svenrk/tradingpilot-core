"""Persist and load OHLCV bars for training and inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MarketBar
from app.modules.market_data import Bar
from app.modules.ml_strategy.features import BarArrays

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


async def upsert_bars(
    session: AsyncSession, symbol: str, timeframe: str, bars: Sequence[Bar]
) -> int:
    """Insert bars, refreshing existing rows (the newest candle may still be forming)."""
    if not bars:
        return 0
    rows = [
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "open_time": int(bar.open_time),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    ]
    dialect = session.get_bind().dialect.name
    insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
    statement = insert(MarketBar).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=["symbol", "timeframe", "open_time"],
        set_={column: getattr(statement.excluded, column) for column in _BAR_COLUMNS},
    )
    await session.execute(statement)
    return len(rows)


async def load_bars(
    session: AsyncSession, symbol: str, timeframe: str, *, limit: int | None = None
) -> list[Bar]:
    statement = (
        select(MarketBar)
        .where(MarketBar.symbol == symbol, MarketBar.timeframe == timeframe)
        .order_by(MarketBar.open_time.desc())
    )
    if limit is not None:
        statement = statement.limit(limit)
    rows = list(reversed((await session.scalars(statement)).all()))
    return [
        Bar(
            open_time=row.open_time,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )
        for row in rows
    ]


async def load_bar_arrays(
    session: AsyncSession, symbol: str, timeframe: str, *, limit: int | None = None
) -> BarArrays:
    bars = await load_bars(session, symbol, timeframe, limit=limit)
    if not bars:
        empty = np.empty(0)
        return BarArrays(np.empty(0, dtype=np.int64), empty, empty, empty, empty, empty)
    return BarArrays.from_bars(bars)


async def count_bars(session: AsyncSession, symbol: str, timeframe: str) -> int:
    from sqlalchemy import func

    return int(
        await session.scalar(
            select(func.count())
            .select_from(MarketBar)
            .where(MarketBar.symbol == symbol, MarketBar.timeframe == timeframe)
        )
        or 0
    )
