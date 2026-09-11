from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EventOutbox, PipelineInbox, Signal, TradingViewEvent


async def enqueue_outbox(
    session: AsyncSession,
    *,
    stream: str,
    payload: dict,
    dedupe_key: str,
) -> EventOutbox:
    record = EventOutbox(stream=stream, dedupe_key=dedupe_key, payload=payload)
    session.add(record)
    await session.flush()
    return record


async def flush_outbox(session: AsyncSession, bus, limit: int = 100) -> int:
    result = await session.execute(
        select(EventOutbox)
        .where(EventOutbox.published_at.is_(None))
        .order_by(EventOutbox.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    records = list(result.scalars())
    if not records:
        return 0
    await bus.publish_many([(record.stream, record.payload) for record in records])
    now = datetime.now(UTC)
    for record in records:
        record.published_at = now
    await session.flush()
    return len(records)


async def purge_history(session: AsyncSession, *, older_than_days: int) -> int:
    """Delete processed/archived rows older than the retention window."""
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    deleted = 0
    for statement in (
        delete(EventOutbox).where(
            EventOutbox.published_at.is_not(None), EventOutbox.published_at < cutoff
        ),
        delete(PipelineInbox).where(PipelineInbox.processed_at < cutoff),
        delete(TradingViewEvent).where(TradingViewEvent.received_at < cutoff),
        delete(Signal).where(Signal.created_at < cutoff),
    ):
        result = await session.execute(statement)
        deleted += int(getattr(result, "rowcount", 0) or 0)
    return deleted


async def record_inbox(session: AsyncSession, *, stream: str, dedupe_key: str) -> bool:
    try:
        async with session.begin_nested():
            marker = PipelineInbox(stream=stream, dedupe_key=dedupe_key)
            session.add(marker)
            await session.flush()
    except IntegrityError:
        return False
    return True
