from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EventOutbox, PipelineInbox


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
    )
    published = 0
    for record in result.scalars():
        await bus.publish(record.stream, record.payload)
        record.published_at = datetime.now(UTC)
        published += 1
    if published:
        await session.flush()
    return published


async def record_inbox(session: AsyncSession, *, stream: str, dedupe_key: str) -> bool:
    try:
        async with session.begin_nested():
            marker = PipelineInbox(stream=stream, dedupe_key=dedupe_key)
            session.add(marker)
            await session.flush()
    except IntegrityError:
        return False
    return True
