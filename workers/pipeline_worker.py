from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from typing import Any
from uuid import uuid4

from sqlalchemy import select

import app.models  # noqa: F401
from app.config import get_settings
from app.core.events.bus import create_event_bus
from app.core.events.reliable import flush_outbox, purge_history, record_inbox
from app.core.module import AppContext, PipelineContext
from app.core.registry import ModuleRegistry
from app.db import get_session_factory, init_db
from app.models import TradingViewEvent
from app.observability import configure_logging, metrics
from app.security import create_store
from workers.loop import run_loop

logger = logging.getLogger(__name__)


async def _run_loop_spec(ctx: AppContext, spec: Any) -> None:
    await spec.coro_factory(ctx)


async def process_pipeline_message(ctx: AppContext, payload: dict) -> bool:
    assert ctx.registry is not None
    async with ctx.session_factory() as session:
        event_id = payload.get("event_id")
        dedupe_key = payload["dedupe_key"]
        if not await record_inbox(session, stream="tv_events", dedupe_key=dedupe_key):
            return True
        event = await session.scalar(
            select(TradingViewEvent).where(TradingViewEvent.id == event_id)
        )
        pipeline_ctx = PipelineContext(app=ctx, session=session, payload=payload, event=event)
        try:
            for step in ctx.registry.pipeline_steps():
                started = time.perf_counter()
                await step.handler(pipeline_ctx)
                metrics.observe(f"pipeline.step.{step.name}", time.perf_counter() - started)
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Pipeline rejected event %s", dedupe_key)
            return False
    await ctx.event_bus.publish(
        "ui_updates",
        {
            "dedupe_key": dedupe_key,
            "signal": getattr(pipeline_ctx.signal, "action", None),
            "approved": getattr(pipeline_ctx.risk_decision, "approved", False),
            "order_id": getattr(pipeline_ctx.order, "id", None),
        },
    )
    return True


async def pump_outbox(ctx: AppContext) -> None:
    async with ctx.session_factory() as session:
        await flush_outbox(session, ctx.event_bus)
        await session.commit()


async def purge_expired_history(ctx: AppContext) -> None:
    async with ctx.session_factory() as session:
        deleted = await purge_history(session, older_than_days=ctx.settings.retention_days)
        await session.commit()
    if deleted:
        logger.info("Purged %s expired history rows", deleted)


async def _handle_message(ctx: AppContext, payload: dict, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        message_id = payload.get("_message_id")
        success = await process_pipeline_message(ctx, payload)
        if success and isinstance(message_id, str):
            await ctx.event_bus.ack("tv_events", "pipeline", message_id)


async def consume_forever(ctx: AppContext) -> None:
    consumer = str(uuid4())
    semaphore = asyncio.Semaphore(ctx.settings.worker_concurrency)
    while True:
        await pump_outbox(ctx)
        payloads = await ctx.event_bus.consume_batch(
            "tv_events",
            consumer=consumer,
            group="pipeline",
            count=ctx.settings.worker_batch_size,
            timeout=1.0,
        )
        if payloads:
            await asyncio.gather(
                *(_handle_message(ctx, payload, semaphore) for payload in payloads)
            )


async def reclaim_pending(ctx: AppContext) -> None:
    """Reprocess messages stranded by consumers that died mid-message."""
    consumer = str(uuid4())
    semaphore = asyncio.Semaphore(ctx.settings.worker_concurrency)
    payloads = await ctx.event_bus.claim_pending(
        "tv_events",
        consumer=consumer,
        group="pipeline",
        min_idle_seconds=ctx.settings.pending_claim_idle_seconds,
        count=ctx.settings.worker_batch_size,
    )
    if payloads:
        logger.warning("Reclaimed %s stale pending messages", len(payloads))
        await asyncio.gather(*(_handle_message(ctx, payload, semaphore) for payload in payloads))


async def report_metrics(ctx: AppContext) -> None:
    """Log hot-spot metrics: step timings, outbox lag, pending count, cache hit rate."""
    pending = await ctx.event_bus.pending_count("tv_events", "pipeline")
    metrics.set_gauge("stream.tv_events.pending", pending)
    logger.info("Worker metrics: %s", metrics.snapshot())


async def run_worker() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    store = create_store(settings)
    event_bus = create_event_bus(settings.redis_url)
    session_factory = get_session_factory(settings.database_url)
    registry = ModuleRegistry.discover(settings.enabled_modules)
    ctx = AppContext(settings=settings, session_factory=session_factory, event_bus=event_bus)
    ctx.registry = registry
    ctx.state["store"] = store
    await init_db(settings.database_url)
    await registry.startup(ctx)
    loops = [
        run_loop(
            "reclaim_pending",
            partial(reclaim_pending, ctx),
            float(ctx.settings.pending_claim_idle_seconds),
        ),
        run_loop(
            "purge_history",
            partial(purge_expired_history, ctx),
            ctx.settings.retention_interval_seconds,
        ),
        run_loop(
            "report_metrics",
            partial(report_metrics, ctx),
            ctx.settings.metrics_interval_seconds,
        ),
    ]
    for module in registry.modules:
        for spec in module.background_loops():
            loops.append(
                run_loop(
                    spec.name,
                    partial(_run_loop_spec, ctx, spec),
                    spec.interval_seconds,
                )
            )
    try:
        await asyncio.gather(consume_forever(ctx), *loops)
    finally:
        await registry.shutdown(ctx)
        await store.close()
        await event_bus.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
