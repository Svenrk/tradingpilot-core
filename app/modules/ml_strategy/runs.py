from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.models import MLTrainingRun


def utc_now() -> datetime:
    return datetime.now(UTC)


def training_run_status(run: MLTrainingRun, *, stale_seconds: int) -> str:
    updated_at = run.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    if (
        run.status == "running"
        and run.finished_at is None
        and (utc_now() - updated_at).total_seconds() > stale_seconds
    ):
        return "stale"
    return run.status


def training_run_payload(run: MLTrainingRun, *, stale_seconds: int) -> dict[str, Any]:
    finished_at = run.finished_at
    started_at = run.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    if finished_at is not None and finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=UTC)
    end_time = finished_at or utc_now()
    return {
        "id": run.id,
        "symbol": run.symbol,
        "timeframe": run.timeframe,
        "status": training_run_status(run, stale_seconds=stale_seconds),
        "stage": run.stage,
        "progress": run.progress,
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": run.updated_at,
        "duration_seconds": max((end_time - started_at).total_seconds(), 0.0),
        "n_bars": run.n_bars,
        "n_samples": run.n_samples,
        "config": run.config,
        "metrics": run.metrics,
        "error": run.error,
        "model_path": run.model_path,
    }


async def create_training_run(
    session_factory,
    *,
    symbol: str,
    timeframe: str,
    config: dict[str, Any],
    status: str = "running",
    stage: str = "starting",
    progress: float | None = 0.0,
) -> int:
    async with session_factory() as session:
        run = MLTrainingRun(
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            stage=stage,
            progress=progress,
            config=config,
        )
        session.add(run)
        await session.flush()
        await session.commit()
        return run.id


async def update_training_run(session_factory, run_id: int, **changes: Any) -> None:
    async with session_factory() as session:
        run = await session.get(MLTrainingRun, run_id)
        if run is None:
            return
        for key, value in changes.items():
            setattr(run, key, value)
        run.updated_at = utc_now()
        await session.commit()


async def list_recent_training_runs(session, *, limit: int) -> list[MLTrainingRun]:
    statement = (
        select(MLTrainingRun)
        .order_by(MLTrainingRun.started_at.desc(), MLTrainingRun.id.desc())
        .limit(limit)
    )
    return list((await session.scalars(statement)).all())
