from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.module import AppContext, BaseModule
from app.db import get_session
from app.models import Order, Position, Signal, TradingViewEvent
from app.schemas import (
    HealthResponse,
    OrderResponse,
    PositionResponse,
    SignalResponse,
    VersionResponse,
)
from app.security import get_session_data, require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["dashboard"])
READ_ROLES = {"read", "paper", "live"}


class UiBroadcaster:
    """Single stream reader fanning out to bounded per-client queues.

    One reader task consumes the broadcast stream regardless of how many
    websocket clients are connected. Slow clients get their oldest messages
    dropped instead of stalling the shared reader.
    """

    def __init__(self, ctx: AppContext, stream: str = "ui_updates", queue_size: int = 100) -> None:
        self._ctx = ctx
        self._stream = stream
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._reader: asyncio.Task | None = None

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        if self._reader is None or self._reader.done():
            self._reader = asyncio.create_task(self._read_forever())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(queue)

    async def _read_forever(self) -> None:
        try:
            async for message in self._ctx.event_bus.iter_broadcast(self._stream):
                for queue in list(self._subscribers):
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        with contextlib.suppress(asyncio.QueueEmpty):
                            queue.get_nowait()
                        with contextlib.suppress(asyncio.QueueFull):
                            queue.put_nowait(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("UI broadcast reader failed")

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        self._subscribers.clear()



@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    return VersionResponse(version=get_settings().app_version)


@router.get("/events")
async def events(
    session: AsyncSession = Depends(get_session),
    user=Depends(require_role("read", "paper", "live")),
):
    del user
    result = await session.execute(
        select(TradingViewEvent).order_by(TradingViewEvent.id.desc()).limit(50)
    )
    return [
        {
            "id": event.id,
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "side": event.side,
            "price": str(event.price),
        }
        for event in result.scalars()
    ]


@router.get("/signals", response_model=list[SignalResponse])
async def signals(
    session: AsyncSession = Depends(get_session),
    user=Depends(require_role("read", "paper", "live")),
):
    del user
    result = await session.execute(select(Signal).order_by(Signal.id.desc()).limit(50))
    return [
        SignalResponse.model_validate(signal, from_attributes=True) for signal in result.scalars()
    ]


@router.get("/orders", response_model=list[OrderResponse])
async def orders(
    session: AsyncSession = Depends(get_session),
    user=Depends(require_role("read", "paper", "live")),
):
    del user
    result = await session.execute(select(Order).order_by(Order.id.desc()).limit(50))
    return [OrderResponse.model_validate(order, from_attributes=True) for order in result.scalars()]


@router.get("/positions", response_model=list[PositionResponse])
async def positions(
    session: AsyncSession = Depends(get_session),
    user=Depends(require_role("read", "paper", "live")),
):
    del user
    result = await session.execute(select(Position).order_by(Position.id.desc()).limit(50))
    return [
        PositionResponse.model_validate(position, from_attributes=True)
        for position in result.scalars()
    ]


@router.websocket("/ws/updates")
async def ws_updates(websocket: WebSocket) -> None:
    ctx = websocket.app.state.ctx
    session_id = websocket.cookies.get(ctx.settings.session_cookie_name)
    session = await get_session_data(ctx.state["store"], session_id)
    csrf_token = websocket.headers.get("X-CSRF-Token") or websocket.query_params.get("csrf_token")
    if session is None or session.role not in READ_ROLES or csrf_token != session.csrf_token:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    broadcaster: UiBroadcaster = ctx.state["ui_broadcaster"]
    queue = broadcaster.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    finally:
        broadcaster.unsubscribe(queue)


class DashboardModule(BaseModule):
    name = "dashboard"

    async def on_startup(self, ctx: AppContext) -> None:
        ctx.state["ui_broadcaster"] = UiBroadcaster(ctx)

    async def on_shutdown(self, ctx: AppContext) -> None:
        broadcaster = ctx.state.get("ui_broadcaster")
        if broadcaster is not None:
            await broadcaster.close()

    def routers(self):
        return [router]


module = DashboardModule()
