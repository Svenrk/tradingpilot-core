from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.module import BaseModule
from app.db import get_session
from app.models import Order, Position, Signal, TradingViewEvent
from app.schemas import (
    HealthResponse,
    OrderResponse,
    PositionResponse,
    SignalResponse,
    VersionResponse,
)
from app.security import require_role

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


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
    await websocket.accept()
    ctx = websocket.app.state.ctx
    async for message in ctx.event_bus.iter_stream(
        "ui_updates", consumer=str(uuid4()), group="dashboard"
    ):
        await websocket.send_json(message)


class DashboardModule(BaseModule):
    name = "dashboard"

    def routers(self):
        return [router]


module = DashboardModule()
