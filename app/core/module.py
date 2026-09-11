from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings

if TYPE_CHECKING:
    from app.core.events.bus import EventBus
    from app.core.registry import ModuleRegistry
    from app.models import Order, Position, TradingViewEvent


@dataclass(slots=True)
class AppContext:
    settings: Settings
    session_factory: async_sessionmaker[AsyncSession]
    event_bus: EventBus
    registry: ModuleRegistry | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineStep:
    name: str
    order: int
    handler: Any


@dataclass(frozen=True, slots=True)
class LoopSpec:
    name: str
    coro_factory: Any
    interval_seconds: float


@dataclass(slots=True)
class PipelineContext:
    app: AppContext
    session: AsyncSession
    payload: dict[str, Any]
    event: TradingViewEvent | None = None
    snapshot: Any = None
    signal: Any = None
    risk_decision: Any = None
    order: Order | None = None
    position: Position | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Module(Protocol):
    name: str
    depends_on: tuple[str, ...]

    def routers(self) -> list[APIRouter]: ...
    def models(self) -> list[type]: ...
    def pipeline_steps(self) -> list[PipelineStep]: ...
    def background_loops(self) -> list[LoopSpec]: ...
    async def on_startup(self, ctx: AppContext) -> None: ...
    async def on_shutdown(self, ctx: AppContext) -> None: ...


class BaseModule:
    name = "base"
    depends_on: tuple[str, ...] = ()

    def routers(self) -> list[APIRouter]:
        return []

    def models(self) -> list[type]:
        return []

    def pipeline_steps(self) -> list[PipelineStep]:
        return []

    def background_loops(self) -> list[LoopSpec]:
        return []

    async def on_startup(self, ctx: AppContext) -> None:
        return None

    async def on_shutdown(self, ctx: AppContext) -> None:
        return None
