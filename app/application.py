from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import models as _models  # noqa: F401
from app.config import get_settings
from app.core.events.bus import create_event_bus
from app.core.module import AppContext
from app.core.registry import ModuleRegistry
from app.db import get_session_factory, init_db
from app.observability import CorrelationIdMiddleware, configure_logging
from app.security import SecurityMiddleware, create_store

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    store = create_store(settings)
    event_bus = create_event_bus(settings.redis_url)
    session_factory = get_session_factory(settings.database_url)
    registry = ModuleRegistry.discover(settings.enabled_modules)
    ctx = AppContext(settings=settings, session_factory=session_factory, event_bus=event_bus)
    ctx.registry = registry
    ctx.state["store"] = store

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Starting TradingPilot Core")
        await init_db(settings.database_url)
        await registry.startup(ctx)
        try:
            yield
        finally:
            await registry.shutdown(ctx)
            await store.close()
            await event_bus.close()

    fastapi_app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    fastapi_app.state.ctx = ctx
    fastapi_app.add_middleware(CorrelationIdMiddleware)
    fastapi_app.add_middleware(SecurityMiddleware, settings=settings, store=store)

    @fastapi_app.get("/")
    async def root() -> dict[str, str]:
        return {"name": settings.app_name, "status": "ok"}

    for module in registry.modules:
        for router in module.routers():
            fastapi_app.include_router(router)
    return fastapi_app


app: FastAPI = create_app()
