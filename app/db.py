from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engines: dict[str, AsyncEngine] = {}
_session_factories: dict[str, async_sessionmaker[AsyncSession]] = {}


def get_engine(url: str | None = None):
    resolved_url = url or get_settings().database_url
    engine = _engines.get(resolved_url)
    if engine is None:
        engine = create_async_engine(resolved_url, future=True)
        _engines[resolved_url] = engine
    return engine


def get_session_factory(url: str | None = None) -> async_sessionmaker[AsyncSession]:
    resolved_url = url or get_settings().database_url
    session_factory = _session_factories.get(resolved_url)
    if session_factory is None:
        session_factory = async_sessionmaker(get_engine(resolved_url), expire_on_commit=False)
        _session_factories[resolved_url] = session_factory
    return session_factory


def reset_db_state() -> None:
    for engine in _engines.values():
        engine.sync_engine.dispose()
    _engines.clear()
    _session_factories.clear()


async def get_session() -> AsyncIterator[AsyncSession]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def init_db(url: str | None = None) -> None:
    engine = get_engine(url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
