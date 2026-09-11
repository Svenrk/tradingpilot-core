from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(url: str | None = None):
    global _engine
    if _engine is None or url is not None:
        _engine = create_async_engine(url or get_settings().database_url, future=True)
    return _engine


def get_session_factory(url: str | None = None) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None or url is not None:
        _session_factory = async_sessionmaker(get_engine(url), expire_on_commit=False)
    return _session_factory


def reset_db_state() -> None:
    global _engine, _session_factory
    _engine = None
    _session_factory = None


async def get_session() -> AsyncIterator[AsyncSession]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def init_db(url: str | None = None) -> None:
    engine = get_engine(url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
