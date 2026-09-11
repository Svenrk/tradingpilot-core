from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.db import reset_db_state


@pytest.fixture
def test_db_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def test_env(monkeypatch: pytest.MonkeyPatch, test_db_url: str) -> None:
    monkeypatch.setenv("TP_DATABASE_URL", test_db_url)
    monkeypatch.setenv("TP_REDIS_URL", "")
    monkeypatch.setenv("TP_TV_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("TP_ADMIN_PASSWORD", "test-password")
    get_settings.cache_clear()
    reset_db_state()


@pytest.fixture
async def client(test_env):
    from app.application import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
            yield api_client
