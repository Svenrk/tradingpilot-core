from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe
from typing import Any, Protocol

from fastapi import HTTPException, Request, status
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.config import Settings

PUBLIC_PATHS = {"/", "/api/v1/health", "/api/v1/version", "/api/v1/auth/login"}


class SessionData(BaseModel):
    user_id: str
    role: str
    csrf_token: str


class KeyValueStore(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> None: ...
    async def incr(self, key: str, ex: int | None = None) -> int: ...
    async def delete(self, key: str) -> None: ...
    async def close(self) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, datetime | None]] = {}

    async def get(self, key: str) -> str | None:
        value = self._data.get(key)
        if value is None:
            return None
        payload, expires_at = value
        if expires_at is not None and datetime.now(UTC) >= expires_at:
            self._data.pop(key, None)
            return None
        return payload

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        expires_at = None if ex is None else datetime.now(UTC) + timedelta(seconds=ex)
        self._data[key] = (value, expires_at)

    async def incr(self, key: str, ex: int | None = None) -> int:
        current = await self.get(key)
        if current is None:
            await self.set(key, "1", ex=ex)
            return 1
        value = int(current) + 1
        _, expires_at = self._data[key]
        self._data[key] = (str(value), expires_at)
        return value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def close(self) -> None:
        return None


class RedisStore:
    def __init__(self, redis_url: str) -> None:
        from redis.asyncio import Redis

        self.redis: Any = Redis.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        await self.redis.set(key, value, ex=ex)

    async def incr(self, key: str, ex: int | None = None) -> int:
        value = await self.redis.incr(key)
        if value == 1 and ex is not None:
            await self.redis.expire(key, ex)
        return int(value)

    async def delete(self, key: str) -> None:
        await self.redis.delete(key)

    async def close(self) -> None:
        await self.redis.aclose()


def create_store(settings: Settings) -> KeyValueStore:
    return RedisStore(settings.redis_url) if settings.redis_url else MemoryStore()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"{base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, hashed_password: str) -> bool:
    salt_b64, digest_b64 = hashed_password.split("$", 1)
    salt = base64.b64decode(salt_b64)
    expected = base64.b64decode(digest_b64)
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return hmac.compare_digest(actual, expected)


async def hash_password_async(password: str) -> str:
    """Run CPU-heavy scrypt hashing off the event loop."""
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(password: str, hashed_password: str) -> bool:
    """Run CPU-heavy scrypt verification off the event loop."""
    return await asyncio.to_thread(verify_password, password, hashed_password)


async def create_session(store: KeyValueStore, settings: Settings, data: SessionData) -> str:
    session_id = token_urlsafe(32)
    await store.set(
        _session_key(session_id), data.model_dump_json(), ex=settings.session_ttl_seconds
    )
    return session_id


async def get_session_data(store: KeyValueStore, session_id: str | None) -> SessionData | None:
    if not session_id:
        return None
    payload = await store.get(_session_key(session_id))
    return SessionData.model_validate_json(payload) if payload else None


async def delete_session(store: KeyValueStore, session_id: str | None) -> None:
    if session_id:
        await store.delete(_session_key(session_id))


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/api/v1/tv/webhook/")


class SecurityMiddleware:
    """Pure ASGI middleware: session/CSRF enforcement plus security headers.

    Avoids Starlette's BaseHTTPMiddleware task-per-request wrapper and skips
    the session-store lookup entirely for public paths.
    """

    _SECURITY_HEADERS = (
        (b"x-frame-options", b"DENY"),
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"no-referrer"),
        (b"cache-control", b"no-store"),
    )

    def __init__(self, app, *, settings: Settings, store: KeyValueStore) -> None:
        self.app = app
        self.settings = settings
        self.store = store

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        scope.setdefault("state", {})
        if is_public_path(scope["path"]):
            scope["state"]["session"] = None
        else:
            request = Request(scope)
            session_id = request.cookies.get(self.settings.session_cookie_name)
            session = await get_session_data(self.store, session_id)
            scope["state"]["session"] = session
            if session is None:
                response = JSONResponse(
                    {"detail": "Authentication required"}, status_code=status.HTTP_401_UNAUTHORIZED
                )
                await response(scope, receive, send)
                return
            if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}:
                csrf_token = request.headers.get("X-CSRF-Token")
                if csrf_token != session.csrf_token:
                    response = JSONResponse(
                        {"detail": "Invalid CSRF token"}, status_code=status.HTTP_403_FORBIDDEN
                    )
                    await response(scope, receive, send)
                    return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {key.lower() for key, _ in headers}
                for key, value in self._SECURITY_HEADERS:
                    if key not in existing:
                        headers.append((key, value))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def require_session(request: Request) -> SessionData:
    session = getattr(request.state, "session", None)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return session


def require_role(*roles: str):
    def _dependency(request: Request) -> SessionData:
        session = require_session(request)
        if session.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return session

    return _dependency
