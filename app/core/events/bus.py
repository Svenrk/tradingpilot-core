from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, Protocol


class EventBus(Protocol):
    async def publish(self, stream: str, payload: dict) -> None: ...
    async def consume(
        self, stream: str, consumer: str, group: str, timeout: float = 1.0
    ) -> dict | None: ...
    async def iter_stream(self, stream: str, consumer: str, group: str) -> AsyncIterator[dict]: ...
    async def iter_broadcast(self, stream: str) -> AsyncIterator[dict]: ...
    async def ack(self, stream: str, group: str, message_id: str) -> None: ...
    async def close(self) -> None: ...


class MemoryEventBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[dict]] = defaultdict(asyncio.Queue)
        self._subscribers: dict[str, set[asyncio.Queue[dict]]] = defaultdict(set)

    async def publish(self, stream: str, payload: dict) -> None:
        await self._queues[stream].put(dict(payload))
        for subscriber in list(self._subscribers[stream]):
            await subscriber.put(dict(payload))

    async def consume(
        self, stream: str, consumer: str, group: str, timeout: float = 1.0
    ) -> dict | None:
        del consumer, group
        try:
            return await asyncio.wait_for(self._queues[stream].get(), timeout=timeout)
        except TimeoutError:
            return None

    async def iter_stream(self, stream: str, consumer: str, group: str) -> AsyncIterator[dict]:
        while True:
            message = await self.consume(stream, consumer=consumer, group=group, timeout=1.0)
            if message is not None:
                yield message

    async def iter_broadcast(self, stream: str) -> AsyncIterator[dict]:
        subscriber: asyncio.Queue[dict] = asyncio.Queue()
        self._subscribers[stream].add(subscriber)
        try:
            while True:
                yield await subscriber.get()
        finally:
            self._subscribers[stream].discard(subscriber)

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        del stream, group, message_id

    async def close(self) -> None:
        return None


class RedisStreamBus:
    def __init__(self, redis_url: str) -> None:
        from redis.asyncio import Redis

        self.redis: Any = Redis.from_url(redis_url, decode_responses=True)

    async def publish(self, stream: str, payload: dict) -> None:
        serialised = {key: json.dumps(value) for key, value in payload.items()}
        await self.redis.xadd(stream, serialised)

    async def _ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except Exception:
            return None

    async def consume(
        self, stream: str, consumer: str, group: str, timeout: float = 1.0
    ) -> dict | None:
        await self._ensure_group(stream, group)
        response = await self.redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=1,
            block=int(timeout * 1000),
        )
        if not response:
            return None
        _, entries = response[0]
        message_id, values = entries[0]
        payload = {key: json.loads(value) for key, value in values.items()}
        payload["_message_id"] = message_id
        return payload

    async def iter_stream(self, stream: str, consumer: str, group: str) -> AsyncIterator[dict]:
        while True:
            message = await self.consume(stream, consumer=consumer, group=group, timeout=1.0)
            if message is not None:
                yield message

    async def iter_broadcast(self, stream: str) -> AsyncIterator[dict]:
        last_id = "$"
        while True:
            response = await self.redis.xread({stream: last_id}, count=1, block=1000)
            if not response:
                continue
            _, entries = response[0]
            for message_id, values in entries:
                last_id = message_id
                yield {key: json.loads(value) for key, value in values.items()}

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        await self.redis.xack(stream, group, message_id)

    async def close(self) -> None:
        await self.redis.aclose()


def create_event_bus(redis_url: str | None):
    return RedisStreamBus(redis_url) if redis_url else MemoryEventBus()
