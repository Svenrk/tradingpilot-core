from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, Protocol


class EventBus(Protocol):
    async def publish(self, stream: str, payload: dict) -> None: ...
    async def publish_many(self, stream_payloads: list[tuple[str, dict]]) -> None: ...
    async def consume(
        self, stream: str, consumer: str, group: str, timeout: float = 1.0
    ) -> dict | None: ...
    async def consume_batch(
        self, stream: str, consumer: str, group: str, count: int = 10, timeout: float = 1.0
    ) -> list[dict]: ...
    async def claim_pending(
        self, stream: str, consumer: str, group: str, min_idle_seconds: float, count: int = 10
    ) -> list[dict]: ...
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

    async def publish_many(self, stream_payloads: list[tuple[str, dict]]) -> None:
        for stream, payload in stream_payloads:
            await self.publish(stream, payload)

    async def consume(
        self, stream: str, consumer: str, group: str, timeout: float = 1.0
    ) -> dict | None:
        del consumer, group
        try:
            return await asyncio.wait_for(self._queues[stream].get(), timeout=timeout)
        except TimeoutError:
            return None

    async def consume_batch(
        self, stream: str, consumer: str, group: str, count: int = 10, timeout: float = 1.0
    ) -> list[dict]:
        first = await self.consume(stream, consumer=consumer, group=group, timeout=timeout)
        if first is None:
            return []
        messages = [first]
        queue = self._queues[stream]
        while len(messages) < count:
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    async def claim_pending(
        self, stream: str, consumer: str, group: str, min_idle_seconds: float, count: int = 10
    ) -> list[dict]:
        del stream, consumer, group, min_idle_seconds, count
        return []

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
        self._ensured_groups: set[tuple[str, str]] = set()

    async def publish(self, stream: str, payload: dict) -> None:
        serialised = {key: json.dumps(value) for key, value in payload.items()}
        await self.redis.xadd(stream, serialised)

    async def publish_many(self, stream_payloads: list[tuple[str, dict]]) -> None:
        if not stream_payloads:
            return
        pipeline = self.redis.pipeline(transaction=False)
        for stream, payload in stream_payloads:
            serialised = {key: json.dumps(value) for key, value in payload.items()}
            pipeline.xadd(stream, serialised)
        await pipeline.execute()

    async def _ensure_group(self, stream: str, group: str) -> None:
        if (stream, group) in self._ensured_groups:
            return
        from redis.exceptions import ResponseError

        try:
            await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._ensured_groups.add((stream, group))

    @staticmethod
    def _decode_entry(message_id: str, values: dict) -> dict:
        payload = {key: json.loads(value) for key, value in values.items()}
        payload["_message_id"] = message_id
        return payload

    async def consume(
        self, stream: str, consumer: str, group: str, timeout: float = 1.0
    ) -> dict | None:
        messages = await self.consume_batch(stream, consumer, group, count=1, timeout=timeout)
        return messages[0] if messages else None

    async def consume_batch(
        self, stream: str, consumer: str, group: str, count: int = 10, timeout: float = 1.0
    ) -> list[dict]:
        await self._ensure_group(stream, group)
        response = await self.redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=int(timeout * 1000),
        )
        if not response:
            return []
        _, entries = response[0]
        return [self._decode_entry(message_id, values) for message_id, values in entries]

    async def claim_pending(
        self, stream: str, consumer: str, group: str, min_idle_seconds: float, count: int = 10
    ) -> list[dict]:
        """Claim messages left pending by dead consumers so they are reprocessed."""
        await self._ensure_group(stream, group)
        _, entries, _ = await self.redis.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_time=int(min_idle_seconds * 1000),
            start_id="0-0",
            count=count,
        )
        return [self._decode_entry(message_id, values) for message_id, values in entries]

    async def iter_stream(self, stream: str, consumer: str, group: str) -> AsyncIterator[dict]:
        while True:
            for message in await self.consume_batch(
                stream, consumer=consumer, group=group, timeout=1.0
            ):
                yield message

    async def iter_broadcast(self, stream: str) -> AsyncIterator[dict]:
        # Broadcast iteration starts at "$" (now): subscribers only see messages
        # published after they connect, and miss anything sent while disconnected.
        last_id = "$"
        while True:
            response = await self.redis.xread({stream: last_id}, count=100, block=1000)
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
