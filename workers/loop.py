from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_loop(name: str, coro_factory: Callable[[], Awaitable[None]], interval: float) -> None:
    while True:
        try:
            await coro_factory()
            logger.debug("Loop %s heartbeat", name)
        except Exception:
            logger.exception("Loop %s failed", name)
        await asyncio.sleep(interval)
