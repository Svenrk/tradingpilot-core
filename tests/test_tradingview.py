import hashlib
import hmac

from sqlalchemy import select

from app.config import get_settings
from app.db import get_session_factory
from app.models import EventOutbox, TradingViewEvent


def _signature(body: bytes) -> str:
    return hmac.new(
        get_settings().tv_webhook_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


async def test_webhook_persists_event_and_outbox(client, test_db_url: str) -> None:
    body = b'{"event_id":"evt-1","symbol":"BTCUSDT","timeframe":"1m","side":"BUY","price":"101.25"}'
    response = await client.post(
        "/api/v1/tv/webhook/demo",
        content=body,
        headers={"X-Signature": _signature(body), "Content-Type": "application/json"},
    )
    assert response.status_code == 202
    session_factory = get_session_factory(test_db_url)
    async with session_factory() as session:
        event = await session.scalar(
            select(TradingViewEvent).where(TradingViewEvent.dedupe_key == "evt-1")
        )
        outbox = await session.scalar(select(EventOutbox).where(EventOutbox.dedupe_key == "evt-1"))
    assert event is not None
    assert outbox is not None


async def test_webhook_rejects_bad_signature(client) -> None:
    response = await client.post(
        "/api/v1/tv/webhook/demo",
        json={"symbol": "BTCUSDT", "timeframe": "1m", "side": "BUY", "price": "101.25"},
        headers={"X-Signature": "bad"},
    )
    assert response.status_code == 401
