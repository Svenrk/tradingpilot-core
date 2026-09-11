from __future__ import annotations

import hashlib
import hmac
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.events.reliable import enqueue_outbox
from app.core.module import BaseModule
from app.db import get_session
from app.models import TradingViewEvent
from app.money import quantize

router = APIRouter(prefix="/api/v1/tv", tags=["tradingview"])


class TradingViewWebhook(BaseModel):
    event_id: str | None = None
    symbol: str
    timeframe: str
    side: str = Field(pattern="^(BUY|SELL|HOLD)$")
    price: str | int | float
    quantity: str | int | float = 1
    strategy: str | None = None


def verify_signature(secret: str, body: bytes, signature: str | None) -> bool:
    if signature is None:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


@router.post("/webhook/{webhook_id}", status_code=status.HTTP_202_ACCEPTED)
async def receive_webhook(
    webhook_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    settings = get_settings()
    body = await request.body()
    signature = request.headers.get("X-Signature")
    if not verify_signature(settings.tv_webhook_secret, body, signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")
    payload = TradingViewWebhook.model_validate_json(body)
    if payload.symbol not in settings.symbol_allowlist:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol not allowed")
    if payload.timeframe not in settings.timeframe_allowlist:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Timeframe not allowed")
    dedupe_key = payload.event_id or hashlib.sha256(body).hexdigest()
    try:
        event = TradingViewEvent(
            webhook_id=webhook_id,
            dedupe_key=dedupe_key,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            side=payload.side,
            price=quantize(payload.price),
            payload=payload.model_dump(mode="json"),
        )
        session.add(event)
        await session.flush()
        await enqueue_outbox(
            session,
            stream="tv_events",
            dedupe_key=dedupe_key,
            payload={"event_id": event.id, "dedupe_key": dedupe_key},
        )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return {"status": "duplicate", "dedupe_key": dedupe_key}
    return {"status": "accepted", "event_id": event.id, "dedupe_key": dedupe_key}


class TradingViewModule(BaseModule):
    name = "tradingview"

    def routers(self):
        return [router]

    def models(self):
        return [TradingViewEvent]


module = TradingViewModule()
