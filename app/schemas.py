from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str


class SignalResponse(BaseModel):
    id: int
    symbol: str
    timeframe: str
    action: Literal["BUY", "SELL", "HOLD"]
    strength: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    rationale: str


class OrderResponse(BaseModel):
    id: int
    client_order_id: str
    symbol: str
    side: str
    status: str
    quantity: Decimal
    price: Decimal


class PositionResponse(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    status: str
    realized_pnl: Decimal
