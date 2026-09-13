from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

NUMERIC = Numeric(18, 8)


class TradingViewEvent(Base):
    __tablename__ = "tradingview_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    webhook_id: Mapped[str] = mapped_column(String(128), index=True)
    dedupe_key: Mapped[str] = mapped_column(String(128), unique=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[Decimal] = mapped_column(NUMERIC)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(8), index=True)
    strength: Mapped[Decimal] = mapped_column(NUMERIC)
    stop_loss: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), index=True)
    quantity: Mapped[Decimal] = mapped_column(NUMERIC)
    price: Mapped[Decimal] = mapped_column(NUMERIC)
    stop_loss: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    broker: Mapped[str] = mapped_column(String(32), default="paper")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=func.now()
    )


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_positions_symbol_status", "symbol", "status"),
        Index("ix_positions_status_closed_at", "status", "closed_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[Decimal] = mapped_column(NUMERIC)
    entry_price: Mapped[Decimal] = mapped_column(NUMERIC)
    stop_loss: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="OPEN")
    realized_pnl: Mapped[Decimal] = mapped_column(NUMERIC, default=Decimal("0"))
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MarketBar(Base):
    """Persisted OHLCV candle; the training corpus for the ML strategy."""

    __tablename__ = "market_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "open_time", name="uq_market_bars_key"),
        Index("ix_market_bars_symbol_timeframe_open_time", "symbol", "timeframe", "open_time"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(16))
    open_time: Mapped[int] = mapped_column(BigInteger)
    open: Mapped[Decimal] = mapped_column(NUMERIC)
    high: Mapped[Decimal] = mapped_column(NUMERIC)
    low: Mapped[Decimal] = mapped_column(NUMERIC)
    close: Mapped[Decimal] = mapped_column(NUMERIC)
    volume: Mapped[Decimal] = mapped_column(NUMERIC)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=func.now()
    )


class MLTrainingRun(Base):
    __tablename__ = "ml_training_runs"
    __table_args__ = (
        Index(
            "ix_ml_training_runs_symbol_timeframe_started_at",
            "symbol",
            "timeframe",
            "started_at",
        ),
        Index("ix_ml_training_runs_status_updated_at", "status", "updated_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=func.now()
    )
    n_bars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class PipelineInbox(Base):
    __tablename__ = "pipeline_inbox"
    __table_args__ = (
        UniqueConstraint("stream", "dedupe_key", name="uq_pipeline_inbox_stream_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    stream: Mapped[str] = mapped_column(String(64), index=True)
    dedupe_key: Mapped[str] = mapped_column(String(128))
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EventOutbox(Base):
    __tablename__ = "event_outbox"

    id: Mapped[int] = mapped_column(primary_key=True)
    stream: Mapped[str] = mapped_column(String(64), index=True)
    dedupe_key: Mapped[str] = mapped_column(String(128), unique=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
