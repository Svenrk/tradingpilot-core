from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


NUMERIC = sa.Numeric(18, 8)


def upgrade() -> None:
    op.create_table(
        "tradingview_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("webhook_id", sa.String(length=128), nullable=False),
        sa.Column("dedupe_key", sa.String(length=128), nullable=False, unique=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("price", NUMERIC, nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=8), nullable=False),
        sa.Column("strength", NUMERIC, nullable=False),
        sa.Column("stop_loss", NUMERIC),
        sa.Column("take_profit", NUMERIC),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("client_order_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("quantity", NUMERIC, nullable=False),
        sa.Column("price", NUMERIC, nullable=False),
        sa.Column("stop_loss", NUMERIC),
        sa.Column("take_profit", NUMERIC),
        sa.Column("broker", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("quantity", NUMERIC, nullable=False),
        sa.Column("entry_price", NUMERIC, nullable=False),
        sa.Column("stop_loss", NUMERIC),
        sa.Column("take_profit", NUMERIC),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("realized_pnl", NUMERIC, nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "pipeline_inbox",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("stream", sa.String(length=64), nullable=False),
        sa.Column("dedupe_key", sa.String(length=128), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("stream", "dedupe_key", name="uq_pipeline_inbox_stream_key"),
    )
    op.create_table(
        "event_outbox",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("stream", sa.String(length=64), nullable=False),
        sa.Column("dedupe_key", sa.String(length=128), nullable=False, unique=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("event_outbox")
    op.drop_table("pipeline_inbox")
    op.drop_table("positions")
    op.drop_table("orders")
    op.drop_table("signals")
    op.drop_table("tradingview_events")
