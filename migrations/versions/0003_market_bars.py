from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_market_bars"
down_revision = "0002_performance_indexes"
branch_labels = None
depends_on = None


NUMERIC = sa.Numeric(18, 8)


def upgrade() -> None:
    op.create_table(
        "market_bars",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("open_time", sa.BigInteger(), nullable=False),
        sa.Column("open", NUMERIC, nullable=False),
        sa.Column("high", NUMERIC, nullable=False),
        sa.Column("low", NUMERIC, nullable=False),
        sa.Column("close", NUMERIC, nullable=False),
        sa.Column("volume", NUMERIC, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("symbol", "timeframe", "open_time", name="uq_market_bars_key"),
    )
    op.create_index(
        "ix_market_bars_symbol_timeframe_open_time",
        "market_bars",
        ["symbol", "timeframe", "open_time"],
    )


def downgrade() -> None:
    op.drop_index("ix_market_bars_symbol_timeframe_open_time", table_name="market_bars")
    op.drop_table("market_bars")
