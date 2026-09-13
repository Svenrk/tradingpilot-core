from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_ml_training_runs"
down_revision = "0003_market_bars"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ml_training_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("n_bars", sa.Integer(), nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("model_path", sa.Text(), nullable=True),
    )
    op.create_index("ix_ml_training_runs_status", "ml_training_runs", ["status"])
    op.create_index("ix_ml_training_runs_symbol", "ml_training_runs", ["symbol"])
    op.create_index("ix_ml_training_runs_timeframe", "ml_training_runs", ["timeframe"])
    op.create_index(
        "ix_ml_training_runs_symbol_timeframe_started_at",
        "ml_training_runs",
        ["symbol", "timeframe", "started_at"],
    )
    op.create_index(
        "ix_ml_training_runs_status_updated_at",
        "ml_training_runs",
        ["status", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ml_training_runs_status_updated_at", table_name="ml_training_runs")
    op.drop_index("ix_ml_training_runs_symbol_timeframe_started_at", table_name="ml_training_runs")
    op.drop_index("ix_ml_training_runs_timeframe", table_name="ml_training_runs")
    op.drop_index("ix_ml_training_runs_symbol", table_name="ml_training_runs")
    op.drop_index("ix_ml_training_runs_status", table_name="ml_training_runs")
    op.drop_table("ml_training_runs")
