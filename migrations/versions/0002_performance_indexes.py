from __future__ import annotations

from alembic import op

revision = "0002_performance_indexes"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_positions_symbol_status", "positions", ["symbol", "status"])
    op.create_index("ix_positions_status_closed_at", "positions", ["status", "closed_at"])
    op.create_index("ix_event_outbox_published_at", "event_outbox", ["published_at"])


def downgrade() -> None:
    op.drop_index("ix_event_outbox_published_at", table_name="event_outbox")
    op.drop_index("ix_positions_status_closed_at", table_name="positions")
    op.drop_index("ix_positions_symbol_status", table_name="positions")
