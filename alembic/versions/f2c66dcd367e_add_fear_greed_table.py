"""add fear_greed table

Revision ID: f2c66dcd367e
Revises: 5e0fbf21cd5f
Create Date: 2026-03-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2c66dcd367e"
down_revision: Union[str, Sequence[str], None] = "5e0fbf21cd5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fear_greed",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Integer(), nullable=False),
        sa.Column("classification", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("timestamp", name="uq_fear_greed_timestamp"),
    )


def downgrade() -> None:
    op.drop_table("fear_greed")
