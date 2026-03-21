"""add confidence to demo_trades

Revision ID: a1b2c3d4e5f6
Revises: 5e0fbf21cd5f
Create Date: 2026-03-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '94993078159f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'demo_trades',
        sa.Column('confidence', sa.Float(), nullable=True, server_default='0.0'),
    )


def downgrade() -> None:
    op.drop_column('demo_trades', 'confidence')
