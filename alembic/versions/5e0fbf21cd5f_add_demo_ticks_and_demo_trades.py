"""add demo_ticks and demo_trades

Revision ID: 5e0fbf21cd5f
Revises: 08ec8ff830fa
Create Date: 2026-03-01 14:55:38.397467

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5e0fbf21cd5f'
down_revision: Union[str, Sequence[str], None] = '08ec8ff830fa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table('demo_ticks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('bot_id', sa.String(length=100), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('action', sa.String(length=20), nullable=False),
        sa.Column('portfolio_value', sa.Float(), nullable=False),
        sa.Column('usdt_balance', sa.Float(), nullable=False),
        sa.Column('btc_balance', sa.Float(), nullable=False),
        sa.Column('reason', sa.String(length=500), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_table('demo_trades',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('bot_id', sa.String(length=100), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('action', sa.String(length=10), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('size_btc', sa.Float(), nullable=False),
        sa.Column('size_usdt', sa.Float(), nullable=False),
        sa.Column('fees', sa.Float(), nullable=False),
        sa.Column('portfolio_value', sa.Float(), nullable=False),
        sa.Column('reason', sa.String(length=500), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('demo_trades')
    op.drop_table('demo_ticks')
