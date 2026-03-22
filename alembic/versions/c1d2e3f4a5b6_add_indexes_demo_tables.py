"""add indexes on demo_ticks and demo_trades (bot_id + timestamp)

Revision ID: c1d2e3f4a5b6
Revises: b3c4d5e6f7a8
Create Date: 2026-03-22

Sense índexos, qualsevol consulta filter_by(bot_id=...) fa un full scan de
demo_ticks i demo_trades. Amb 10 bots corrents durant mesos, demo_ticks pot
tenir >5M files (1 tick/min × 10 bots × 6 mesos). Les consultes de
get_last_state() i get_portfolio_history() es fan lentes.

Afegim:
  - ix_demo_ticks_bot_id_ts   : (bot_id, timestamp DESC) — cobreix get_last_state() i get_portfolio_history()
  - ix_demo_trades_bot_id_ts  : (bot_id, timestamp)      — cobreix get_trades()
"""
from alembic import op

revision = 'c1d2e3f4a5b6'
down_revision = 'b3c4d5e6f7a8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_demo_ticks_bot_id_ts',
        'demo_ticks',
        ['bot_id', 'timestamp'],
    )
    op.create_index(
        'ix_demo_trades_bot_id_ts',
        'demo_trades',
        ['bot_id', 'timestamp'],
    )


def downgrade() -> None:
    op.drop_index('ix_demo_trades_bot_id_ts', table_name='demo_trades')
    op.drop_index('ix_demo_ticks_bot_id_ts',  table_name='demo_ticks')
