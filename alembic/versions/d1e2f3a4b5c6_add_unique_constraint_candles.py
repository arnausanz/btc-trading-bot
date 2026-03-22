"""add unique constraint to candles table (DT1 audit fix)

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-03-22

La taula candles no tenia UniqueConstraint, a diferència de totes les altres
taules de dades (fear_greed, funding_rates, open_interest, blockchain_metrics).
Executar download_data.py dues vegades sobre el mateix rang insertava duplicats
silenciosament.

Aquesta migració:
  1. Elimina els duplicats existents (mantenint el registre amb MIN(id))
  2. Afegeix UniqueConstraint(exchange, symbol, timeframe, timestamp)

IMPORTANT: Executar alembic upgrade head ANTES de tornar a fer servir
download_data.py o el DemoRunner.
"""
from alembic import op
import sqlalchemy as sa


revision = 'd1e2f3a4b5c6'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Step 1: Remove duplicate rows, keeping the one with MIN(id) per unique key.
    # This is safe — duplicates have identical OHLCV values by construction.
    op.execute("""
        DELETE FROM candles
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM candles
            GROUP BY exchange, symbol, timeframe, timestamp
        )
    """)

    # Step 2: Add the unique constraint now that duplicates are gone.
    op.create_unique_constraint(
        'uq_candles_exchange_symbol_tf_ts',
        'candles',
        ['exchange', 'symbol', 'timeframe', 'timestamp'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_candles_exchange_symbol_tf_ts', 'candles', type_='unique')
