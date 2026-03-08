"""add on-chain tables (funding_rates, open_interest, blockchain_metrics)

Revision ID: 94993078159f
Revises: f2c66dcd367e
Create Date: 2026-03-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "94993078159f"
down_revision: Union[str, Sequence[str], None] = "f2c66dcd367e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Funding Rate: cada 8h, historial complet des de set. 2019
    op.create_table(
        "funding_rates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rate", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_funding_rate_symbol_ts"),
    )
    op.create_index(
        "ix_funding_rates_symbol_timestamp",
        "funding_rates",
        ["symbol", "timestamp"],
    )

    # Open Interest: 1h, darrers 30 dies (límit API de Binance)
    op.create_table(
        "open_interest",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_interest_btc", sa.Float(), nullable=False),
        sa.Column("open_interest_usdt", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "symbol", "timeframe", "timestamp",
            name="uq_open_interest_symbol_tf_ts",
        ),
    )
    op.create_index(
        "ix_open_interest_symbol_tf_timestamp",
        "open_interest",
        ["symbol", "timeframe", "timestamp"],
    )

    # Blockchain Metrics: hash-rate, n-unique-addresses, transaction-fees (diari)
    op.create_table(
        "blockchain_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("metric", sa.String(length=50), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("metric", "timestamp", name="uq_blockchain_metric_ts"),
    )
    op.create_index(
        "ix_blockchain_metrics_metric_timestamp",
        "blockchain_metrics",
        ["metric", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_blockchain_metrics_metric_timestamp", table_name="blockchain_metrics")
    op.drop_table("blockchain_metrics")

    op.drop_index("ix_open_interest_symbol_tf_timestamp", table_name="open_interest")
    op.drop_table("open_interest")

    op.drop_index("ix_funding_rates_symbol_timestamp", table_name="funding_rates")
    op.drop_table("funding_rates")
