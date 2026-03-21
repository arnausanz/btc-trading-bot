"""add gate tables: gate_positions and gate_near_misses

Revision ID: b3c4d5e6f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-03-20

Afegeix les dues taules del Gate System:
  - gate_positions: persisteix posicions obertes del GateBot entre reinicis
  - gate_near_misses: registra avaluacions on P1+P2+P3 passen (calibratge de llindars)

No modifica cap taula existent.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'b3c4d5e6f7a8'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── gate_positions ────────────────────────────────────────────────────────
    op.create_table(
        'gate_positions',
        sa.Column('id',            sa.Integer(),                    primary_key=True, autoincrement=True),
        sa.Column('bot_id',        sa.String(100),                  nullable=False),
        sa.Column('opened_at',     sa.DateTime(timezone=True),      nullable=False),
        sa.Column('entry_price',   sa.Float(),                      nullable=False),
        sa.Column('stop_level',    sa.Float(),                      nullable=False),
        sa.Column('target_level',  sa.Float(),                      nullable=False),
        sa.Column('highest_price', sa.Float(),                      nullable=False),
        sa.Column('size_usdt',     sa.Float(),                      nullable=False),
        sa.Column('regime',        sa.String(20),                   nullable=False),
        sa.Column('decel_counter', sa.Integer(),                    nullable=False, server_default='0'),
    )

    # ── gate_near_misses ──────────────────────────────────────────────────────
    op.create_table(
        'gate_near_misses',
        sa.Column('id',                sa.Integer(),               primary_key=True, autoincrement=True),
        sa.Column('timestamp',         sa.DateTime(timezone=True), nullable=False),
        sa.Column('bot_id',            sa.String(100),             nullable=False),
        # P1
        sa.Column('p1_regime',         sa.String(20),              nullable=False),
        sa.Column('p1_confidence',     sa.Float(),                 nullable=False),
        # P2
        sa.Column('p2_multiplier',     sa.Float(),                 nullable=False),
        # P3
        sa.Column('p3_level_type',     sa.String(20),              nullable=True),
        sa.Column('p3_level_strength', sa.Float(),                 nullable=True),
        sa.Column('p3_risk_reward',    sa.Float(),                 nullable=True),
        sa.Column('p3_volume_ratio',   sa.Float(),                 nullable=True),
        # P4
        sa.Column('p4_d1_ok',          sa.Boolean(),               nullable=True),
        sa.Column('p4_d2_ok',          sa.Boolean(),               nullable=True),
        sa.Column('p4_rsi_ok',         sa.Boolean(),               nullable=True),
        sa.Column('p4_macd_ok',        sa.Boolean(),               nullable=True),
        sa.Column('p4_score',          sa.Float(),                 nullable=True),
        sa.Column('p4_triggered',      sa.Boolean(),               nullable=True),
        # P5
        sa.Column('p5_veto_reason',    sa.String(100),             nullable=True),
        sa.Column('p5_position_size',  sa.Float(),                 nullable=True),
        # Resultat
        sa.Column('executed',          sa.Boolean(),               nullable=False, server_default='false'),
    )


def downgrade() -> None:
    op.drop_table('gate_near_misses')
    op.drop_table('gate_positions')
