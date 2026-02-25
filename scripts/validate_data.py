# scripts/validate_data.py
import sys
import logging
from datetime import timezone

sys.path.append(".")

from core.db.session import SessionLocal
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

EXPECTED_GAPS = {
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
}


def check_counts(session):
    """Quantes candles tenim per timeframe."""
    logger.info("=== RECOMPTE PER TIMEFRAME ===")
    result = session.execute(text("""
        SELECT timeframe, COUNT(*) as total
        FROM candles
        GROUP BY timeframe
        ORDER BY timeframe
    """))
    for row in result:
        logger.info(f"  {row.timeframe}: {row.total:,} candles")


def check_range(session):
    """Rang temporal de cada timeframe."""
    logger.info("=== RANG TEMPORAL ===")
    result = session.execute(text("""
        SELECT timeframe, MIN(timestamp) as primera, MAX(timestamp) as ultima
        FROM candles
        GROUP BY timeframe
        ORDER BY timeframe
    """))
    for row in result:
        logger.info(f"  {row.timeframe}: {row.primera.date()} → {row.ultima.date()}")


def check_gaps(session):
    """Detecta gaps (forats) a les dades de cada timeframe."""
    logger.info("=== GAPS DETECTATS ===")
    for timeframe, interval in EXPECTED_GAPS.items():
        result = session.execute(text(f"""
            SELECT COUNT(*) as gaps
            FROM (
                SELECT timestamp,
                       LAG(timestamp) OVER (ORDER BY timestamp) as anterior
                FROM candles
                WHERE timeframe = :timeframe AND symbol = 'BTC/USDT'
            ) sub
            WHERE timestamp - anterior > INTERVAL '{interval}'
        """), {"timeframe": timeframe})
        gaps = result.scalar()
        if gaps == 0:
            logger.info(f"  {timeframe}: OK, sense gaps")
        else:
            logger.warning(f"  {timeframe}: {gaps} gaps detectats!")
def check_gap_detail(session):
    """Mostra exactament on són els gaps."""
    logger.info("=== DETALL GAPS 1h ===")
    result = session.execute(text("""
        SELECT anterior, timestamp, timestamp - anterior as diferencia
        FROM (
            SELECT timestamp,
                   LAG(timestamp) OVER (ORDER BY timestamp) as anterior
            FROM candles
            WHERE timeframe = '1h' AND symbol = 'BTC/USDT'
        ) sub
        WHERE timestamp - anterior > INTERVAL '1 hour'
    """))
    for row in result:
        logger.info(f"  Gap: {row.anterior} → {row.timestamp} ({row.diferencia})")

def check_duplicates(session):
    """Detecta candles duplicades."""
    logger.info("=== DUPLICATS ===")
    result = session.execute(text("""
        SELECT timeframe, COUNT(*) as duplicats
        FROM (
            SELECT timeframe, timestamp, COUNT(*) as cnt
            FROM candles
            WHERE symbol = 'BTC/USDT'
            GROUP BY timeframe, timestamp
            HAVING COUNT(*) > 1
        ) sub
        GROUP BY timeframe
    """))
    rows = result.fetchall()
    if not rows:
        logger.info("  OK, sense duplicats")
    else:
        for row in rows:
            logger.warning(f"  {row.timeframe}: {row.duplicats} duplicats!")


if __name__ == "__main__":
    session = SessionLocal()
    try:
        check_counts(session)
        check_range(session)
        check_gaps(session)
        check_gap_detail(session)
        check_duplicates(session)
        logger.info("=== VALIDACIÓ COMPLETADA ===")
    finally:
        session.close()