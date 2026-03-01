# core/db/demo_repository.py
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from core.db.models import DemoTickDB, DemoTradeDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)


class DemoRepository:
    """
    Capa d'accés a dades per al Demo.
    Centralitza totes les operacions de lectura i escriptura
    de demo_ticks i demo_trades.
    """

    def save_tick(
        self,
        bot_id: str,
        timestamp: datetime,
        price: float,
        action: str,
        portfolio_value: float,
        usdt_balance: float,
        btc_balance: float,
        reason: str,
    ) -> None:
        session: Session = SessionLocal()
        try:
            tick = DemoTickDB(
                bot_id=bot_id,
                timestamp=timestamp,
                price=price,
                action=action,
                portfolio_value=portfolio_value,
                usdt_balance=usdt_balance,
                btc_balance=btc_balance,
                reason=reason,
            )
            session.add(tick)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error guardant tick: {e}")
        finally:
            session.close()

    def save_trade(
        self,
        bot_id: str,
        timestamp: datetime,
        action: str,
        price: float,
        size_btc: float,
        size_usdt: float,
        fees: float,
        portfolio_value: float,
        reason: str,
    ) -> None:
        session: Session = SessionLocal()
        try:
            trade = DemoTradeDB(
                bot_id=bot_id,
                timestamp=timestamp,
                action=action,
                price=price,
                size_btc=size_btc,
                size_usdt=size_usdt,
                fees=fees,
                portfolio_value=portfolio_value,
                reason=reason,
            )
            session.add(trade)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error guardant trade: {e}")
        finally:
            session.close()

    def get_last_portfolio_value(self, bot_id: str) -> float | None:
        """Recupera l'últim portfolio value d'un bot — per recuperar estat."""
        session: Session = SessionLocal()
        try:
            result = (
                session.query(DemoTickDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTickDB.timestamp.desc())
                .first()
            )
            return result.portfolio_value if result else None
        finally:
            session.close()

    def get_trades(self, bot_id: str) -> list[dict]:
        """Retorna tots els trades d'un bot ordenats per timestamp."""
        session: Session = SessionLocal()
        try:
            rows = (
                session.query(DemoTradeDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTradeDB.timestamp)
                .all()
            )
            return [{
                "timestamp": r.timestamp,
                "action": r.action,
                "price": r.price,
                "size_btc": r.size_btc,
                "size_usdt": r.size_usdt,
                "fees": r.fees,
                "portfolio_value": r.portfolio_value,
                "reason": r.reason,
            } for r in rows]
        finally:
            session.close()

    def get_portfolio_history(self, bot_id: str) -> list[dict]:
        """Retorna l'historial de portfolio value d'un bot."""
        session: Session = SessionLocal()
        try:
            rows = (
                session.query(DemoTickDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTickDB.timestamp)
                .all()
            )
            return [{
                "timestamp": r.timestamp,
                "portfolio_value": r.portfolio_value,
                "price": r.price,
                "action": r.action,
            } for r in rows]
        finally:
            session.close()

    def get_last_state(self, bot_id: str) -> dict | None:
        """
        Recupera l'últim estat conegut d'un bot.
        Retorna None si és la primera vegada que arrenca.
        """
        session: Session = SessionLocal()
        try:
            result = (
                session.query(DemoTickDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTickDB.timestamp.desc())
                .first()
            )
            if not result:
                return None
            return {
                "portfolio_value": result.portfolio_value,
                "usdt_balance": result.usdt_balance,
                "btc_balance": result.btc_balance,
                "timestamp": result.timestamp,
            }
        finally:
            session.close()