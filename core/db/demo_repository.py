# core/db/demo_repository.py
import logging
import time
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from core.db.models import DemoTickDB, DemoTradeDB, CandleDB, FearGreedDB, GatePositionDB, GateNearMissDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)


class DemoRepository:
    """Data access layer for the Demo."""

    # ──────────────────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────────────────

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
            session.add(DemoTickDB(
                bot_id=bot_id,
                timestamp=timestamp,
                price=price,
                action=action,
                portfolio_value=portfolio_value,
                usdt_balance=usdt_balance,
                btc_balance=btc_balance,
                reason=reason,
            ))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving tick: {e}")
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
        confidence: float = 0.0,
    ) -> None:
        session: Session = SessionLocal()
        try:
            session.add(DemoTradeDB(
                bot_id=bot_id,
                timestamp=timestamp,
                action=action,
                price=price,
                size_btc=size_btc,
                size_usdt=size_usdt,
                fees=fees,
                portfolio_value=portfolio_value,
                confidence=confidence,
                reason=reason,
            ))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trade: {e}")
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────────────────
    # Read operations — state restore
    # ──────────────────────────────────────────────────────────────────────

    def get_last_state(self, bot_id: str) -> dict | None:
        """Returns the last recorded state of a bot, or None if first run."""
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

    # ──────────────────────────────────────────────────────────────────────
    # Reset operations
    # ──────────────────────────────────────────────────────────────────────

    def reset_bot_state(self, bot_id: str) -> tuple[int, int]:
        """
        Deletes all ticks, trades and open gate positions for a bot.
        Returns (ticks_deleted, trades_deleted).

        After this, get_last_state() returns None → bot starts fresh with
        initial_capital on next run.

        Gate positions are also cleared to avoid a capital inconsistency:
        if the bot had open positions, their cost basis would mismatch the
        freshly reset portfolio_value.
        """
        session: Session = SessionLocal()
        try:
            ticks     = session.query(DemoTickDB).filter_by(bot_id=bot_id).delete()
            trades    = session.query(DemoTradeDB).filter_by(bot_id=bot_id).delete()
            positions = session.query(GatePositionDB).filter_by(bot_id=bot_id).delete()
            session.commit()
            logger.info(
                f"Reset {bot_id}: deleted {ticks} ticks, {trades} trades, "
                f"{positions} gate positions."
            )
            return ticks, trades
        except Exception as e:
            session.rollback()
            logger.error(f"Error resetting bot {bot_id}: {e}")
            raise
        finally:
            session.close()

    def reset_all_states(self) -> dict[str, tuple[int, int]]:
        """
        Deletes all demo ticks, trades and open gate positions for every bot.
        Returns {bot_id: (ticks, trades)}.
        """
        session: Session = SessionLocal()
        try:
            # Get all distinct bot_ids before deleting
            bot_ids = [r[0] for r in session.query(DemoTickDB.bot_id).distinct().all()]
            results = {}
            for bot_id in bot_ids:
                ticks  = session.query(DemoTickDB).filter_by(bot_id=bot_id).delete()
                trades = session.query(DemoTradeDB).filter_by(bot_id=bot_id).delete()
                session.query(GatePositionDB).filter_by(bot_id=bot_id).delete()
                results[bot_id] = (ticks, trades)
            session.commit()
            logger.info(f"Full reset: {results}")
            return results
        except Exception as e:
            session.rollback()
            logger.error(f"Error in full reset: {e}")
            raise
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────────────────
    # Read operations — trades
    # ──────────────────────────────────────────────────────────────────────

    def get_trades(self, bot_id: str) -> list[dict]:
        """Returns all trades of a bot ordered by timestamp."""
        session: Session = SessionLocal()
        try:
            rows = (
                session.query(DemoTradeDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTradeDB.timestamp)
                .all()
            )
            return [{
                "bot_id":          bot_id,
                "timestamp":       r.timestamp,
                "action":          r.action,
                "price":           r.price,
                "size_btc":        r.size_btc,
                "size_usdt":       r.size_usdt,
                "fees":            r.fees,
                "portfolio_value": r.portfolio_value,
                "confidence":      r.confidence or 0.0,
                "reason":          r.reason,
            } for r in rows]
        finally:
            session.close()

    def get_all_trades(self, bot_ids: list[str]) -> list[dict]:
        """Returns all trades for a list of bots, sorted by timestamp."""
        all_trades: list[dict] = []
        for bot_id in bot_ids:
            all_trades.extend(self.get_trades(bot_id))
        all_trades.sort(key=lambda t: t["timestamp"])
        return all_trades

    # ──────────────────────────────────────────────────────────────────────
    # Read operations — portfolio history
    # ──────────────────────────────────────────────────────────────────────

    def get_portfolio_history(self, bot_id: str) -> list[dict]:
        """Returns the full portfolio history of a bot (all ticks)."""
        session: Session = SessionLocal()
        try:
            rows = (
                session.query(DemoTickDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTickDB.timestamp)
                .all()
            )
            return [{
                "timestamp":       r.timestamp,
                "portfolio_value": r.portfolio_value,
                "price":           r.price,
                "action":          r.action,
            } for r in rows]
        finally:
            session.close()

    def get_all_portfolios(self, bot_ids: list[str]) -> dict[str, list[dict]]:
        """Returns portfolio histories keyed by bot_id."""
        return {bot_id: self.get_portfolio_history(bot_id) for bot_id in bot_ids}

    def get_first_tick(self, bot_id: str) -> dict | None:
        """Returns the very first tick (start date + BTC price) for a bot."""
        session: Session = SessionLocal()
        try:
            result = (
                session.query(DemoTickDB)
                .filter_by(bot_id=bot_id)
                .order_by(DemoTickDB.timestamp.asc())
                .first()
            )
            if not result:
                return None
            return {
                "timestamp": result.timestamp,
                "price":     result.price,
            }
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────────────────
    # Read operations — health checks
    # ──────────────────────────────────────────────────────────────────────

    def get_candle_last_update(self, symbol: str = "BTC/USDT", timeframe: str = "1h") -> datetime | None:
        """Returns the timestamp of the most recent candle in the DB."""
        session: Session = SessionLocal()
        try:
            result = (
                session.query(func.max(CandleDB.timestamp))
                .filter_by(symbol=symbol, timeframe=timeframe)
                .scalar()
            )
            return result
        except Exception as e:
            logger.error(f"Error reading candle last update: {e}")
            return None
        finally:
            session.close()

    def get_fear_greed_last(self) -> dict | None:
        """Returns the most recent Fear & Greed entry."""
        session: Session = SessionLocal()
        try:
            result = (
                session.query(FearGreedDB)
                .order_by(FearGreedDB.timestamp.desc())
                .first()
            )
            if not result:
                return None
            return {
                "timestamp":      result.timestamp,
                "value":          result.value,
                "classification": result.classification,
            }
        except Exception as e:
            logger.error(f"Error reading fear_greed: {e}")
            return None
        finally:
            session.close()

    def check_db_connection(self) -> bool:
        """Returns True if the DB connection works."""
        session: Session = SessionLocal()
        try:
            session.execute(session.get_bind().connect().execute.__func__.__doc__ and
                            __import__("sqlalchemy").text("SELECT 1"))
            return True
        except Exception:
            try:
                # Simpler check
                session.query(DemoTickDB).limit(1).all()
                return True
            except Exception:
                return False
        finally:
            session.close()

    def ping_db(self) -> bool:
        """Quick DB ping — returns True if connection is alive."""
        session: Session = SessionLocal()
        try:
            from sqlalchemy import text
            session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────────────────
    # Gate System — posicions obertes
    # ──────────────────────────────────────────────────────────────────────

    def save_gate_position(self, bot_id: str, position: dict) -> int:
        """
        Persisteix una nova posició oberta del GateBot.
        Retorna l'id generat per poder-lo guardar en memòria.
        """
        session: Session = SessionLocal()
        try:
            row = GatePositionDB(
                bot_id=bot_id,
                opened_at=position["opened_at"],
                entry_price=position["entry_price"],
                stop_level=position["stop_level"],
                target_level=position["target_level"],
                highest_price=position["highest_price"],
                size_usdt=position["size_usdt"],
                regime=position["regime"],
                decel_counter=position.get("decel_counter", 0),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving gate position: {e}")
            return -1
        finally:
            session.close()

    def update_gate_position(self, position_id: int, updates: dict) -> None:
        """Actualitza camps d'una posició oberta (stop_level, highest_price, decel_counter, etc.)."""
        session: Session = SessionLocal()
        try:
            row = session.query(GatePositionDB).filter_by(id=position_id).first()
            if row:
                for k, v in updates.items():
                    setattr(row, k, v)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating gate position {position_id}: {e}")
        finally:
            session.close()

    def delete_gate_position(self, position_id: int) -> None:
        """Elimina una posició un cop tancada."""
        session: Session = SessionLocal()
        try:
            row = session.query(GatePositionDB).filter_by(id=position_id).first()
            if row:
                session.delete(row)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting gate position {position_id}: {e}")
        finally:
            session.close()

    def get_open_gate_positions(self, bot_id: str) -> list[dict]:
        """Retorna totes les posicions obertes del GateBot (per restaurar en reinici)."""
        session: Session = SessionLocal()
        try:
            rows = (
                session.query(GatePositionDB)
                .filter_by(bot_id=bot_id)
                .order_by(GatePositionDB.opened_at)
                .all()
            )
            return [{
                "id":            r.id,
                "opened_at":     r.opened_at,
                "entry_price":   r.entry_price,
                "stop_level":    r.stop_level,
                "target_level":  r.target_level,
                "highest_price": r.highest_price,
                "size_usdt":     r.size_usdt,
                "regime":        r.regime,
                "decel_counter": r.decel_counter,
            } for r in rows]
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────────────────
    # Gate System — near-miss log
    # ──────────────────────────────────────────────────────────────────────

    def save_gate_near_miss(self, snapshot: dict) -> None:
        """
        Guarda un registre de near-miss (P1+P2+P3 passen).
        snapshot és un dict amb tots els camps de GateNearMissDB.
        """
        session: Session = SessionLocal()
        try:
            session.add(GateNearMissDB(**snapshot))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving gate near miss: {e}")
        finally:
            session.close()
