# data/sources/ohlcv.py
import ccxt
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from core.models import Candle
from core.db.models import CandleDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Timeframe duration in seconds — used to exclude the still-open current candle.
_TF_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400,
}


class OHLCVFetcher:
    """
    Downloads OHLCV candles from an exchange via ccxt
    and persists them to the DB.
    """

    def __init__(self, exchange_id: str = "binance"):
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,  # respects API rate limits automatically
        })
        logger.info(f"OHLCVFetcher initialized with exchange: {exchange_id}")

    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> int:
        """
        Downloads candles between since and until and saves them to the DB.
        Returns the number of candles saved.
        """
        until = until or datetime.now(timezone.utc)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)

        total_saved = 0
        current_since = since_ms

        logger.info(f"Starting download {symbol} {timeframe} from {since.date()} to {until.date()}")

        while current_since < until_ms:
            raw = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000,  # max per request at Binance
            )

            if not raw:
                break

            candles = []
            for row in raw:
                ts_ms, open_, high, low, close, volume = row
                if ts_ms >= until_ms:
                    break
                try:
                    candle = Candle(
                        exchange=self.exchange.id,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                    )
                    candles.append(candle)
                except Exception as e:
                    logger.warning(f"Invalid candle discarded: {e}")

            saved = self._save(candles)
            total_saved += saved
            current_since = raw[-1][0] + 1
            logger.info(f"  {saved} candles saved until {datetime.fromtimestamp(raw[-1][0]/1000, tz=timezone.utc).date()}")

        logger.info(f"Download completed: {total_saved} total candles")
        return total_saved

    def _save(self, candles: list[Candle]) -> int:
        """
        Saves a list of candles to the DB, skipping duplicates.
        Uses a single batch query to check existing timestamps (not N+1).
        """
        if not candles:
            return 0

        session: Session = SessionLocal()
        try:
            # Load existing timestamps in one query (batch lookup, not per-candle).
            # All candles in a batch share the same exchange/symbol/timeframe.
            exchange  = candles[0].exchange
            symbol    = candles[0].symbol
            timeframe = candles[0].timeframe
            ts_list   = [c.timestamp for c in candles]

            existing = {
                row.timestamp
                for row in session.query(CandleDB.timestamp)
                .filter(
                    CandleDB.exchange  == exchange,
                    CandleDB.symbol    == symbol,
                    CandleDB.timeframe == timeframe,
                    CandleDB.timestamp.in_(ts_list),
                )
                .all()
            }

            new_records = [
                CandleDB(
                    exchange=c.exchange,
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    timestamp=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                )
                for c in candles
                if c.timestamp not in existing
            ]

            if new_records:
                session.add_all(new_records)
                session.commit()
            return len(new_records)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_last_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Returns the timestamp of the last candle in the DB."""
        session = SessionLocal()
        try:
            from sqlalchemy import func
            result = session.query(func.max(CandleDB.timestamp)).filter_by(
                symbol=symbol,
                timeframe=timeframe,
            ).scalar()
            return result
        finally:
            session.close()

    def update(self, symbol: str, timeframe: str) -> int:
        """
        Downloads only new candles from the last one in the DB until now.
        Idempotent — can be run multiple times without duplicates.

        The upper bound is set to the START of the current (still-open) candle
        period so that incomplete candles are never downloaded. A candle is only
        valid once its full period has elapsed.
        """
        last = self.get_last_timestamp(symbol=symbol, timeframe=timeframe)

        if last is None:
            logger.warning(f"No data for {symbol} {timeframe}, downloading from 2019")
            since = datetime(2019, 1, 1, tzinfo=timezone.utc)
        else:
            since = last
            logger.info(f"Updating {symbol} {timeframe} from {since.date()}")

        # Exclude the currently-forming (incomplete) candle.
        # Floor `now` to the start of the current candle period.
        now = datetime.now(timezone.utc)
        tf_secs = _TF_SECONDS.get(timeframe, 3600)
        until = datetime.fromtimestamp(
            (int(now.timestamp()) // tf_secs) * tf_secs,
            tz=timezone.utc,
        )

        return self.fetch_and_store(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            until=until,
        )
