# data/sources/ohlcv.py
import ccxt
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from core.models import Candle
from core.db.models import CandleDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)


class OHLCVFetcher:
    """
    Descarrega candeles OHLCV d'un exchange via ccxt
    i les persisteix a la DB.
    """

    def __init__(self, exchange_id: str = "binance"):
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,  # respecta els límits de l'API automàticament
        })
        logger.info(f"OHLCVFetcher inicialitzat amb exchange: {exchange_id}")

    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> int:
        """
        Descarrega candles entre since i until i les guarda a la DB.
        Retorna el nombre de candles guardades.
        """
        until = until or datetime.now(timezone.utc)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)

        total_saved = 0
        current_since = since_ms

        logger.info(f"Iniciant descàrrega {symbol} {timeframe} des de {since.date()} fins {until.date()}")

        while current_since < until_ms:
            raw = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000,  # màxim per request a Binance
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
                    logger.warning(f"Candle invàlida descartada: {e}")

            saved = self._save(candles)
            total_saved += saved
            current_since = raw[-1][0] + 1
            logger.info(f"  {saved} candles guardades fins {datetime.fromtimestamp(raw[-1][0]/1000, tz=timezone.utc).date()}")

        logger.info(f"Descàrrega completada: {total_saved} candles totals")
        return total_saved

    def _save(self, candles: list[Candle]) -> int:
        """Guarda una llista de candles a la DB, ignorant duplicats."""
        if not candles:
            return 0

        session: Session = SessionLocal()
        try:
            saved = 0
            for candle in candles:
                exists = session.query(CandleDB).filter_by(
                    exchange=candle.exchange,
                    symbol=candle.symbol,
                    timeframe=candle.timeframe,
                    timestamp=candle.timestamp,
                ).first()

                if not exists:
                    session.add(CandleDB(
                        exchange=candle.exchange,
                        symbol=candle.symbol,
                        timeframe=candle.timeframe,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                    ))
                    saved += 1

            session.commit()
            return saved
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_last_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Retorna el timestamp de l'última candle a la DB."""
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
        Descarrega només les candles noves des de l'última a la DB fins ara.
        És idempotent — es pot executar múltiples vegades sense duplicats.
        """
        last = self.get_last_timestamp(symbol=symbol, timeframe=timeframe)

        if last is None:
            logger.warning(f"No hi ha dades per {symbol} {timeframe}, descarrega des de 2019")
            since = datetime(2019, 1, 1, tzinfo=timezone.utc)
        else:
            since = last
            logger.info(f"Actualitzant {symbol} {timeframe} des de {since.date()}")

        return self.fetch_and_store(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
        )