# data/sources/blockchain.py
"""
Descàrrega i emmagatzematge de mètriques diàries de la xarxa Bitcoin
via Blockchain.com Charts API (gratuïta, sense API key).

Mètriques suportades (definides a core.models.BLOCKCHAIN_METRICS):
  - hash-rate          → potència de mineria (TH/s)
  - n-unique-addresses → adreces actives úniques (count/dia)
  - transaction-fees   → comissions totals diàries (BTC)

Totes les mètriques es guarden a la mateixa taula `blockchain_metrics`
amb una columna `metric` com a discriminador. Afegir noves mètriques
no requereix canvi d'esquema.

API docs: https://www.blockchain.com/explorer/api/charts_api
"""
import logging
import requests
from datetime import datetime, timezone
from sqlalchemy import func
from sqlalchemy.orm import Session
from core.models import BlockchainMetricEntry, BLOCKCHAIN_METRICS
from core.db.models import BlockchainMetricDB
from core.db.session import SessionLocal

logger = logging.getLogger(__name__)

_API_BASE = "https://api.blockchain.info/charts"
_DEFAULT_METRICS = list(BLOCKCHAIN_METRICS)  # totes les mètriques definides
_REQUEST_TIMEOUT = 30  # segons


class BlockchainFetcher:
    """
    Descarrega mètriques diàries de la xarxa Bitcoin des de Blockchain.com Charts
    i les persiteix a la taula `blockchain_metrics`.
    """

    def fetch_and_store(
        self,
        metric: str,
        timespan: str = "all",
    ) -> int:
        """
        Descarrega dades d'una mètrica concreta per al període indicat.
        timespan: 'all', '5days', '30days', '1years', etc.
                  (veure docs de Blockchain.com Charts API)
        Retorna el nombre de registres nous guardats.
        """
        url = f"{_API_BASE}/{metric}"
        params = {
            "format": "json",
            "timespan": timespan,
            "sampled": "false",  # dades crues sense mostreig
        }

        logger.info(f"Fetching blockchain metric '{metric}' (timespan={timespan})")

        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error en la crida a Blockchain.com per '{metric}': {e}") from e

        payload = response.json()

        if payload.get("status") != "ok":
            raise RuntimeError(
                f"Blockchain.com API retorna error per '{metric}': {payload}"
            )

        raw_values = payload.get("values", [])
        if not raw_values:
            logger.warning(f"Blockchain.com no ha retornat dades per '{metric}'")
            return 0

        entries: list[BlockchainMetricEntry] = []
        for point in raw_values:
            try:
                entry = BlockchainMetricEntry(
                    metric=metric,
                    timestamp=datetime.fromtimestamp(
                        int(point["x"]), tz=timezone.utc
                    ),
                    value=float(point["y"]),
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(
                    f"Punt de dades de '{metric}' descartat: {e} — raw={point}"
                )

        saved = self._save(entries)
        logger.info(
            f"'{metric}': {saved} nous registres guardats ({len(entries)} rebuts)"
        )
        return saved

    def _save(self, entries: list[BlockchainMetricEntry]) -> int:
        """Guarda entrades a la BD, ignorant duplicats (idempotent)."""
        if not entries:
            return 0

        session: Session = SessionLocal()
        try:
            # Carrega tots els timestamps existents per aquesta mètrica d'un sol cop
            existing_ts = {
                row[0]
                for row in session.query(BlockchainMetricDB.timestamp)
                .filter(BlockchainMetricDB.metric == entries[0].metric)
                .all()
            }

            to_insert = [
                BlockchainMetricDB(
                    metric=e.metric,
                    timestamp=e.timestamp,
                    value=e.value,
                )
                for e in entries
                if e.timestamp not in existing_ts
            ]

            if to_insert:
                session.bulk_save_objects(to_insert)
                session.commit()

            return len(to_insert)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_last_timestamp(self, metric: str) -> datetime | None:
        """Retorna el timestamp del darrer registre emmagatzemat per a la mètrica."""
        session = SessionLocal()
        try:
            return (
                session.query(func.max(BlockchainMetricDB.timestamp))
                .filter(BlockchainMetricDB.metric == metric)
                .scalar()
            )
        finally:
            session.close()

    def update(self, metric: str) -> int:
        """
        Actualització incremental d'una mètrica: descarrega els darrers 5 dies.
        Segur per a crons diaris — la deduplicació gestiona els solapaments.
        Si la BD és buida, descarrega tot l'historial disponible.
        Retorna el nombre de nous registres guardats.
        """
        last = self.get_last_timestamp(metric)

        if last is None:
            logger.info(
                f"No hi ha dades de '{metric}' a la BD, descàrrega completa"
            )
            return self.fetch_and_store(metric, timespan="all")

        logger.info(f"Update '{metric}' (darrer registre: {last.date()})")
        # Sempre agafem 5 dies de buffer per cobrir possibles endarreriments de l'API
        return self.fetch_and_store(metric, timespan="5days")

    def update_all(self, metrics: list[str] | None = None) -> dict[str, int]:
        """
        Actualitza totes les mètriques (o les indicades).
        Retorna {'metric-name': N_new_records, ...}.
        """
        metrics = metrics or _DEFAULT_METRICS
        results: dict[str, int] = {}
        for metric in metrics:
            try:
                results[metric] = self.update(metric)
            except Exception as e:
                logger.error(f"Error actualitzant '{metric}': {e}")
                results[metric] = 0
        return results

    def fetch_all(self, metrics: list[str] | None = None) -> dict[str, int]:
        """
        Descàrrega inicial de tot l'historial per a totes les mètriques (o les indicades).
        Retorna {'metric-name': N_new_records, ...}.
        """
        metrics = metrics or _DEFAULT_METRICS
        results: dict[str, int] = {}
        for metric in metrics:
            try:
                results[metric] = self.fetch_and_store(metric, timespan="all")
            except Exception as e:
                logger.error(f"Error descarregant '{metric}': {e}")
                results[metric] = 0
        return results
