# scripts/check_data_completeness.py
"""
Data completeness checker.

Connects to the DB and reports on the state of all data sources:
  - candles (1h, 4h, 1d)
  - fear_greed (daily)
  - funding_rates (every 8h)
  - open_interest (1h from REST + 5m from Vision)
  - blockchain_metrics (daily, per metric)

For each source it checks:
  - Total records
  - Date range (first → last)
  - Coverage vs the training + test periods defined in settings.yaml
  - Gaps (consecutive records with a larger-than-expected gap)

Exit code: 0 if all critical checks pass, 1 if any critical gap found.

Usage:
    python scripts/check_data_completeness.py
    python scripts/check_data_completeness.py --verbose    # also lists gap details
    python scripts/check_data_completeness.py --from 2023-01-01  # check from custom date
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone, date
from typing import Optional

sys.path.append(".")

import yaml
from sqlalchemy import text, func
from sqlalchemy.orm import Session

from core.db.session import SessionLocal
from core.db.models import (
    CandleDB, FearGreedDB, FundingRateDB, OpenInterestDB, BlockchainMetricDB
)
from core.models import BLOCKCHAIN_METRICS

# ─── Colours / symbols ────────────────────────────────────────────────────────
OK   = "✅"
WARN = "⚠️ "
FAIL = "❌"
INFO = "ℹ️ "

# ─── Training / test period ───────────────────────────────────────────────────
def _load_periods() -> tuple[date, date]:
    try:
        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)
        bt = cfg.get("backtesting", {})
        train_until = date.fromisoformat(bt["train_until"])
        test_from   = date.fromisoformat(bt["test_from"])
        return train_until, test_from
    except Exception:
        return date(2024, 12, 31), date(2025, 1, 1)


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_date(dt: Optional[datetime]) -> Optional[date]:
    return dt.date() if dt else None


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "n/a"


def _gap_label(gap: timedelta) -> str:
    h = gap.total_seconds() / 3600
    if h < 48:
        return f"{h:.0f}h"
    return f"{gap.days}d"


def _find_gaps(
    session: Session,
    table,
    timestamp_col,
    expected_delta: timedelta,
    filters: list = None,
    max_gaps_to_show: int = 5,
    verbose: bool = False,
) -> list[tuple[datetime, datetime, timedelta]]:
    """
    Returns a list of (gap_start, gap_end, gap_size) tuples where
    gap_size > expected_delta * 1.5 (to tolerate minor timing jitter).
    """
    q = session.query(timestamp_col)
    if filters:
        for f in filters:
            q = q.filter(f)
    timestamps = [row[0] for row in q.order_by(timestamp_col).all()]

    if len(timestamps) < 2:
        return []

    threshold = expected_delta * 1.5
    gaps = []
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i - 1]
        if diff > threshold:
            gaps.append((timestamps[i - 1], timestamps[i], diff))

    return gaps


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _row(status: str, label: str, detail: str) -> None:
    print(f"  {status}  {label:<38} {detail}")


# ─── Per-source checks ────────────────────────────────────────────────────────

def check_candles(session: Session, train_until: date, test_from: date,
                  verbose: bool, issues: list) -> None:
    _section("Candles (OHLCV)")

    symbol = "BTC/USDT"
    timeframe_deltas = {"1h": timedelta(hours=1), "4h": timedelta(hours=4),
                        "1d": timedelta(days=1)}

    for tf, expected_delta in timeframe_deltas.items():
        count = (session.query(func.count(CandleDB.id))
                 .filter(CandleDB.symbol == symbol, CandleDB.timeframe == tf)
                 .scalar()) or 0

        first = (session.query(func.min(CandleDB.timestamp))
                 .filter(CandleDB.symbol == symbol, CandleDB.timeframe == tf)
                 .scalar())
        last = (session.query(func.max(CandleDB.timestamp))
                .filter(CandleDB.symbol == symbol, CandleDB.timeframe == tf)
                .scalar())

        if count == 0:
            _row(FAIL, f"candles {tf}", "NO DATA")
            issues.append(f"CRITICAL: no candles for {tf}")
            continue

        # Coverage check
        train_ok = first and _as_date(first) <= test_from
        test_ok  = last  and _as_date(last)  >= test_from

        # Expected count (approximate, crypto 24/7)
        if first and last:
            span = (last - first).total_seconds()
            period_secs = expected_delta.total_seconds()
            expected = int(span / period_secs) + 1
            coverage = count / expected * 100
        else:
            expected = coverage = 0

        status = OK if (train_ok and test_ok and coverage > 98) else WARN
        detail = (f"{count:,} records | "
                  f"{_as_date(first)} → {_as_date(last)} | "
                  f"~{coverage:.0f}% coverage")
        _row(status, f"candles {tf}", detail)

        if not train_ok:
            issues.append(f"candles {tf}: first record {_as_date(first)} is after test_from {test_from}")
        if not test_ok:
            issues.append(f"candles {tf}: last record {_as_date(last)} doesn't reach test period")
        if coverage < 95:
            issues.append(f"candles {tf}: coverage only {coverage:.0f}% (expected ~{expected:,} records)")

        # Gap detection
        gaps = _find_gaps(
            session, CandleDB, CandleDB.timestamp, expected_delta,
            filters=[CandleDB.symbol == symbol, CandleDB.timeframe == tf],
            verbose=verbose,
        )
        if gaps:
            _row(WARN, f"  gaps in {tf}", f"{len(gaps)} gap(s) found")
            issues.append(f"candles {tf}: {len(gaps)} gap(s) detected")
            if verbose:
                for start, end, size in gaps[:5]:
                    print(f"       ↳ {_as_date(start)} → {_as_date(end)}  ({_gap_label(size)})")
        else:
            _row(OK, f"  gaps in {tf}", "none")


def check_fear_greed(session: Session, train_until: date, test_from: date,
                     verbose: bool, issues: list) -> None:
    _section("Fear & Greed Index (alternative.me, daily)")

    count = session.query(func.count(FearGreedDB.id)).scalar() or 0
    first = session.query(func.min(FearGreedDB.timestamp)).scalar()
    last  = session.query(func.max(FearGreedDB.timestamp)).scalar()

    if count == 0:
        _row(FAIL, "fear_greed", "NO DATA — run: python scripts/download_fear_greed.py")
        issues.append("CRITICAL: no Fear & Greed data")
        return

    # Fear & Greed started 2018-02-01
    source_start = date(2018, 2, 1)
    train_ok = first and _as_date(first) <= source_start + timedelta(days=30)
    test_ok  = last  and _as_date(last)  >= test_from

    # Expected: 1 per day
    if first and last:
        expected = (last.date() - first.date()).days + 1
        coverage = count / expected * 100
    else:
        expected = coverage = 0

    status = OK if (train_ok and test_ok and coverage > 98) else WARN
    _row(status, "fear_greed",
         f"{count:,} records | {_as_date(first)} → {_as_date(last)} | ~{coverage:.0f}% coverage")

    if not test_ok:
        issues.append(f"fear_greed: last record {_as_date(last)}, doesn't reach test period {test_from}")

    gaps = _find_gaps(session, FearGreedDB, FearGreedDB.timestamp, timedelta(days=1),
                      verbose=verbose)
    if gaps:
        _row(WARN, "  gaps", f"{len(gaps)} gap(s) > 1 day")
        issues.append(f"fear_greed: {len(gaps)} gap(s)")
        if verbose:
            for start, end, size in gaps[:5]:
                print(f"       ↳ {_as_date(start)} → {_as_date(end)}  ({_gap_label(size)})")
    else:
        _row(OK, "  gaps", "none")


def check_funding_rates(session: Session, train_until: date, test_from: date,
                        verbose: bool, issues: list) -> None:
    _section("Funding Rates (Binance USDT-M, every 8h)")

    symbol = "BTC/USDT:USDT"
    count = (session.query(func.count(FundingRateDB.id))
             .filter(FundingRateDB.symbol == symbol)
             .scalar()) or 0
    first = (session.query(func.min(FundingRateDB.timestamp))
             .filter(FundingRateDB.symbol == symbol).scalar())
    last  = (session.query(func.max(FundingRateDB.timestamp))
             .filter(FundingRateDB.symbol == symbol).scalar())

    if count == 0:
        _row(FAIL, "funding_rates", "NO DATA — run: python scripts/download_futures.py")
        issues.append("CRITICAL: no funding rate data")
        return

    # Funding rates available since 2019-09-13
    source_start = date(2019, 9, 13)
    train_ok = first and _as_date(first) <= source_start + timedelta(days=30)
    test_ok  = last  and _as_date(last)  >= test_from

    if first and last:
        expected = int((last - first).total_seconds() / (8 * 3600)) + 1
        coverage = count / expected * 100
    else:
        expected = coverage = 0

    status = OK if (train_ok and test_ok and coverage > 97) else WARN
    _row(status, "funding_rates",
         f"{count:,} records | {_as_date(first)} → {_as_date(last)} | ~{coverage:.0f}% coverage")

    if not test_ok:
        issues.append(f"funding_rates: last record {_as_date(last)}, doesn't reach test period")

    gaps = _find_gaps(
        session, FundingRateDB, FundingRateDB.timestamp, timedelta(hours=8),
        filters=[FundingRateDB.symbol == symbol], verbose=verbose,
    )
    if gaps:
        _row(WARN, "  gaps", f"{len(gaps)} gap(s) > 12h")
        issues.append(f"funding_rates: {len(gaps)} gap(s)")
        if verbose:
            for start, end, size in gaps[:5]:
                print(f"       ↳ {_as_date(start)} → {_as_date(end)}  ({_gap_label(size)})")
    else:
        _row(OK, "  gaps", "none")


def check_open_interest(session: Session, train_until: date, test_from: date,
                        verbose: bool, issues: list) -> None:
    _section("Open Interest (Binance USDT-M)")

    symbol = "BTC/USDT:USDT"
    tf_checks = [
        ("1h",  timedelta(hours=1),  "REST API — rolling 30d only"),
        ("5m",  timedelta(minutes=5), "Vision S3 — since 2021-12-01"),
    ]

    for tf, expected_delta, note in tf_checks:
        count = (session.query(func.count(OpenInterestDB.id))
                 .filter(OpenInterestDB.symbol == symbol,
                         OpenInterestDB.timeframe == tf)
                 .scalar()) or 0
        first = (session.query(func.min(OpenInterestDB.timestamp))
                 .filter(OpenInterestDB.symbol == symbol,
                         OpenInterestDB.timeframe == tf)
                 .scalar())
        last  = (session.query(func.max(OpenInterestDB.timestamp))
                 .filter(OpenInterestDB.symbol == symbol,
                         OpenInterestDB.timeframe == tf)
                 .scalar())

        if count == 0:
            status = WARN if tf == "1h" else FAIL
            _row(status, f"open_interest {tf}", f"NO DATA  [{note}]")
            if tf == "5m":
                issues.append("CRITICAL: no Vision OI (5m) — run: python scripts/download_binance_vision.py")
            continue

        if first and last:
            period_secs = expected_delta.total_seconds()
            expected = int((last - first).total_seconds() / period_secs) + 1
            coverage = count / expected * 100
        else:
            expected = coverage = 0

        # For 5m (Vision), check coverage since 2021-12-01
        if tf == "5m":
            vision_start = date(2021, 12, 1)
            history_ok = first and _as_date(first) <= vision_start + timedelta(days=5)
            status = OK if (history_ok and coverage > 97) else WARN
        else:
            # 1h OI is rolling 30 days, don't penalise for lack of history
            status = OK if coverage > 90 else WARN

        _row(status, f"open_interest {tf}",
             f"{count:,} records | {_as_date(first)} → {_as_date(last)} | ~{coverage:.0f}%  [{note}]")

        if coverage < 90:
            issues.append(f"open_interest {tf}: coverage {coverage:.0f}% is low")

        gaps = _find_gaps(
            session, OpenInterestDB, OpenInterestDB.timestamp, expected_delta,
            filters=[OpenInterestDB.symbol == symbol,
                     OpenInterestDB.timeframe == tf],
            verbose=verbose,
        )
        if gaps:
            _row(WARN, f"  gaps in OI {tf}", f"{len(gaps)} gap(s)")
            if verbose:
                for start, end, size in gaps[:5]:
                    print(f"       ↳ {_as_date(start)} → {_as_date(end)}  ({_gap_label(size)})")
        else:
            _row(OK, f"  gaps in OI {tf}", "none")

    # Extra: days with incomplete 5m data (should be 288/day)
    _check_oi_vision_daily_completeness(session, symbol, verbose, issues)


def _check_oi_vision_daily_completeness(session: Session, symbol: str,
                                         verbose: bool, issues: list) -> None:
    """Checks how many days have fewer than 288 records (5m OI from Vision)."""
    result = session.execute(text("""
        SELECT
            DATE(timestamp AT TIME ZONE 'UTC') AS day,
            COUNT(*) AS cnt
        FROM open_interest
        WHERE symbol = :sym AND timeframe = '5m'
        GROUP BY day
        HAVING COUNT(*) < 280
        ORDER BY day
    """), {"sym": symbol}).fetchall()

    if not result:
        _row(OK, "  Vision 5m daily completeness", "all days have ≥280 records")
    else:
        _row(WARN, "  Vision 5m daily completeness",
             f"{len(result)} day(s) with <280 records (expected 288)")
        issues.append(f"OI Vision 5m: {len(result)} incomplete day(s)")
        if verbose:
            for row in result[:10]:
                print(f"       ↳ {row[0]}: {row[1]}/288 records")


def check_blockchain_metrics(session: Session, train_until: date, test_from: date,
                               verbose: bool, issues: list) -> None:
    _section("Blockchain Metrics (Blockchain.com, daily)")

    for metric in sorted(BLOCKCHAIN_METRICS):
        count = (session.query(func.count(BlockchainMetricDB.id))
                 .filter(BlockchainMetricDB.metric == metric)
                 .scalar()) or 0
        first = (session.query(func.min(BlockchainMetricDB.timestamp))
                 .filter(BlockchainMetricDB.metric == metric)
                 .scalar())
        last  = (session.query(func.max(BlockchainMetricDB.timestamp))
                 .filter(BlockchainMetricDB.metric == metric)
                 .scalar())

        if count == 0:
            _row(FAIL, f"blockchain/{metric}", "NO DATA — run: python scripts/download_blockchain.py")
            issues.append(f"CRITICAL: no blockchain data for '{metric}'")
            continue

        test_ok = last and _as_date(last) >= test_from
        if first and last:
            expected = (last.date() - first.date()).days + 1
            coverage = count / expected * 100
        else:
            expected = coverage = 0

        status = OK if (test_ok and coverage > 95) else WARN
        _row(status, f"blockchain/{metric}",
             f"{count:,} records | {_as_date(first)} → {_as_date(last)} | ~{coverage:.0f}%")

        if not test_ok:
            issues.append(f"blockchain/{metric}: last record {_as_date(last)} doesn't reach test period")

        gaps = _find_gaps(
            session, BlockchainMetricDB, BlockchainMetricDB.timestamp,
            timedelta(days=1),
            filters=[BlockchainMetricDB.metric == metric],
            verbose=verbose,
        )
        if gaps:
            _row(WARN, f"  gaps in {metric}", f"{len(gaps)} gap(s) > 1.5 days")
            issues.append(f"blockchain/{metric}: {len(gaps)} gap(s)")
            if verbose:
                for start, end, size in gaps[:5]:
                    print(f"       ↳ {_as_date(start)} → {_as_date(end)}  ({_gap_label(size)})")
        else:
            _row(OK, f"  gaps in {metric}", "none")


def check_cross_coverage(session: Session, train_until: date, test_from: date,
                          issues: list) -> None:
    """
    Cross-check: do all external data sources cover the SAME period as candles?
    The key concern is that if one source starts later or ends earlier, some
    training samples will have NaN features for that source.
    """
    _section("Cross-coverage check (vs candles 1h)")

    # Candle range is the reference
    candle_first = session.query(func.min(CandleDB.timestamp)).filter(
        CandleDB.symbol == "BTC/USDT", CandleDB.timeframe == "1h").scalar()
    candle_last  = session.query(func.max(CandleDB.timestamp)).filter(
        CandleDB.symbol == "BTC/USDT", CandleDB.timeframe == "1h").scalar()

    if not candle_first:
        _row(WARN, "cross-coverage", "no candle reference — skipping")
        return

    checks = {
        "candles 1h (reference)":   (candle_first, candle_last),
        "fear_greed":               (
            session.query(func.min(FearGreedDB.timestamp)).scalar(),
            session.query(func.max(FearGreedDB.timestamp)).scalar(),
        ),
        "funding_rates":            (
            session.query(func.min(FundingRateDB.timestamp)).filter(
                FundingRateDB.symbol == "BTC/USDT:USDT").scalar(),
            session.query(func.max(FundingRateDB.timestamp)).filter(
                FundingRateDB.symbol == "BTC/USDT:USDT").scalar(),
        ),
        "open_interest 5m (Vision)":(
            session.query(func.min(OpenInterestDB.timestamp)).filter(
                OpenInterestDB.timeframe == "5m").scalar(),
            session.query(func.max(OpenInterestDB.timestamp)).filter(
                OpenInterestDB.timeframe == "5m").scalar(),
        ),
        "blockchain/hash-rate":     (
            session.query(func.min(BlockchainMetricDB.timestamp)).filter(
                BlockchainMetricDB.metric == "hash-rate").scalar(),
            session.query(func.max(BlockchainMetricDB.timestamp)).filter(
                BlockchainMetricDB.metric == "hash-rate").scalar(),
        ),
    }

    for name, (first, last) in checks.items():
        if not first:
            _row(WARN, f"  {name}", "no data")
            continue

        starts_before_train = _as_date(first) <= date(2020, 1, 1)  # well before
        ends_after_test     = last and _as_date(last) >= test_from

        status = OK if (starts_before_train and ends_after_test) else WARN
        note = ""
        if not starts_before_train:
            note = f"⟵ starts {_as_date(first)} (candles start {_as_date(candle_first)})"
        if not ends_after_test:
            note += f" ends {_as_date(last)}"

        _row(status, f"  {name}",
             f"{_as_date(first)} → {_as_date(last)}  {note}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Data completeness checker")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show individual gap details")
    args = parser.parse_args()

    train_until, test_from = _load_periods()

    print("=" * 60)
    print("  DATA COMPLETENESS REPORT")
    print(f"  Training period: until {train_until}")
    print(f"  Test period:     from  {test_from}")
    print(f"  Run at: {_utc_now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    issues: list[str] = []
    session: Session = SessionLocal()

    try:
        check_candles(session, train_until, test_from, args.verbose, issues)
        check_fear_greed(session, train_until, test_from, args.verbose, issues)
        check_funding_rates(session, train_until, test_from, args.verbose, issues)
        check_open_interest(session, train_until, test_from, args.verbose, issues)
        check_blockchain_metrics(session, train_until, test_from, args.verbose, issues)
        check_cross_coverage(session, train_until, test_from, issues)
    finally:
        session.close()

    # ─── Summary ──────────────────────────────────────────────────────────────
    _section("Summary")
    if not issues:
        print(f"  {OK}  All checks passed. Data looks complete.\n")
        return 0
    else:
        critical = [i for i in issues if i.startswith("CRITICAL")]
        warnings = [i for i in issues if not i.startswith("CRITICAL")]

        if critical:
            print(f"  {FAIL}  {len(critical)} critical issue(s):")
            for i in critical:
                print(f"       • {i}")
        if warnings:
            print(f"  {WARN}  {len(warnings)} warning(s):")
            for i in warnings:
                print(f"       • {i}")
        print()

        return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
