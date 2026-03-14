# Base de Dades — Referència

> PostgreSQL. Connexió configurada a `config/settings.yaml` → `DATABASE_URL`.
> Models SQLAlchemy a `core/db/models.py`. Sessió a `core/db/session.py`.

---

## Taules

### `candles`
Dades OHLCV descarregades de Binance. Font principal de tot el sistema.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `exchange` | str(50) | Ex: `"binance"` |
| `symbol` | str(20) | Ex: `"BTC/USDT"` |
| `timeframe` | str(10) | `"1h"`, `"4h"`, `"12h"`, `"1d"` |
| `timestamp` | datetime tz | UTC, inici de la candle |
| `open` | float | Preu d'obertura |
| `high` | float | Màxim |
| `low` | float | Mínim |
| `close` | float | Preu de tancament |
| `volume` | float | Volum en BTC |

**Timeframes disponibles:** `1h` (baseline), `4h`, `12h` (professional RL), `1d`. Tots descarregats des de 2019 amb `python scripts/download_data.py`.

**Index recomanat:** `(symbol, timeframe, timestamp)` — és la consulta més freqüent.

---

### `signals`
Senyals generats pels bots durant el backtesting. Persistits per auditoria.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `bot_id` | str(100) | Ex: `"trend_bot_v1"` |
| `timestamp` | datetime tz | Moment del senyal |
| `action` | str(20) | `"buy"`, `"sell"`, `"hold"`, o float `"-0.7"` (RL continu) |
| `size` | float | Fracció del capital [0.0 – 1.0] |
| `confidence` | float | Confiança del bot [0.0 – 1.0] |
| `reason` | str(500) | Text explicatiu del senyal |

---

### `orders`
Ordres enviades a l'exchange (paper o real). Una ordre per senyal accionable.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | str(100) PK | UUID generat |
| `signal_id` | str(100) | Referència al senyal origen |
| `exchange` | str(50) | `"paper"` |
| `symbol` | str(20) | `"BTC/USDT"` |
| `side` | enum | `"buy"` o `"sell"` |
| `status` | enum | `"pending"`, `"filled"`, `"cancelled"`, `"failed"` |
| `price_target` | float | Preu sol·licitat |
| `price_filled` | float? | Preu real d'execució (amb slippage) |
| `size` | float | Quantitat en BTC |
| `size_quote` | float? | Quantitat en USDT |
| `fees` | float | Comissions pagades |
| `created_at` | datetime tz | Quan es va crear |
| `filled_at` | datetime? | Quan es va executar |
| `metadata_` | JSON | Informació addicional |

---

### `trades`
Round-trips completats: cada parell (ordre d'obertura, ordre de tancament).

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | str(100) PK | UUID generat |
| `order_open_id` | str FK → orders | L'ordre de compra |
| `order_close_id` | str FK → orders | L'ordre de venda |
| `symbol` | str(20) | `"BTC/USDT"` |
| `pnl_realized` | float | PnL en USDT |
| `pnl_pct` | float | PnL en % |
| `duration_seconds` | float | Temps entre compra i venda |
| `metadata_` | JSON | Informació addicional |

---

### `demo_ticks`
Registre de cada tick del DemoRunner (cada ~60s per bot actiu).

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `bot_id` | str(100) | Identificador del bot |
| `timestamp` | datetime tz | Moment del tick |
| `price` | float | Preu BTC en USDT |
| `action` | str(20) | Acció presa (`buy`/`sell`/`hold`) |
| `portfolio_value` | float | Valor total en USDT |
| `usdt_balance` | float | Saldo USDT |
| `btc_balance` | float | Saldo BTC |
| `reason` | str(500) | Explicació de la decisió |

**Volum:** ~60 registres/hora per bot actiu. Amb 6 bots → ~360 files/hora → ~8.600/dia.

---

### `demo_trades`
Trades executats durant el Demo (compres i vendes reals).

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `bot_id` | str(100) | Identificador del bot |
| `timestamp` | datetime tz | Moment del trade |
| `action` | str(10) | `"buy"` o `"sell"` |
| `price` | float | Preu d'execució |
| `size_btc` | float | Quantitat en BTC |
| `size_usdt` | float | Quantitat en USDT |
| `fees` | float | Comissions pagades |
| `portfolio_value` | float | Valor del portfolio post-trade |
| `reason` | str(500) | Motiu del trade |

---

## Consultes útils

### Estat actual del portfolio per bot
```sql
SELECT bot_id,
       price,
       portfolio_value,
       usdt_balance,
       btc_balance,
       timestamp
FROM demo_ticks
WHERE timestamp = (SELECT MAX(timestamp) FROM demo_ticks dt2 WHERE dt2.bot_id = demo_ticks.bot_id)
ORDER BY portfolio_value DESC;
```

### PnL acumulat per bot
```sql
SELECT bot_id,
       COUNT(*) as total_trades,
       SUM(CASE WHEN action = 'sell' THEN size_usdt - fees ELSE -(size_usdt + fees) END) as pnl_usdt
FROM demo_trades
GROUP BY bot_id
ORDER BY pnl_usdt DESC;
```

### Evolució del portfolio d'un bot (per gràfic)
```sql
SELECT timestamp, portfolio_value
FROM demo_ticks
WHERE bot_id = 'trend_bot_v1'
ORDER BY timestamp;
```

### Historial de preus BTC (últimes 200 candles d'1h o 12h)
```sql
SELECT timestamp, open, high, low, close, volume
FROM candles
WHERE symbol = 'BTC/USDT' AND timeframe = '12h'   -- o '1h'
ORDER BY timestamp DESC
LIMIT 200;
```

### Rang de dades disponibles per timeframe
```sql
SELECT timeframe, COUNT(*) as total, MIN(timestamp) as des_de, MAX(timestamp) as fins
FROM candles
WHERE symbol = 'BTC/USDT'
GROUP BY timeframe
ORDER BY timeframe;
```

### Trades recents d'un bot
```sql
SELECT timestamp, action, price, size_btc, size_usdt, fees, reason
FROM demo_trades
WHERE bot_id = 'trend_bot_v1'
ORDER BY timestamp DESC
LIMIT 20;
```

### Drawdown màxim del demo (per bot)
```sql
WITH portfolio AS (
    SELECT bot_id, timestamp, portfolio_value,
           MAX(portfolio_value) OVER (PARTITION BY bot_id ORDER BY timestamp) as max_val
    FROM demo_ticks
)
SELECT bot_id,
       MIN((portfolio_value - max_val) / max_val * 100) as max_drawdown_pct
FROM portfolio
GROUP BY bot_id
ORDER BY max_drawdown_pct;
```

---

## Accés des del codi

```python
# Sessió directa
from core.db.session import SessionLocal
session = SessionLocal()

# Query candles
from core.db.models import CandleDB
candles = (session.query(CandleDB)
           .filter_by(symbol="BTC/USDT", timeframe="1h")
           .order_by(CandleDB.timestamp.desc())
           .limit(200).all())

# Repository per al demo (abstracció recomanada)
from core.db.demo_repository import DemoRepository
repo = DemoRepository(session)
repo.save_tick(bot_id, timestamp, price, action, portfolio_value, usdt, btc, reason)
```

---

### `fear_greed`
Fear & Greed Index diari d'[alternative.me](https://api.alternative.me/fng/). Disponible des de febrer 2018.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `timestamp` | datetime tz (UNIQUE) | UTC, mignight del dia |
| `value` | int | 0 (Extreme Fear) → 100 (Extreme Greed) |
| `classification` | str(50) | `"Extreme Fear"`, `"Fear"`, `"Neutral"`, `"Greed"`, `"Extreme Greed"` |

**Script de descàrrega inicial:** `python scripts/download_fear_greed.py`
**Script d'update (cron diari):** `python scripts/update_fear_greed.py`

---

### `funding_rates`
Funding rate del contracte perpetu BTC/USDT:USDT de Binance USDT-M. Cada 8h, des de setembre 2019.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `symbol` | str(30) | `"BTC/USDT:USDT"` |
| `timestamp` | datetime tz | UTC, cada 8h (00:00, 08:00, 16:00) |
| `rate` | float | Ex: `0.0001` = 0.01% per 8h; pot ser negatiu |

**Script de descàrrega inicial:** `python scripts/download_futures.py`
**Script d'update (cron horari):** `python scripts/update_futures.py`

---

### `open_interest`
Open interest del contracte perpetu BTC/USDT:USDT. Dues fonts coexisteixen a la mateixa taula diferenciades per `timeframe`:

| `timeframe` | Font | Cobertura | Granularitat |
|-------------|------|-----------|--------------|
| `"5m"` | Binance Vision S3 (bucket públic) | des de 2021-12-01 | 5 minuts (288/dia) |
| `"1h"` | Binance REST API | darrers 30 dies | 1 hora |

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `symbol` | str(30) | `"BTC/USDT:USDT"` |
| `timeframe` | str(10) | `"5m"` o `"1h"` |
| `timestamp` | datetime tz | UTC |
| `open_interest_btc` | float | Quantitat en BTC |
| `open_interest_usdt` | float | Valor en USDT |

**Script de descàrrega inicial (Vision/historial):** `python scripts/download_binance_vision.py`
**Script de descàrrega inicial (REST/30d):** `python scripts/download_futures.py`
**Script d'update (cron diari Vision):** `python scripts/update_binance_vision.py`
**Script d'update (cron horari REST):** `python scripts/update_futures.py`

⚠️ **Nota sobre re-descàrrega:** Si perds les dades de Vision (5m), pots recuperar-les íntegrament amb `download_binance_vision.py` — el script és idempotent (salta dies ja complets). Les dades REST (1h) només cobreixen els últims 30 dies.

---

### `blockchain_metrics`
Mètriques diàries de la xarxa Bitcoin via [Blockchain.com Charts API](https://www.blockchain.com/explorer/api/charts_api). Gratuïta, sense API key. Des de ~2009.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `metric` | str(50) | Discriminador: `"hash-rate"`, `"n-unique-addresses"`, `"transaction-fees"` |
| `timestamp` | datetime tz | UTC, midnight del dia |
| `value` | float | Valor de la mètrica (unitat depèn de la mètrica) |

**Mètriques disponibles:**

| Mètrica | Unitat | Significat |
|---------|--------|-----------|
| `hash-rate` | TH/s | Potència de mineria total de la xarxa |
| `n-unique-addresses` | count/dia | Adreces actives úniques (proxy d'activitat on-chain) |
| `transaction-fees` | BTC/dia | Comissions totals pagades (proxy de demanda de blockspace) |

**Afegir una nova mètrica** no requereix canvi d'esquema: afegeix el nom a `BLOCKCHAIN_METRICS` a `core/models.py` i crida `fetch_and_store(metric, timespan="all")`.

**Script de descàrrega inicial:** `python scripts/download_blockchain.py`
**Script d'update (cron diari):** `python scripts/update_blockchain.py`

---

## Comprovació de completesa

```bash
python scripts/check_data_completeness.py          # resum
python scripts/check_data_completeness.py --verbose # mostra gaps individuals
```

Reporta per a cada font: total de registres, rang de dates, cobertura estimada, gaps i cross-check entre fonts.

---

## Taules futures (ROADMAP)

| Taula | Per a | Notes |
|-------|-------|-------|
| `backtest_results` | Resultats de backtests | Per evitar recalcular sempre |

---

*Última actualització: Març 2026 · Versió 1.3*
