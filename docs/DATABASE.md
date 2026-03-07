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
| `timeframe` | str(10) | `"1h"`, `"4h"`, `"1d"` |
| `timestamp` | datetime tz | UTC, inici de la candle |
| `open` | float | Preu d'obertura |
| `high` | float | Màxim |
| `low` | float | Mínim |
| `close` | float | Preu de tancament |
| `volume` | float | Volum en BTC |

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

### Historial de preus BTC (últimes 200 candles d'1h)
```sql
SELECT timestamp, open, high, low, close, volume
FROM candles
WHERE symbol = 'BTC/USDT' AND timeframe = '1h'
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

## Taules futures (ROADMAP)

| Taula | Per a | Notes |
|-------|-------|-------|
| `fear_greed` | Fear & Greed Index | `timestamp`, `value` (0-100), `classification` |
| `external_signals` | Qualsevol font externa | Taula genèrica: `source`, `key`, `value`, `timestamp` |
| `backtest_results` | Resultats de backtests | Per evitar recalcular sempre |

---

*Última actualització: Març 2026 · Versió 1.1*
