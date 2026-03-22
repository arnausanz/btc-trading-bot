# Base de Dades — Referència

> PostgreSQL. Connexió configurada a `config/settings.yaml` → `DATABASE_URL`.
> Models SQLAlchemy a `core/db/models.py`. Sessió a `core/db/session.py`.
> Per a la configuració: veure **[04_CONFIGURATION.md](./04_CONFIGURATION.md)**.

---

## Mapa de taules per grup

Les taules estan completament separades: **no hi ha cap FK entre grups**. Pots truncar o esborrar les taules d'un grup sense afectar els altres.

| Grup | Taules | Pots truncar/esborrar? |
|------|--------|----------------------|
| **Dades de mercat** | `candles` | ⚠️ Mai — cal re-descarregar tot des de 2019 |
| **Dades externes lleugeres** | `fear_greed`, `funding_rates` | ✅ Sí — `download_fear_greed.py` / `download_futures.py` recreen tot |
| **Dades on-chain** | `open_interest`, `blockchain_metrics` | ✅ Sí — scripts de re-descàrrega disponibles; si els bots on-chain es descarten, es poden esborrar sense conseqüències |
| **Demo trading** | `demo_ticks`, `demo_trades` | ✅ Sí — via `/reset` Telegram o `reset_demo.py` (tots els bots tornen a `initial_capital`) |
| **Gate System** | `gate_positions`, `gate_near_misses` | ⚠️ `gate_positions` s'esborra automàticament en reset del bot. `gate_near_misses` és un log d'anàlisi — pots esborrar-lo sense impacte operatiu |
| **Legacy (no usades en demo)** | `signals`, `orders`, `trades` | ✅ Sí — infraestructura de live trading futur; no s'usen en paper trading |

---

## Reset d'un bot — Què passa exactament?

Quan fas `/reset bot_id` via Telegram o `python scripts/reset_demo.py bot_id`:

1. **S'esborren** tots els registres de `demo_ticks` per a `bot_id` — el portfolio history desapareix.
2. **S'esborren** tots els registres de `demo_trades` per a `bot_id` — el trade log desapareix.
3. **S'esborren** totes les `gate_positions` obertes per a `bot_id` — evita inconsistència de capital si el Gate bot tenia posicions actives.
4. La propera vegada que el DemoRunner arrenca, `get_last_state()` retorna `None` → el bot **torna a `initial_capital`** (el que declara el seu YAML) i opera com si fos el primer dia.

> ⚠️ **El reset és destructiu i permanent.** No hi ha arxiu ni soft-delete. Si vols conservar les estadístiques passades, fes un backup previ: `pg_dump -t demo_ticks -t demo_trades btc_trading > backup_pre_reset.sql`.

---

## Índexos

| Índex | Taula | Columnes | Per a |
|-------|-------|---------|-------|
| `ix_demo_ticks_bot_id_ts` | `demo_ticks` | `(bot_id, timestamp)` | `get_last_state()`, `get_portfolio_history()` |
| `ix_demo_trades_bot_id_ts` | `demo_trades` | `(bot_id, timestamp)` | `get_trades()` |
| `ix_blockchain_metrics_metric_timestamp` | `blockchain_metrics` | `(metric, timestamp)` | Queries de features on-chain |
| `ix_open_interest_symbol_tf_timestamp` | `open_interest` | `(symbol, timeframe, timestamp)` | Queries de features OI |
| `ix_funding_rates_symbol_timestamp` | `funding_rates` | `(symbol, timestamp)` | Queries de funding rate |

Els índexos de `demo_*` es creen amb la migració `c1d2e3f4a5b6_add_indexes_demo_tables.py`.

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

### `signals` _(legacy — no usada en demo)_
Infraestructura de live trading futur. No s'usa en paper trading — el DemoRunner registra les decisions directament a `demo_ticks` i `demo_trades`.

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

### `orders` _(legacy — no usada en demo)_
Infraestructura de live trading futur. Una ordre per senyal accionable.

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

### `trades` _(legacy — no usada en demo)_
Round-trips completats per live trading: cada parell (ordre d'obertura, ordre de tancament). Té FK cap a `orders`.

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

**Volum:** ~60 registres/hora per bot actiu (1 tick/min). Amb 10 bots → ~600 files/hora → ~14.400/dia → ~5.3M/any.
**Índex:** `ix_demo_ticks_bot_id_ts` — garanteix queries ràpides fins i tot amb milions de files.

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

---

### `gate_positions`
Posicions obertes del GateBot, persistides entre reinicis del DemoRunner.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `bot_id` | str(100) | Ex: `"gate_v1"` |
| `opened_at` | datetime tz | Moment d'obertura de la posició |
| `entry_price` | float | Preu d'entrada |
| `stop_level` | float | Preu del stop loss inicial |
| `target_level` | float | Preu objectiu (take profit) |
| `highest_price` | float | Preu màxim assolit (per trailing stop) |
| `size_usdt` | float | Mida de la posició en USDT |
| `regime` | str(50) | Règim P1 en el moment de l'entrada (ex: `"STRONG_BULL"`) |
| `decel_counter` | int | Nombre de candles 4H consecutives en desacceleració |

**Notes:** S'esborra quan la posició es tanca. Permet que el GateBot recuperi el seu estat complet si el DemoRunner es reinicia. Script de creació: `alembic upgrade head`.

---

### `gate_near_misses`
Registre de cada avaluació on P1 + P2 + P3 passen simultàniament. Permet analitzar quines portes bloquegen més trades i per quin motiu.

| Columna | Tipus | Descripció |
|---------|-------|-----------|
| `id` | int PK | Autoincrement |
| `timestamp` | datetime tz | Moment de l'avaluació |
| `bot_id` | str(100) | Ex: `"gate_v1"` |
| `p1_regime` | str(50) | Règim detectat (`STRONG_BULL`, etc.) |
| `p1_confidence` | float | Probabilitat màxima del règim [0–1] |
| `p2_multiplier` | float | Position multiplier de P2 [0–1] |
| `p3_level_type` | str(50) | Tipus de nivell (`fractal`, `fibonacci`, `vp`, `confluence`) |
| `p3_strength` | float | Força del nivell [0–1] |
| `p3_risk_reward` | float | R:R del nivell |
| `p3_volume_ratio` | float | Volum d'acostament / vol_ma20 (filtre: <0.8 = tanca) |
| `p4_d1_ok` | bool | Senyal 1 P4 actiu |
| `p4_d2_ok` | bool | Senyal 2 P4 actiu |
| `p4_rsi_ok` | bool | Senyal 3 P4 actiu (RSI-2) |
| `p4_macd_ok` | bool | Senyal 4 P4 actiu (MACD) |
| `p4_score` | float | Score P4 [0–1] |
| `p4_triggered` | bool | Si P4 va obrir porta |
| `p5_veto_reason` | str(200) | Motiu de veto P5 (null si no hi ha veto) |
| `p5_position_size` | float | Mida calculada per P5 (Kelly) |
| `executed` | bool | Si el trade es va executar finalment |

**Volum estimat:** ~10–30 registres/dia (cada cop que P1+P2+P3 coincideixen a les 4H).

**Consultes d'anàlisi útils:**

```sql
-- Quina porta bloqueja més?
SELECT
  CASE
    WHEN NOT p4_triggered THEN 'P4'
    WHEN p5_veto_reason IS NOT NULL THEN 'P5'
    ELSE 'executat'
  END as bloqueig,
  COUNT(*) as total
FROM gate_near_misses
GROUP BY 1 ORDER BY 2 DESC;

-- P4: quin senyal és el més restrictiu?
SELECT
  SUM(CASE WHEN NOT p4_d1_ok THEN 1 ELSE 0 END) as d1_falla,
  SUM(CASE WHEN NOT p4_d2_ok THEN 1 ELSE 0 END) as d2_falla,
  SUM(CASE WHEN NOT p4_rsi_ok THEN 1 ELSE 0 END) as rsi_falla,
  SUM(CASE WHEN NOT p4_macd_ok THEN 1 ELSE 0 END) as macd_falla
FROM gate_near_misses WHERE NOT p4_triggered;

-- Règims que arriben a P3 però no a trade
SELECT p1_regime, COUNT(*) as near_misses,
       SUM(executed::int) as executats,
       AVG(p3_volume_ratio) as avg_vol_ratio
FROM gate_near_misses
GROUP BY p1_regime ORDER BY near_misses DESC;
```

---

## Taules futures (ROADMAP)

| Taula | Per a | Notes |
|-------|-------|-------|
| `backtest_results` | Resultats de backtests | Per evitar recalcular sempre |

---

*Última actualització: Març 2026 · Versió 3.0 (Mapa de grups + reset documentat + índexos demo_ticks/trades + legacy signals/orders/trades)*
