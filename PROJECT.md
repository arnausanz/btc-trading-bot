# BTC Trading Bot Platform — Documentació del Projecte

**Versió:** 1.0  
**Data:** 2026-03-01  
**Stack:** Python 3.12, PostgreSQL/TimescaleDB, MLflow, Optuna, Stable-Baselines3

---

## Índex

1. [Visió General](#1-visió-general)
2. [Estructura de Directoris](#2-estructura-de-directoris)
3. [Arquitectura per Capes](#3-arquitectura-per-capes)
4. [Components Principals](#4-components-principals)
5. [Flux de Dades](#5-flux-de-dades)
6. [Com Afegir Coses Noves](#6-com-afegir-coses-noves)
7. [Base de Dades](#7-base-de-dades)
8. [Configuració YAML](#8-configuració-yaml)
9. [Comandes de Referència](#9-comandes-de-referència)
10. [Variables d'Entorn](#10-variables-dentorn)
11. [Si Vols Canviar de BTC a ETH](#11-si-vols-canviar-de-btc-a-eth)

---

## 1. Visió General

Plataforma de trading algorítmic dissenyada per a demo 24/7 amb paper trading (sense diners reals). L'objectiu és provar rendibilitat de múltiples estratègies abans de considerar un exchange real.

**Principis de disseny:**
- Afegir un bot nou = 1-2 fitxers nous, zero canvis al core
- Tot configurable via YAML, sense hardcoding
- Cada bot té el seu portfolio independent
- Persistència completa: si el sistema cau, es recupera l'estat automàticament

**Modes d'operació:**
- **Backtesting:** simula un bot sobre dades històriques (2022-avui)
- **Demo Runner:** executa bots en temps real amb dades reals i PaperExchange
- **Entrenament:** entrena models ML/RL sobre dades històriques

---

## 2. Estructura de Directoris

```
btc-trading-bot/
│
├── alembic/                    # Migracions de base de dades
│   └── versions/               # Fitxers de migració (no tocar manualment)
│
├── bots/                       # Implementacions de bots
│   ├── classical/              # Bots clàssics (DCA, Trend, Grid, Hold)
│   ├── ml/                     # Bots basats en ML (Random Forest, XGBoost)
│   └── rl/                     # Bots basats en RL (DQN, PPO)
│       ├── rewards/            # Reward functions per a RL
│       └── policies/           # Polítiques custom (reservat)
│
├── config/                     # Tota la configuració YAML
│   ├── bots/                   # Un YAML per bot (dca.yaml, trend.yaml...)
│   ├── exchanges/              # Configuració del PaperExchange
│   ├── training/               # Configuració d'experiments d'entrenament
│   ├── demo.yaml               # Configuració del Demo Runner
│   ├── settings.yaml           # Configuració global
│   └── logging.yaml            # Configuració de logs
│
├── core/                       # Nucli del sistema (no tocar sense entendre)
│   ├── backtesting/            # Motor de backtesting, mètriques, comparador, optimitzador
│   ├── db/                     # Models SQLAlchemy, sessions, repositori demo
│   ├── engine/                 # Runner (backtesting) i DemoRunner (temps real)
│   ├── interfaces/             # Interfícies ABC: BaseBot, BaseExchange, BaseStrategy
│   ├── risk/                   # Gestió de risc (reservat per a futur)
│   ├── config.py               # Constants globals (MLFLOW_TRACKING_URI)
│   └── models.py               # Models Pydantic: Candle, Signal, Order, Trade
│
├── data/                       # Tot el relacionat amb dades
│   ├── feature_store/          # Feature Store (reservat per a futur)
│   ├── observation/            # ObservationBuilder: construeix les dades per a cada bot
│   ├── processing/             # Indicadors tècnics i construcció de datasets ML
│   └── sources/                # OHLCVFetcher: descàrrega de dades de Binance
│
├── docs/                       # Documentació
│   └── decisions/              # ADRs — per què es van prendre decisions clau
│
├── exchanges/                  # Implementacions d'exchanges
│   └── paper.py                # PaperExchange: simulador amb fees i slippage
│
├── models/                     # Models ML/RL entrenats (fitxers .pkl, .zip)
│
├── monitoring/                 # Monitoratge i visualització
│   ├── dashboard.py            # Dashboard Streamlit
│   └── telegram_notifier.py    # Bot de Telegram per a alertes i comandos
│
├── scripts/                    # Scripts executables (punt d'entrada)
│   ├── download_data.py        # Descàrrega inicial de dades històriques
│   ├── update_data.py          # Actualitza dades noves des de l'última candle
│   ├── validate_data.py        # Diagnòstic de la qualitat de les dades
│   ├── run_comparison.py       # Compara tots els bots en backtest
│   ├── optimize_bots.py        # Optuna: cerca els millors paràmetres
│   ├── train_models.py         # Entrena models ML
│   ├── train_rl.py             # Entrena agents RL
│   └── run_demo.py             # Arrenca el Demo Runner
│
└── tests/
    ├── unit/                   # Tests unitaris (no necessiten DB externa)
    ├── integration/            # Tests d'integració (necessiten DB real)
    └── backtests/              # Backtests de referència
```

---

## 3. Arquitectura per Capes

```
┌─────────────────────────────────────────────────────┐
│  SCRIPTS (punt d'entrada)                           │
│  run_demo.py, run_comparison.py, train_models.py... │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  ENGINE                                             │
│  DemoRunner (temps real) / Runner (backtesting)     │
└──────────┬────────────────────────┬─────────────────┘
           │                        │
┌──────────▼──────────┐  ┌──────────▼──────────────────┐
│  BOTS               │  │  EXCHANGES                   │
│  BaseBot            │  │  BaseExchange                │
│  ├── DCABot         │  │  └── PaperExchange           │
│  ├── TrendBot       │  │      (futur: LiveExchange)   │
│  ├── GridBot        │  └──────────────────────────────┘
│  ├── MLBot          │
│  └── RLBot (futur)  │
└──────────┬──────────┘
           │ observation_schema()
┌──────────▼──────────────────────────────────────────┐
│  DATA LAYER                                         │
│  ObservationBuilder → compute_features → DB         │
└─────────────────────────────────────────────────────┘
```

**Flux d'una iteració del Runner:**
1. Runner obté el preu actual (o llegeix la candle del backtest)
2. Crida `bot.observation_schema()` per saber quines dades necessita
3. `ObservationBuilder` construeix l'observació (finestra de N candles amb features)
4. Runner crida `bot.on_observation(obs)` → el bot retorna un `Signal`
5. Runner crida `exchange.send_order(signal)` → l'exchange retorna un `Order`
6. Runner guarda l'historial i actualitza el portfolio

---

## 4. Components Principals

### 4.1 BaseBot (`core/interfaces/base_bot.py`)

Tota estratègia de trading és un bot. Implementa dos mètodes:

```python
def observation_schema(self) -> ObservationSchema:
    # Declara quines dades necessites
    return ObservationSchema(
        features=["close", "rsi_14"],
        timeframes=["1h"],
        lookback=200,  # quantes candles enrere
    )

def on_observation(self, observation: dict) -> Signal:
    # Rep les dades i retorna una decisió
    # observation["1h"]["features"] → DataFrame amb les últimes N candles
    # observation["1h"]["current_price"] → preu actual
    # observation["portfolio"] → {"USDT": 8000, "BTC": 0.05}
    ...
    return Signal(action=Action.BUY, size=0.5, ...)
```

### 4.2 ObservationBuilder (`data/observation/builder.py`)

Construeix l'observació que cada bot necessita. Precarrega les dades a memòria (cache) i serveix finestres de N candles amb features tècniques calculades.

**Cache:** Un cop carregades les dades, es queden en memòria durant tota l'execució. Això és important per al DemoRunner: no fa queries a la DB cada 60 segons, sinó que serveix des de la cache.

**Limitació actual:** La cache no s'actualitza automàticament. El DemoRunner usa sempre les dades carregades a l'inici. Per a trading en temps real, caldria un mecanisme per afegir la candle nova a la cache cada hora.

### 4.3 PaperExchange (`exchanges/paper.py`)

Simula un exchange real. Manté un portfolio virtual i aplica fees (0.1%) i slippage (0.05%) en cada operació.

```python
exchange.set_current_price(67000.0)  # actualitza el preu
order = exchange.send_order(signal)   # executa la ordre
value = exchange.get_portfolio_value() # valor total en USDT
```

### 4.4 BacktestEngine (`core/backtesting/engine.py`)

Executa un backtest complet. Crea un PaperExchange net, executa el Runner sobre totes les candles, calcula mètriques i registra a MLflow.

```python
engine = BacktestEngine(bot=TrendBot())
metrics = engine.run(symbol="BTC/USDT", timeframe="1h")
print(metrics.summary())
```

### 4.5 DemoRunner (`core/engine/demo_runner.py`)

Executa múltiples bots en paral·lel en temps real. Cada bot té el seu propi PaperExchange (portfolios separats). Obté el preu actual de Binance cada 60 segons i processa un tick per a cada bot.

**Persistència:** Guarda cada tick a `demo_ticks` i cada trade a `demo_trades`. En reiniciar, recupera l'últim estat de la DB.

### 4.6 DemoRepository (`core/db/demo_repository.py`)

Capa d'accés a dades per al Demo. Centralitza totes les operacions de lectura i escriptura de `demo_ticks` i `demo_trades`.

### 4.7 TelegramNotifier (`monitoring/telegram_notifier.py`)

Envia notificacions i escolta comandos via un thread background (long polling). Les funcions de status i trades s'injecten des del DemoRunner per evitar acoblament directe.

**Comandos disponibles:** `/status`, `/trades`, `/help`

---

## 5. Flux de Dades

### 5.1 Descàrrega inicial
```
Binance API (ccxt)
    → OHLCVFetcher.fetch_and_store()
    → PostgreSQL (taula candles)
```

### 5.2 Càlcul de features
```
PostgreSQL (taula candles)
    → compute_features()  [data/processing/technical.py]
    → DataFrame amb EMA, RSI, MACD, Bollinger Bands, ATR
    → ObservationBuilder cache (en memòria)
```

### 5.3 Backtesting
```
ObservationBuilder cache
    → Runner.run() (loop sobre totes les candles)
    → Bot.on_observation() → Signal
    → PaperExchange.send_order() → Order
    → BacktestMetrics (Sharpe, Drawdown, etc.)
    → MLflow (registre automàtic)
```

### 5.4 Demo Runner
```
Binance API (preu actual, cada 60s)
    → DemoRunner._process_tick()
    → ObservationBuilder (des de cache)
    → Bot.on_observation() → Signal
    → PaperExchange.send_order() → Order
    → DemoRepository.save_tick() → PostgreSQL
    → TelegramNotifier (si hi ha trade)
```

---

## 6. Com Afegir Coses Noves

### 6.1 Afegir un bot clàssic nou

1. Crea `bots/classical/nom_bot.py` que hereti de `BaseBot`
2. Implementa `observation_schema()` i `on_observation()`
3. Crea `config/bots/nom.yaml` amb els paràmetres
4. Afegeix el bot al `bot_classes` dict de `DemoRunner._load_bots()`
5. Afegeix el bot a `scripts/run_comparison.py` si el vols comparar

**Exemple mínim:**
```python
class MomBot(BaseBot):
    def observation_schema(self):
        return ObservationSchema(features=["close", "rsi_14"], timeframes=["1h"], lookback=50)
    
    def on_observation(self, obs):
        rsi = obs["1h"]["features"]["rsi_14"].iloc[-1]
        if rsi < 30:
            return Signal(bot_id=self.bot_id, action=Action.BUY, size=0.5, ...)
        return Signal(bot_id=self.bot_id, action=Action.HOLD, size=0.0, ...)
```

### 6.2 Afegir un model ML nou

1. Crea `bots/ml/nom_model.py` amb mètodes `train()`, `predict()`, `save()`, `load()`
2. Afegeix el cas al `_load_model()` de `MLBot`
3. Crea `config/training/nom_experiment.yaml`
4. Afegeix el cas a `scripts/train_models.py`

### 6.3 Afegir una reward function per a RL

1. Crea un fitxer a `bots/rl/rewards/` (o afegeix a `builtins.py`)
2. Usa el decorator `@register("nom")`
3. Importa el mòdul a `bots/rl/environment.py` (com `from bots.rl.rewards import builtins`)
4. Al YAML de training, posa `reward_type: "nom"`

### 6.4 Afegir un exchange real

1. Crea `exchanges/live_binance.py` que hereti de `BaseExchange`
2. Implementa tots els mètodes abstractes connectant a la API real
3. La interfície és idèntica al PaperExchange — el Runner no nota la diferència

### 6.5 Afegir un timeframe nou

1. Descarrega les dades: `fetcher.fetch_and_store("BTC/USDT", "15m", since=...)`
2. A l'`ObservationSchema` del bot, afegeix `"15m"` a `timeframes`
3. L'`ObservationBuilder` detecta automàticament el nou timeframe i carrega les dades

### 6.6 Afegir una feature nova al Feature Store

1. Afegeix el mètode estàtic a `TechnicalIndicators` a `data/processing/technical.py`
2. Crida'l a `compute_features()`
3. Al bot, declara el nom de la columna nova a `observation_schema().features`

### 6.7 Afegir un camp nou a la DB

1. Modifica el model SQLAlchemy a `core/db/models.py`
2. Genera la migració: `poetry run alembic revision --autogenerate -m "descripció"`
3. Aplica la migració: `poetry run alembic upgrade head`
4. Si cal, modifica el model Pydantic corresponent a `core/models.py`

---

## 7. Base de Dades

### Taules principals

| Taula | Contingut |
|---|---|
| `candles` | Dades OHLCV de BTC/USDT en 1h, 4h, 1d |
| `demo_ticks` | Un registre per bot per tick del DemoRunner (cada 60s) |
| `demo_trades` | Un registre per cada BUY/SELL executat durant el Demo |
| `signals` | Reservat (no s'usa activament encara) |
| `orders` | Reservat (no s'usa activament encara) |
| `trades` | Reservat (no s'usa activament encara) |

### Queries útils

```sql
-- Quantes candles tenim per timeframe
SELECT timeframe, COUNT(*), MIN(timestamp), MAX(timestamp)
FROM candles GROUP BY timeframe;

-- Últims ticks del Demo
SELECT bot_id, timestamp AT TIME ZONE 'Europe/Madrid' as ts_local,
       price, action, portfolio_value
FROM demo_ticks ORDER BY timestamp DESC LIMIT 20;

-- Evolució del portfolio per bot
SELECT bot_id, DATE_TRUNC('day', timestamp) as dia,
       AVG(portfolio_value) as portfolio_mig
FROM demo_ticks
GROUP BY bot_id, dia ORDER BY bot_id, dia;

-- PnL acumulat per bot
SELECT bot_id,
       MAX(portfolio_value) - 10000 as pnl_max,
       LAST(portfolio_value ORDER BY timestamp) - 10000 as pnl_actual
FROM demo_ticks GROUP BY bot_id;

-- Tots els trades d'un bot
SELECT timestamp AT TIME ZONE 'Europe/Madrid' as ts_local,
       action, price, size_btc, size_usdt, fees, portfolio_value
FROM demo_trades WHERE bot_id = 'trend_v1' ORDER BY timestamp;

-- Resum de trades per bot
SELECT bot_id,
       COUNT(*) as total_trades,
       SUM(CASE WHEN action='buy' THEN 1 ELSE 0 END) as buys,
       SUM(CASE WHEN action='sell' THEN 1 ELSE 0 END) as sells,
       SUM(fees) as total_fees
FROM demo_trades GROUP BY bot_id;

-- Detecta gaps a les dades 1h
SELECT anterior, timestamp, timestamp - anterior as diferencia
FROM (
    SELECT timestamp, LAG(timestamp) OVER (ORDER BY timestamp) as anterior
    FROM candles WHERE timeframe = '1h' AND symbol = 'BTC/USDT'
) sub WHERE timestamp - anterior > INTERVAL '1 hour';
```

---

## 8. Configuració YAML

### config/bots/trend.yaml
```yaml
bot_id: trend_v1
timeframe: "1h"
lookback: 200
features: ["close", "rsi_14"]
ema_fast: 47
ema_slow: 179
rsi_overbought: 70
rsi_oversold: 30
trade_size: 0.53   # fracció del capital USDT a invertir
```

### config/demo.yaml
```yaml
demo:
  symbol: "BTC/USDT"
  timeframe: "1h"
  update_interval_seconds: 60

bots:
  - config_path: "config/bots/trend.yaml"
    enabled: true
  - config_path: "config/bots/dca.yaml"
    enabled: true

exchange:
  config_path: "config/exchanges/paper.yaml"

telegram:
  enabled: true
```

### config/exchanges/paper.yaml
```yaml
initial_capital: 10000.0
fee_rate: 0.001       # 0.1% per operació
slippage_rate: 0.0005 # 0.05% slippage simulat
```

### config/training/rf_experiment_1.yaml
```yaml
experiment_name: "rf_experiment_1"
model_type: "random_forest"
data:
  symbol: "BTC/USDT"
  timeframes: ["1h"]
  forward_window: 24    # prediu si el preu pujarà en les properes 24h
  threshold_pct: 0.01   # considera pujada si > 1%
model:
  n_estimators: 100
  max_depth: 10
output:
  model_path: "models/rf_v1.pkl"
```

---

## 9. Comandes de Referència

```bash
# Infraestructura
make setup              # Instal·la dependències i aixeca Docker
make db-shell           # Obre una sessió PostgreSQL interactiva

# Dades
make download-data      # Descàrrega inicial (2022 → avui)
make update-data        # Actualitza candles noves
make validate-data      # Diagnòstic: gaps, duplicats, rangs

# Backtesting i optimització
make compare            # Compara tots els bots en backtest
make optimize           # Cerca millors paràmetres amb Optuna

# Models ML/RL
make train              # Entrena Random Forest i XGBoost
make train-rl           # Entrena agents DQN/PPO
make mlflow             # UI MLflow → http://localhost:5001

# Demo
make demo               # Arrenca el Demo Runner (temps real, 24/7)
make dashboard          # Dashboard Streamlit → http://localhost:8501

# Migracions DB
poetry run alembic revision --autogenerate -m "descripció"
poetry run alembic upgrade head
poetry run alembic history

# Tests
poetry run pytest tests/unit/ -v
poetry run pytest tests/unit/test_models.py -v  # test específic
```

---

## 10. Variables d'Entorn

Fitxer `.env` a l'arrel del projecte (mai comitar):

```bash
# Base de dades
DATABASE_URL=postgresql://btc_user:btc_password@localhost:5432/btc_trading

# Telegram (opcional, necessari si telegram.enabled: true al demo.yaml)
TELEGRAM_TOKEN=xxxxxxxxx:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=xxxxxxxxx
```

---

## 11. Si Vols Canviar de BTC a ETH

Canvis necessaris (tots a fitxers de configuració, zero canvis de codi):

**1. Descarregar dades d'ETH:**
```python
# scripts/download_data.py — canvia el symbol
fetcher.fetch_and_store(symbol="ETH/USDT", timeframe="1h", since=...)
```

**2. Actualitzar config/demo.yaml:**
```yaml
demo:
  symbol: "ETH/USDT"   # ← canvia aquí
  timeframe: "1h"
```

**3. Actualitzar scripts/run_comparison.py:**
```python
comparator = BotComparator(bots=bots, symbol="ETH/USDT", timeframe="1h")
```

**4. Actualitzar scripts/optimize_bots.py i train_models.py:**
```yaml
# config/training/rf_experiment_1.yaml
data:
  symbol: "ETH/USDT"   # ← canvia aquí
```

**Res més.** El codi és completament agnòstic al symbol. `PaperExchange` treballa amb qualsevol parella, `ObservationBuilder` carrega les dades del symbol que li passis, i els bots no saben quin actiu estan operant.

**Consideració important:** Els paràmetres optimitzats (ema_fast=47, ema_slow=179...) s'han optimitzat per a BTC. Per a ETH, caldria re-executar `make optimize` per trobar els paràmetres adequats per a la volatilitat d'ETH.

---

## 12. Consideracions de Futur

### Desplegament en servidor (Fase 5)
- **Oracle Cloud Free Tier** (recomanat): 4 ARM vCPUs, 24 GB RAM, gratuït
- **Hetzner CX22** (alternativa): 2 vCPUs, 4 GB RAM, ~4€/mes
- Gestió amb Docker Compose + systemd per a restart automàtic
- Cron per a `make update-data` cada hora

### Features pendents
- Actualització automàtica de la cache del DemoRunner (ara usa dades estàtiques)
- Sentiment data: Fear & Greed Index (API gratuïta)
- On-chain data: SOPR, exchange flows
- Multi-actiu: suport per a BTC + ETH simultàniament
- Models DL: LSTM, Transformer per a sèries temporals

### Limitacions conegudes
- La cache del `ObservationBuilder` no s'actualitza en temps real; el DemoRunner sempre usa la última candle completa de la DB, no el preu actual de Binance. Això és correcte per a estratègies horàries però no per a estratègies intraday
- `demo_ticks` creixerà ~1.000 files/dia amb 2 bots actius (~365k files/any). Considera una política de retenció o TimescaleDB compression després de 6 mesos
- Els models ML (RF, XGBoost) es guarden com a pickle; si canvies la versió de sklearn, pot haver incompatibilitat en carregar
