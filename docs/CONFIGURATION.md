# Configuració — Referència Completa

> Referència de tots els fitxers de configuració YAML, variables globals i arguments CLI.
> Per al flux complet d'ús: veure **[PROJECT.md](./PROJECT.md)**.

---

## Estructura de configuració — mapa

```
config/
├── settings.yaml          → Globals: BD, exchange, símbols, walk-forward dates
├── demo.yaml              → Quins bots s'executen al DemoRunner
├── exchanges/
│   └── paper.yaml         → Configuració del PaperExchange (simulació)
└── models/
    ├── {model}.yaml        → Config UNIFICADA per a cada model/bot
    └── {model}_optimized.yaml  → Config optimitzada per Optuna (auto-generat)
```

**Un sol YAML per model** conté tota la informació: dades, entrenament, optimització i desplegament.

**Regla d'auto-carrega:** els scripts comproven si existeix `{model}_optimized.yaml` i el carreguen per defecte. Si no existeix, usen `{model}.yaml`.

---

## `config/settings.yaml`

Variables globals del sistema:

```yaml
database:
  host: localhost
  port: 5432
  name: btc_trading
  user: btc_user
  password: btc_password

exchanges:
  default: paper
  binance:
    api_key: ""        # Necessari per descarregar dades en viu
    api_secret: ""

data:
  default_symbol: BTC/USDT
  default_timeframes: [1h, 4h, 1d]
  historical_since: "2019-01-01"

backtesting:
  train_until: "2024-12-31"   # ← Walk-forward: última data d'entrenament
  test_from:   "2025-01-01"   # ← Walk-forward: primera data de test
```

---

## `config/demo.yaml`

Controla quins bots s'executen al DemoRunner:

```yaml
demo:
  symbol: BTC/USDT
  timeframe: 1h
  update_interval_seconds: 60

bots:
  - config_path: config/models/trend.yaml
    enabled: true
  - config_path: config/models/xgboost.yaml
    enabled: true
  - config_path: config/models/ppo.yaml
    enabled: false    # ← desactivat sense eliminar

exchange:
  config_path: config/exchanges/paper.yaml

telegram:
  enabled: true
```

---

## `config/models/{model}.yaml` — Schema unificat

**Camp discriminant: `category`**

| `category` | Exemples | Descripció |
|------------|----------|------------|
| `classic`  | hold, dca, trend, grid | Bots basats en regles, sense ML |
| `ML`       | xgboost, gru, patchtst | Models supervisats (predicció de preu) |
| `RL`       | ppo, sac | Agents de reinforcement learning |

---

### Schema — `category: classic`

```yaml
category: classic
model_type: dca           # Identifica la classe de bot (dca/trend/grid/hold)
bot_id: dca_v1
symbol: BTC/USDT
timeframe: 1h
lookback: 50
features: [close]

# Paràmetres d'estratègia (a nivell arrel, llegits directament pel bot)
buy_every_n_ticks: 163
buy_size: 0.05

# Search space per Optuna — llegit per optimize_bots.py
optimization:
  n_trials: 100
  metric: sharpe_ratio
  direction: maximize
  search_space:
    buy_every_n_ticks:
      type: int
      low: 6
      high: 168
    buy_size:
      type: float
      low: 0.05
      high: 0.5
```

---

### Schema — `category: ML`

```yaml
category: ML
model_type: xgboost       # Clau al _MODEL_REGISTRY de MLBot
symbol: BTC/USDT
timeframes: [1h]          # Llista (suporta multi-timeframe)
forward_window: 24        # Horitzó de predicció (hores)
threshold_pct: 0.005      # Mínim % de moviment per a senyal positiu

# Features (passades a FeatureBuilder i DatasetBuilder)
features:
  select: null            # null = totes les columnes; llista = subset
  external: {}            # fonts externes (fear_greed, funding_rate, etc.)

# Secció d'entrenament (llegida per train_models.py i MLOptimizer)
training:
  experiment_name: xgboost_v1
  model_path: models/xgboost_v1.pkl
  model:
    n_estimators: 500
    max_depth: 5
    learning_rate: 0.02
    scale_pos_weight: 2.0

# Search space per Optuna — llegit per optimize_models.py
optimization:
  n_trials: 30
  metric: accuracy_mean
  direction: maximize
  search_space:
    n_estimators: {type: int, low: 100, high: 500}
    max_depth: {type: int, low: 3, high: 10}
    learning_rate: {type: float, low: 0.01, high: 0.3, log: true}

# Configuració de desplegament (llegida per MLBot)
bot:
  bot_id: ml_xgb_v1
  lookback: 200
  min_confidence: 0.4
  trade_size: 0.5
  prediction_threshold: 0.35
```

---

### Schema — `category: RL`

```yaml
category: RL
model_type: ppo            # Clau al _AGENT_REGISTRY
symbol: BTC/USDT
timeframe: 1h

# Features — CRÍTIC: obs_shape = len(select) × lookback
# Ha de ser idèntic entre entrenament i desplegament.
features:
  lookback: 96
  select:
    - close
    - rsi_14
    - macd
    - macd_signal
    - macd_hist
    - atr_14
    - bb_upper_20
    - bb_lower_20
    - bb_middle_20
  external: {}             # fonts externes (fear_greed, funding_rate, etc.)

# Secció d'entrenament (llegida per train_rl.py i RLOptimizer)
training:
  experiment_name: ppo_optimized
  train_pct: 0.8
  model_path: models/ppo_btc_v1
  environment:
    initial_capital: 10000.0
    fee_rate: 0.001
    reward_scaling: 100.0
    reward_type: sharpe    # simple | sharpe | sortino | penalize_inaction
  model:
    total_timesteps: 500000
    learning_rate: 0.00012
    batch_size: 128
    n_steps: 2048
    n_epochs: 10
    gamma: 0.967
    policy: MlpPolicy

# Search space per Optuna — llegit per optimize_models.py
optimization:
  n_trials: 15
  probe_timesteps: 20000   # Timesteps per trial (molt menys que total_timesteps)
  metric: val_return_pct
  direction: maximize
  search_space:
    learning_rate: {type: float, low: 0.0001, high: 0.001, log: true}
    n_steps: {type: categorical, choices: [512, 1024, 2048]}
    reward_type: {type: categorical, choices: [simple, sharpe, sortino]}
    lookback: {type: categorical, choices: [30, 50, 96]}

# Configuració de desplegament (llegida per RLBot)
bot:
  bot_id: rl_ppo_v1
  trade_size: 0.5
```

---

## Regla crítica: `obs_shape` per a RL

```
obs_shape = len(features.select) × features.lookback
```

- `features.select` ha de ser **idèntic** al que s'ha usat per entrenar
- `features.lookback` ha de ser **idèntic** a `training.environment.lookback`
- **Mai canviar sense re-entrenar el model** — causaria `ValueError: Unexpected observation shape`

---

## Features externes

```yaml
features:
  external:
    fear_greed: true                    # afegeix: fear_greed_value, fear_greed_class
    funding_rate: true                  # afegeix: funding_rate
    funding_rate_symbol: BTC/USDT:USDT # opcional, sobreescriu el valor per defecte
    open_interest:
      - symbol: BTC/USDT:USDT
        timeframe: 1h                   # afegeix: oi_btc_1h, oi_usdt_1h
    blockchain:
      - hash-rate                       # afegeix: hash_rate
      - n-unique-addresses              # afegeix: n_unique_addresses
```

Veure `ppo_onchain.yaml` com a exemple complet.

---

## Arguments CLI — referència

### `optimize_bots.py`
```
--bots      dca trend grid    # Selecció de bots (default: tots)
--trials    50                # Override de n_trials (default: usa el YAML)
```

### `optimize_models.py`
```
--models    rf xgb lgbm catboost gru patchtst   # Models ML (default: tots)
--agents    sac ppo                              # Agents RL (default: tots)
--trials    15                                   # Override de n_trials
--no-ml                                          # Salta tots els ML
--no-rl                                          # Salta tots els RL
```

### `train_models.py`
```
--models    rf xgb lgbm catboost gru patchtst   # Models a entrenar (default: tots)
```

### `train_rl.py`
```
--agents    sac ppo                # Agents a entrenar (default: tots)
```

### `run_comparison.py`
```
--all                              # Executa tots els bots (inclou RL)
--bots      dca trend rf ppo       # Executa només els bots indicats
--no-rl                            # Exclou agents RL
```

**Noms de bots disponibles:** `hold`, `dca`, `trend`, `grid`, `rf`, `xgb`, `lgbm`, `catboost`, `gru`, `patchtst`, `ppo`, `sac`

### `run_demo.py`
```
(sense arguments)  # Llegeix tot de config/demo.yaml
```

---

## Variables d'entorn

| Variable | Valor default | Descripció |
|----------|--------------|-----------|
| `DATABASE_URL` | llegit de `settings.yaml` | URL de connexió PostgreSQL |
| `BINANCE_API_KEY` | — | Clau API de Binance (opcional per dades) |
| `BINANCE_API_SECRET` | — | Secret API de Binance |
| `TELEGRAM_BOT_TOKEN` | — | Token del bot de Telegram |
| `TELEGRAM_CHAT_ID` | — | Chat ID per a notificacions |
| `OMP_NUM_THREADS` | 1 | Limita threads OpenMP (evita oversubscription) |
| `MKL_NUM_THREADS` | 1 | Limita threads MKL |
| `MLFLOW_TRACKING_URI` | `mlruns/` | On guarda els experiments MLflow |

---

## PaperExchange — simulació

Configuració de la simulació a `config/exchanges/paper.yaml`:

```yaml
initial_balance_usdt: 10000.0  # Capital inicial
fee_rate: 0.001                 # 0.1% per operació (Binance taker fee)
slippage_rate: 0.0001           # 0.01% d'impacte de mercat simulat
```

---

## MLflow — experiments

Tots els backtests i entrenaments queden registrats a MLflow:

```bash
mlflow ui   # Obre http://localhost:5000
```

Cada experiment registra: `model_type`, `params`, `sharpe`, `drawdown`, `win_rate`, `total_return`.

---

*Última actualització: Març 2026 · Versió 2.0*
