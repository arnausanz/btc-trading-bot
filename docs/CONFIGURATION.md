# Configuració — Referència Completa

> Referència de tots els fitxers de configuració YAML, variables globals i arguments CLI.
> Per al flux complet d'ús: veure **[PROJECT.md](./PROJECT.md)**.

---

## Fitxers de configuració — mapa

```
config/
├── settings.yaml          → Globals: BD, exchange, símbols, walk-forward dates
├── demo.yaml              → Quins bots s'activen al DemoRunner
├── bots/
│   ├── {bot}.yaml         → Config base d'un bot (hand-crafted o punt de partida)
│   └── {bot}_optimized.yaml → Config optimitzada per Optuna (auto-generat)
├── training/
│   ├── {model}_experiment_N.yaml → Config d'entrenament per a ML/RL
│   └── {model}_optimized.yaml    → Config optimitzada per Optuna (auto-generat)
└── optimization/
    └── {model}.yaml       → Search space d'Optuna per a cada model
```

**Regla d'auto-carrega:** tots els scripts comproven si existeix `_optimized.yaml` i el carreguen per defecte. Si no existeix, usen `_experiment_1.yaml` o el YAML base.

---

## `config/settings.yaml`

Variables globals del sistema. Les més importants:

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

Controla quins bots s'executen al DemoRunner.

```yaml
demo:
  symbol: BTC/USDT
  timeframe: 1h
  update_interval_seconds: 60

bots:
  - config_path: config/bots/trend.yaml
    enabled: true
  - config_path: config/bots/ml_bot.yaml
    enabled: true
  - config_path: config/bots/rl_bot_ppo.yaml
    enabled: false    # ← desactivat sense eliminar

exchange:
  config_path: config/exchanges/paper.yaml

telegram:
  enabled: true
```

---

## `config/bots/{bot}.yaml`

Exemple per a un bot clàssic (`trend.yaml`):

```yaml
bot_id: trend_v1
symbol: BTC/USDT
timeframe: 1h
lookback: 200
features:
  - close
  - rsi_14

ema_fast: 47        # ← paràmetres del bot
ema_slow: 179
rsi_overbought: 70
rsi_oversold: 33
trade_size: 0.53
```

Exemple per a un bot ML (`ml_bot.yaml`):

```yaml
bot_id: ml_rf_v1
symbol: BTC/USDT
timeframe: 1h
model_type: random_forest          # ← clau al _MODEL_REGISTRY
model_path: models/random_forest.pkl
config_path: config/training/rf_experiment_1.yaml
```

---

## `config/training/{model}_experiment_N.yaml`

Config d'entrenament per a un model ML:

```yaml
model_type: random_forest
params:
  n_estimators: 100
  max_depth: 6
  min_samples_split: 5
  max_features: sqrt

# Walk-forward (hereta de settings.yaml si no s'especifica)
train_until: "2024-12-31"
target: price_up_1pct_in_24h     # columna target del DatasetBuilder
features:                          # features d'entrada (columnes del DataFrame)
  - close
  - rsi_14
  - ema_9
  - ema_21
  - macd
  - bb_upper
  - bb_lower
  - atr_14
  - volume_sma_20
```

---

## `config/optimization/{model}.yaml`

Search space per a Optuna. Camps principals:

```yaml
base_config: config/training/rf_experiment_1.yaml   # punt de partida
model_type: random_forest
n_trials: 30
metric: accuracy_mean          # accuracy_mean | sharpe_ratio | precision_mean
direction: maximize            # maximize | minimize

search_space:
  n_estimators:
    type: int
    low: 50
    high: 500
  max_depth:
    type: int
    low: 3
    high: 20
  learning_rate:               # per a models amb LR
    type: float
    low: 0.001
    high: 0.3
    log: true                  # escala logarítmica
  max_features:
    type: categorical
    choices: [sqrt, log2]
```

**Tipus de paràmetres Optuna:** `int`, `float`, `categorical`.

---

## Arguments CLI — referència

### `optimize_bots.py`
```
--bots      dca trend grid hold    # Selecció de bots (default: tots)
--trials    50                      # Override de n_trials
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
--steps     500000                 # Timesteps d'entrenament (default: 500k)
```

### `run_comparison.py`
```
--all                              # Executa tots els bots configurats
--bots      dca trend rf           # Executa només els bots indicats
```

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

*Última actualització: Març 2026 · Versió 1.1*
