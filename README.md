# BTC Trading Bot

Plataforma de **paper trading** algorísmic per a BTC/USDT.
Suporta tres famílies d'estratègies — clàssiques, ML supervisat i RL —
les optimitza automàticament amb Optuna, les compara rigorosament en
walk-forward validation i les executa en paral·lel 24/7 amb notificacions Telegram.

**No és producció.** És un laboratori de recerca per acumular evidència estadística sobre quines estratègies funcionen, amb diners virtuals.

---

## Requisits

```bash
Python 3.11+
PostgreSQL 14+
```

```bash
pip install -r requirements.txt
```

Variables d'entorn (o `config/settings.yaml`):

```yaml
database:
  host: localhost
  port: 5432
  name: btc_trading
  user: btc_user
  password: btc_password
```

---

## Quick start

```bash
# 1. Descarregar dades històriques (detecta automàticament els timeframes necessaris)
python scripts/download_data.py

# 2. Entrenar models ML
python scripts/train_models.py

# 3. Entrenar agents RL
python scripts/train_rl.py

# 4. Comparar tots els bots en out-of-sample
python scripts/run_comparison.py --all

# 5. Executar paper trading 24/7
python scripts/run_demo.py
```

---

## Estratègies disponibles

**Clàssics** (sense entrenament, regles deterministes):

| Bot | Lògica | Config |
|-----|--------|--------|
| HoldBot | Buy & hold — *benchmark de referència* | `config/models/hold.yaml` |
| DCABot | Dollar-Cost Averaging | `config/models/dca.yaml` |
| TrendBot | EMA crossover + RSI | `config/models/trend.yaml` |
| GridBot | Bollinger Bands | `config/models/grid.yaml` |
| MeanReversionBot | Z-score + RSI + filtre de volum | `config/models/mean_reversion.yaml` |
| MomentumBot | ROC + volum + MACD | `config/models/momentum.yaml` |

**ML supervisat** (entrenament offline, predicció binària):

| Model | Tipus | Config |
|-------|-------|--------|
| Random Forest | Tree-based | `config/models/random_forest.yaml` |
| XGBoost | Tree-based | `config/models/xgboost.yaml` |
| LightGBM | Tree-based | `config/models/lightgbm.yaml` |
| CatBoost | Tree-based | `config/models/catboost.yaml` |
| GRU | Deep Learning | `config/models/gru.yaml` |
| PatchTST | Transformer | `config/models/patchtst.yaml` |

**Reinforcement Learning** (entrenament per interacció):

| Agent | Tipus | Config |
|-------|-------|--------|
| PPO | Discret (BUY/SELL/HOLD) | `config/models/ppo.yaml` |
| SAC | Continu [0–1] | `config/models/sac.yaml` |
| PPO on-chain | PPO + features on-chain | `config/models/ppo_onchain.yaml` |
| SAC on-chain | SAC + features on-chain | `config/models/sac_onchain.yaml` |
| PPO professional | PPO 12H + política professional | `config/models/ppo_professional.yaml` |
| SAC professional | SAC 12H + política professional | `config/models/sac_professional.yaml` |
| TD3 professional | TD3 12H + política professional | `config/models/td3_professional.yaml` |

**Regla d'or:** cap bot va a demo fins que supera HoldBot en Sharpe i Calmar en el període de test out-of-sample.

---

## Scripts principals

| Script | Propòsit |
|--------|----------|
| `download_data.py` | Descàrrega OHLCV des de 2019 (timeframes llegits del YAML) |
| `download_fear_greed.py` | Fear & Greed Index des de 2018 |
| `download_blockchain.py` | Mètriques on-chain des de 2009 |
| `update_data.py` | Actualització incremental de tots els OHLCV |
| `train_models.py` | Entrena tots els models ML (o els que specifiques) |
| `train_rl.py` | Entrena agents RL (500k steps) |
| `optimize_bots.py` | Optuna per a bots clàssics |
| `optimize_models.py` | Optuna per a ML + RL |
| `run_comparison.py` | Backtest walk-forward + ranking Sharpe |
| `run_demo.py` | Paper trading 24/7 + Telegram |
| `validate_data.py` | Comprova gaps i duplicats a la BD |
| `check_data_completeness.py` | Resum de cobertura de totes les fonts |

---

## Afegir un model nou

El sistema usa **auto-discovery**: els registres de models es poblen automàticament llegint els YAMLs de `config/models/`. Per afegir un model nou:

1. Crea `bots/ml/my_model.py` (hereta `BaseTreeModel` si és tree-based, o `BaseMLModel` directament)
2. Crea `config/models/my_model.yaml` amb `category: ML`, `module: bots.ml.my_model`, `class_name: MyModel`
3. Ja apareix a `train_models.py`, `run_comparison.py` i `ml_bot.py` sense cap edició addicional

Per a bots clàssics, el mateix: crea el bot + YAML amb `category: classic`, `module:` i `class_name:`.

Guia completa: `docs/EXTENDING.md`.

---

## Estructura de directoris

```
btc-trading-bot/
├── bots/
│   ├── classical/       # 6 bots clàssics
│   ├── ml/              # MLBot + 6 backends (base_tree_model.py, gru, patchtst)
│   └── rl/              # RLBot + agents (PPO, SAC, TD3) + environments + rewards
├── config/
│   ├── models/          # 1 YAML per model: config + training + optimization + bot
│   ├── settings.yaml    # BD, exchange, walk-forward dates
│   └── demo.yaml        # Bots actius al DemoRunner
├── core/
│   ├── backtesting/     # BacktestEngine, Comparator, Optimizers, agent_validator
│   ├── db/              # Models SQLAlchemy, sessions
│   ├── engine/          # Runner, DemoRunner
│   ├── interfaces/      # BaseBot, BaseMLModel, BaseRLAgent, BaseExchange
│   └── config_utils.py  # apply_best_params, discover_configs
├── data/
│   ├── processing/      # FeatureBuilder, DatasetBuilder, TechnicalIndicators,
│   │                    # TimeSeriesDataset (torch), ExternalLoader
│   ├── observation/     # ObservationBuilder
│   └── sources/         # Fetchers: OHLCV, Fear&Greed, futures, blockchain, Vision
├── docs/                # PROJECT.md, MODELS.md, EXTENDING.md, CONFIGURATION.md, DATABASE.md
├── models/              # Models entrenats: *.pkl, *.pt, *.zip
├── scripts/             # Tots els punts d'entrada
└── tests/               # 123 tests (smoke + unit + integration)
```

---

## Tests

```bash
pytest tests/smoke/ tests/unit/ -v          # 123 tests, zero BD necessària
pytest tests/integration/ -m integration    # necessita PostgreSQL + dades
```

---

## Documentació

- **[docs/PROJECT.md](docs/PROJECT.md)** — Arquitectura, flux d'un tick, walk-forward, scripts
- **[docs/MODELS.md](docs/MODELS.md)** — Descripció de cada model i els seus paràmetres
- **[docs/EXTENDING.md](docs/EXTENDING.md)** — Com afegir bots, models, agents i fonts de dades
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — Referència completa dels YAMLs
- **[docs/DATABASE.md](docs/DATABASE.md)** — Esquema BD, taules, consultes útils
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — Tasques pendents i visió de futur

---

*Última actualització: Març 2026*
