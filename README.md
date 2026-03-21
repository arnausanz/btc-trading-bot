# BTC Trading Bot

Plataforma de **paper trading** algorísmic per a BTC/USDT.
Suporta tres famílies d'estratègies — clàssiques, ML supervisat i RL —
les optimitza automàticament amb Optuna, les compara rigorosament en
walk-forward validation i les executa en paral·lel 24/7 amb notificacions Telegram.

**No és producció.** És un laboratori de recerca per acumular evidència estadística sobre quines estratègies funcionen, amb diners virtuals.

---

## Documentació — Comença aquí

> **Si ets nou al projecte**, obre primer [`docs/00_START_HERE.md`](docs/00_START_HERE.md).

| Document | Contingut |
|----------|-----------|
| **[00_START_HERE.md](docs/00_START_HERE.md)** | Mapa del projecte, glossari, FAQ, ordre de lectura |
| **[01_ARCHITECTURE.md](docs/01_ARCHITECTURE.md)** | Arquitectura, flux de dades, bots disponibles, scripts |
| **[02_GATE_SYSTEM.md](docs/02_GATE_SYSTEM.md)** | Gate System (5 portes seqüencials) — documentació completa |
| **[03_ML_RL_MODELS.md](docs/03_ML_RL_MODELS.md)** | Models ML supervisats i RL — paràmetres i decisions |
| **[04_CONFIGURATION.md](docs/04_CONFIGURATION.md)** | Referència de tots els YAMLs i arguments CLI |
| **[05_DATABASE.md](docs/05_DATABASE.md)** | Esquema BD, taules, consultes útils |
| **[06_EXTENDING.md](docs/06_EXTENDING.md)** | Com afegir nous bots, models i fonts de dades |
| **[07_OPERATIONS.md](docs/07_OPERATIONS.md)** | Runbook: com engegar, monitoritzar i mantenir el sistema |
| **[08_DECISIONS.md](docs/08_DECISIONS.md)** | 11 Architecture Decision Records amb justificació |
| **[ROADMAP.md](docs/ROADMAP.md)** | Tasques pendents i visió de futur |
| **[examples/trade_walkthrough.md](docs/examples/trade_walkthrough.md)** | Un trade del Gate System pas a pas |
| **[examples/near_miss_analysis.md](docs/examples/near_miss_analysis.md)** | Anàlisi d'una oportunitat rebutjada |

---

## Quick Start

```bash
# 1. Setup
cp .env.example .env              # editar DATABASE_URL, TELEGRAM_TOKEN
pip install -r requirements.txt
alembic upgrade head              # crear esquema BD

# 2. Dades
python scripts/download_data.py
python scripts/download_fear_greed.py
python scripts/update_futures.py

# 3. Entrenar (veure 07_OPERATIONS.md per al cicle complet)
python scripts/train_models.py
python scripts/train_rl.py
python scripts/train_gate_regime.py

# 4. Validar
python scripts/run_comparison.py --all

# 5. Demo 24/7
python scripts/run_demo.py
```

---

## Estratègies disponibles

**Clàssics** (sense entrenament):
HoldBot · DCABot · TrendBot · GridBot · MeanReversionBot · MomentumBot · EnsembleBot

**ML supervisat** (predicció binària):
Random Forest · XGBoost · LightGBM · CatBoost · GRU · PatchTST · TFT (pendent)

**Reinforcement Learning** (política de trading):
PPO · SAC · PPO on-chain · SAC on-chain · PPO professional · SAC professional · TD3 professional · TD3 multiframe

**Gate System** (5 portes seqüencials):
P1 Règim (HMM+XGBoost) · P2 Salut · P3 Estructura · P4 Momentum · P5 Risc

**Regla d'or:** cap bot va a demo fins que supera HoldBot en Sharpe i Calmar en el període de test out-of-sample.

---

## Tests

```bash
pytest tests/smoke/ tests/unit/ -v          # 123 tests, sense BD necessària
pytest tests/integration/ -m integration    # necessita PostgreSQL + dades
```

---

## Estructura de directoris

```
btc-trading-bot/
├── bots/
│   ├── classical/       HoldBot, DCABot, TrendBot, GridBot, MeanReversionBot, MomentumBot, EnsembleBot
│   ├── ml/              MLBot + models (RF, XGBoost, LightGBM, CatBoost, GRU, PatchTST, TFT)
│   ├── rl/              RLBot + agents (PPO, SAC, TD3) + environments + rewards
│   └── gate/            GateBot + 5 portes + regime_models (HMM, XGBoost)
├── config/
│   ├── models/          1 YAML per model: config + training + optimization
│   ├── settings.yaml    BD, exchange, walk-forward dates
│   └── demo.yaml        Bots actius al DemoRunner
├── core/
│   ├── backtesting/     BacktestEngine, Comparator, Optimizers
│   ├── db/              Models SQLAlchemy, sessions, repository
│   ├── engine/          DemoRunner, BacktestEngine runner
│   └── interfaces/      BaseBot, BaseMLModel, BaseRLAgent, BaseExchange
├── data/
│   ├── processing/      FeatureBuilder, DatasetBuilder, TechnicalIndicators, ExternalLoader
│   └── observation/     ObservationBuilder
├── docs/                Documentació completa (veure taula d'índex a dalt)
├── models/              Models entrenats: *.pkl, *.pt, *.zip
├── scripts/             Tots els punts d'entrada
└── tests/               smoke/ + unit/ + integration/
```

---

*Última actualització: Març 2026*
