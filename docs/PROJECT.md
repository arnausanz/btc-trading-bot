# BTC Trading Bot — Font de Veritat del Projecte

> Documentació de referència per entendre, operar i estendre el sistema.
> Versions de codi: `bots/`, `core/`, `scripts/`, `config/`.

---

## Què és

Plataforma de trading algorísmic en **paper trading** per a BTC/USDT. Suporta tres famílies d'estratègies (clàssiques, ML, RL), permet optimitzar i comparar-les rigorosament amb walk-forward validation, i té un runner que executa tots els bots en paral·lel 24/7 amb persistència i notificacions Telegram.

**No és producció.** És un laboratori de recerca. L'objectiu és acumular prou evidència estadística per saber quines estratègies funcionen i quines no, amb diners virtuals.

---

## Arquitectura general

```
┌──────────────────────────────────────────────────────────────┐
│  SCRIPTS (punts d'entrada)                                   │
│  optimize_bots · optimize_models · train_models · train_rl   │
│  run_comparison · run_demo                                   │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐         ┌─────────────────┐
│  DemoRunner  │         │  BacktestEngine │
│  (temps real)│         │  (simulació)    │
└──────┬───────┘         └────────┬────────┘
       │                          │
       └──────────┬───────────────┘
                  ▼
       ┌──────────────────────┐
       │  BOTS (estratègies)  │
       │  Classical / ML / RL │
       └──────────┬───────────┘
                  │  observation_schema()
                  ▼
       ┌──────────────────────┐
       │  DATA LAYER          │
       │  ObservationBuilder  │
       │  TechnicalIndicators │
       │  PostgreSQL (candles) │
       └──────────────────────┘
```

---

## Flux d'un tick (com funciona)

```
1. Binance API (ccxt) → candles OHLCV → PostgreSQL [taula: candles]

2. candles → TechnicalIndicators.compute_features() → DataFrame
   Features: EMA-9/21/50/200, RSI-14, MACD, Bollinger, ATR, Volume-SMA
   (en memòria, no persistides)

3. DemoRunner — cada 60s:
   Binance API → preu actual
     Per a cada bot actiu:
       ObservationBuilder.build() → finestra de N candles + features
       bot.on_observation(obs) → Signal (BUY / SELL / HOLD)
       PaperExchange.send_order(signal) → Order (fees 0.1% + slippage 0.01%)
       DemoRepository.save_tick() + save_trade() → PostgreSQL
       TelegramNotifier → notificació si hi ha trade / cada hora status

4. BacktestEngine — offline:
   Mateix flux però sobre TOTES les candles [TEST_FROM .. avui]
   → BacktestMetrics → Sharpe, Drawdown, Win Rate, Calmar
   → MLflow → registre automàtic de paràmetres i mètriques
```

---

## Walk-Forward Validation

Evita lookahead bias (entrenar amb dades del futur).

```
Dades disponibles: [2024-01-01 ────────────── 2024-12-31 | 2025-01-01 ── avui]
                        TRAIN (optimització/entrenament)    TEST (validació real)
```

Configurat a `core/config.py`:
```python
TRAIN_UNTIL = "2024-12-31"   # última data permesa per a entrenar
TEST_FROM   = "2025-01-01"   # primera data del backtest de validació
```

**Regla d'or:** cap bot s'executa en demo fins que supera HoldBot en Sharpe i Calmar en el període de TEST (mai vist durant l'optimització).

---

## Bots disponibles

| Família | Bot | Tipus de senyal | Config (unificat) |
|---------|-----|----------------|--------|
| classic | HoldBot | Discret (BUY/HOLD) — benchmark | `config/models/hold.yaml` |
| classic | DCABot | Discret (BUY/HOLD) | `config/models/dca.yaml` |
| classic | TrendBot | Discret (BUY/SELL/HOLD) | `config/models/trend.yaml` |
| classic | GridBot | Discret (BUY/SELL/HOLD) | `config/models/grid.yaml` |
| classic | MeanReversionBot | Discret (BUY/SELL/HOLD) | `config/models/mean_reversion.yaml` |
| classic | MomentumBot | Discret (BUY/SELL/HOLD) | `config/models/momentum.yaml` |
| ML | MLBot (Random Forest) | Discret | `config/models/random_forest.yaml` |
| ML | MLBot (XGBoost) | Discret | `config/models/xgboost.yaml` |
| ML | MLBot (LightGBM) | Discret | `config/models/lightgbm.yaml` |
| ML | MLBot (CatBoost) | Discret | `config/models/catboost.yaml` |
| ML | MLBot (GRU) | Discret | `config/models/gru.yaml` |
| ML | MLBot (PatchTST) | Discret | `config/models/patchtst.yaml` |
| RL | RLBot (PPO) | Discret | `config/models/ppo.yaml` |
| RL | RLBot (SAC) | Continu | `config/models/sac.yaml` |
| RL | RLBot (PPO on-chain) | Discret | `config/models/ppo_onchain.yaml` |
| RL | RLBot (SAC on-chain) | Continu | `config/models/sac_onchain.yaml` |
| RL | RLBot (PPO professional) | Discret-5 | `config/models/ppo_professional.yaml` |
| RL | RLBot (SAC professional) | Continu | `config/models/sac_professional.yaml` |
| RL | RLBot (TD3 professional) | Continu | `config/models/td3_professional.yaml` |

Veure **[MODELS.md](./MODELS.md)** per a descripció detallada de cada model.

> **Auto-discovery:** els registres de models (`_MODEL_REGISTRY`, `BOT_REGISTRY`, `ALL_CONFIGS`)
> es poblen automàticament llegint `config/models/*.yaml`.  Cada YAML declara `module` i
> `class_name` per a la càrrega dinàmica.  Afegir un model = crear YAML + classe.  Zero edicions als scripts.

---

## Mètriques de Backtesting

| Mètrica | Fórmula | Bon valor |
|---------|---------|-----------|
| **Sharpe Ratio** | `mean(returns) / std(returns) * sqrt(252*24)` | > 1.0 |
| **Max Drawdown** | Pitjor caiguda pic → vall (%) | > −20% |
| **Calmar Ratio** | `retorn_anual / abs(max_drawdown)` | > 0.5 |
| **Win Rate** | % trades guanyadors (round-trips buy→sell) | > 50% |
| **Total Return** | Guany/pèrdua total (%) sobre el capital inicial | Positiu |

⚠️ **Nota:** totes les mètriques usen `sqrt(252*24)` perquè les dades són horàries. Amb dades diàries s'usaria `sqrt(252)`.

---

## Estructura de carpetes

```
btc-trading-bot/
├── bots/
│   ├── classical/       DCABot, TrendBot, GridBot, HoldBot, MeanReversionBot, MomentumBot
│   ├── ml/              MLBot + Random Forest, XGBoost, LightGBM, CatBoost, GRU, PatchTST
│   └── rl/              RLBot + agents (PPO, SAC, TD3) + environments + rewards + constants
├── config/
│   ├── models/          UN YAML per model/bot (base + training + optimization + bot)
│   │   └── {bot}.yaml          Config completa (paràmetres Optuna dins best_params)
│   ├── exchanges/       Configuració del paper exchange
│   ├── settings.yaml    Paràmetres globals (BD, exchange, etc.)
│   └── demo.yaml        Quins bots s'executen al DemoRunner
├── core/
│   ├── backtesting/     BacktestEngine, Comparator, Optimizers, agent_validator
│   ├── engine/          Runner (backtest), DemoRunner (paper trading)
│   ├── db/              models.py (esquema), session.py, repository.py
│   ├── interfaces/      BaseBot, BaseMLModel, BaseRLAgent, BaseExchange, BaseStrategy
│   ├── config_utils.py  apply_best_params, discover_configs (auto-discovery)
│   └── models.py        Candle, Signal, Order, Trade (Pydantic)
├── data/
│   ├── processing/      FeatureBuilder, DatasetBuilder, TechnicalIndicators,
│   │                    TimeSeriesDataset (torch), ExternalLoader
│   └── observation/     ObservationBuilder
├── exchanges/
│   └── paper.py         PaperExchange (simulació amb fees + slippage)
├── monitoring/
│   └── dashboard.py     Streamlit dashboard (preus de la BD)
├── scripts/             Punts d'entrada — veure secció Scripts
├── tests/
│   ├── smoke/           66 tests, zero deps — imports, configs, models, math
│   ├── unit/            57 tests — lògica amb dades mock
│   └── integration/     5 tests — pipeline complet (necessita BD real)
├── models/              Models entrenats: *.pkl, *.pt, *.zip
├── [OLD]config/         Arxius antics (conservats per referència)
└── docs/                Tota la documentació del projecte
```

---

## Scripts — referència ràpida

| Script | Propòsit | Comanda típica |
|--------|----------|----------------|
| `optimize_bots.py` | Optuna per a bots clàssics | `python scripts/optimize_bots.py --bots dca trend grid` |
| `optimize_models.py` | Optuna per a ML + RL | `python scripts/optimize_models.py --no-rl --trials 15` |
| `train_models.py` | Entrena models ML | `python scripts/train_models.py` |
| `train_rl.py` | Entrena agents RL (500k steps) | `python scripts/train_rl.py` |
| `run_comparison.py` | Backtesta tots i compara | `python scripts/run_comparison.py --all` |
| `run_demo.py` | Executa paper trading 24/7 | `python scripts/run_demo.py` |
| `validate_data.py` | Valida gaps i duplicats a la BD | `python scripts/validate_data.py` |

**Paràmetres Optuna:** els millors hiperparàmetres es guarden dins el YAML base
a la secció `best_params`. `apply_best_params()` els aplica en entrenament/inferència.
No hi ha fitxers `*_optimized.yaml` separats.

---

## Cicle de vida complet

```
optimize_bots   →  config/models/{bot}.yaml [secció best_params]
optimize_models →  config/models/{model}.yaml [secció best_params]
                        ↓
train_models    →  models/{model}.pkl / .pt
train_rl        →  models/{agent}.zip
                        ↓
run_comparison  →  BacktestMetrics + MLflow (validació)
                        ↓
run_demo        →  Paper trading 24/7 + Telegram + DB
```

**YAML unificat per model:** cada fitxer `config/models/{model}.yaml` conté
la configuració completa: features, training, optimization i bot deployment.
No cal gestionar múltiples fitxers per model.

---

## Tests

```bash
pytest tests/smoke/ tests/unit/ -v      # 123 tests, zero BD necessària
pytest tests/integration/ -m integration # necessita PostgreSQL
```

| Nivell | Fitxers | Tests | Deps |
|--------|---------|-------|------|
| smoke | smoke/ | 66 | cap |
| unit | unit/ | 57 | cap (dades mock) |
| integration | integration/ | 5 | PostgreSQL + dades |

---

## Documentació

- **PROJECT.md** ← Ets aquí — visió general, arquitectura, flux
- **[MODELS.md](./MODELS.md)** — Tots els models: descripció, paràmetres, pros/contres
- **[DATABASE.md](./DATABASE.md)** — Esquema BD, taules, consultes útils
- **[EXTENDING.md](./EXTENDING.md)** — Com afegir bots, models, agents, fonts de dades
- **[CONFIGURATION.md](./CONFIGURATION.md)** — Referència de tots els YAMLs i CLI args
- **[ROADMAP.md](./ROADMAP.md)** — Tasques pendents, millores, arquitectures futures
- **[decisions/](./decisions/)** — Architecture Decision Records

---

*Última actualització: Març 2026 · Versió 2.0*
