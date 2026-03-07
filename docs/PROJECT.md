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

| Família | Bot | Tipus de senyal | Config |
|---------|-----|----------------|--------|
| Classical | DCABot | Discret (BUY/HOLD) | `config/bots/dca.yaml` |
| Classical | TrendBot | Discret (BUY/SELL/HOLD) | `config/bots/trend.yaml` |
| Classical | GridBot | Discret (BUY/SELL/HOLD) | `config/bots/grid.yaml` |
| Classical | HoldBot | Discret (BUY/HOLD) — benchmark | `config/bots/hold.yaml` |
| ML | MLBot | Discret (qualsevol model ML) | `config/bots/ml_bot*.yaml` |
| RL | RLBot | Discret o Continu | `config/bots/rl_bot*.yaml` |

Veure **[MODELS.md](./MODELS.md)** per a descripció detallada de cada model.

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
│   ├── classical/       DCABot, TrendBot, GridBot, HoldBot
│   ├── ml/              MLBot + Random Forest, XGBoost, LightGBM, CatBoost, GRU, PatchTST
│   └── rl/              RLBot + PPO (discret), SAC (continu), entorn Gym
├── config/
│   ├── bots/            {bot}.yaml + {bot}_optimized.yaml (auto-generat per Optuna)
│   ├── training/        {model}_experiment_1.yaml + {model}_optimized.yaml
│   ├── optimization/    search spaces per a Optuna
│   ├── settings.yaml    paràmetres globals (BD, exchange, etc.)
│   └── demo.yaml        quins bots s'executen al DemoRunner
├── core/
│   ├── backtesting/     BacktestEngine, BacktestMetrics, Optimizer, Comparator
│   ├── engine/          Runner (backtest), DemoRunner (paper trading)
│   ├── db/              models.py (esquema), session.py, repository.py
│   ├── interfaces/      BaseBot, BaseMLModel, BaseRLAgent, BaseExchange, BaseStrategy
│   └── models.py        Candle, Signal, Order, Trade (Pydantic)
├── data/
│   ├── processing/      TechnicalIndicators, DatasetBuilder
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
└── docs/                Tota la documentació del projecte
```

---

## Scripts — referència ràpida

| Script | Propòsit | Comanda típica |
|--------|----------|----------------|
| `optimize_bots.py` | Optuna per a bots clàssics | `python scripts/optimize_bots.py --bots dca trend grid --trials 50` |
| `optimize_models.py` | Optuna per a ML + RL | `python scripts/optimize_models.py --no-rl --trials 15` |
| `train_models.py` | Entrena models ML | `python scripts/train_models.py` |
| `train_rl.py` | Entrena agents RL (500k steps) | `python scripts/train_rl.py` |
| `run_comparison.py` | Backtesta tots i compara | `python scripts/run_comparison.py --all` |
| `run_demo.py` | Executa paper trading 24/7 | `python scripts/run_demo.py` |
| `validate_data.py` | Valida gaps i duplicats a la BD | `python scripts/validate_data.py` |

**Tots els scripts auto-carreguen `_optimized.yaml` si existeix.** Si no, usen `_experiment_1.yaml` o el YAML base.

---

## Cicle de vida complet

```
optimize_bots   →  config/bots/{bot}_optimized.yaml
optimize_models →  config/training/{model}_optimized.yaml
                        ↓
train_models    →  models/{model}.pkl / .pt
train_rl        →  agents/{agent}.zip
                        ↓
run_comparison  →  BacktestMetrics + MLflow (validació)
                        ↓
run_demo        →  Paper trading 24/7 + Telegram + DB
```

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

*Última actualització: Març 2026 · Versió 1.1*
