# Arquitectura del Sistema

> Com funciona el projecte de cap a peus: peces principals, flux de dades, patterns clau.
> Per a la posada en marxa, veure **[07_OPERATIONS.md](./07_OPERATIONS.md)**.
> Per als models en detall, veure **[02_GATE_SYSTEM.md](./02_GATE_SYSTEM.md)** i **[03_ML_RL_MODELS.md](./03_ML_RL_MODELS.md)**.

---

## Visió general

Plataforma de **paper trading** algorísmic per a BTC/USDT. Tres famílies d'estratègies (clàssiques, ML supervisat, RL), optimitzades automàticament amb Optuna, comparades amb walk-forward validation rigorosa, i executades en paral·lel 24/7 amb notificacions Telegram.

**No és producció.** Acumula evidència estadística sobre quines estratègies funcionen, amb diners virtuals.

---

## Les tres capes del sistema

```
┌──────────────────────────────────────────────────────────┐
│  CAPA 3 — SCRIPTS (punts d'entrada per a l'usuari)       │
│  optimize_bots  ·  optimize_models  ·  train_models      │
│  train_rl  ·  train_gate_regime  ·  run_comparison       │
│  run_demo  ·  download_data  ·  validate_data            │
└──────────────────────┬───────────────────────────────────┘
                       │ usa
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────┐             ┌─────────────────┐
│  DemoRunner  │             │  BacktestEngine │
│  (temps real)│             │  (simulació)    │
└──────┬───────┘             └────────┬────────┘
       │                              │
       └──────────────┬───────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  CAPA 2 — BOTS (estratègies)                             │
│                                                          │
│  classical/  →  HoldBot, DCABot, TrendBot, GridBot       │
│                 MeanReversionBot, MomentumBot             │
│                 EnsembleBot (meta-bot)                   │
│                                                          │
│  ml/         →  MLBot wrapping:                          │
│                 RandomForest, XGBoost, LightGBM          │
│                 CatBoost, GRU, PatchTST, TFT             │
│                                                          │
│  rl/         →  RLBot wrapping:                          │
│                 PPO, SAC, TD3 (baseline/on-chain/        │
│                 professional/multiframe)                 │
│                                                          │
│  gate/       →  GateBot (5 portes seqüencials)           │
│                 P1 Règim · P2 Salut · P3 Estructura      │
│                 P4 Momentum · P5 Risc                    │
└────────────────────────┬─────────────────────────────────┘
                         │  observation_schema()
                         ▼
┌──────────────────────────────────────────────────────────┐
│  CAPA 1 — DATA LAYER                                     │
│  ObservationBuilder  ←  FeatureBuilder / TechnicalInds   │
│  PostgreSQL (candles OHLCV + features calculades)        │
│  ExternalLoader (Fear&Greed, funding rate, on-chain)     │
└──────────────────────────────────────────────────────────┘

> **Nota sobre `merge_asof(direction='backward')`:** El `FeatureBuilder` alinea les fonts
> externes (Fear&Greed, funding rates, on-chain) amb les candles via
> `pd.merge_asof(direction='backward')`. Això significa que cada candle rep el valor
> **més recent del passat** de la font externa (backward-fill). No introdueix lookahead
> bias perquè sempre s'usa informació passada. Si la font externa no té dades recents,
> l'últim valor conegut es propaga endavant fins al primer nou registre.
```

---

## Flux d'un tick — pas a pas

```
1. INGESTA (cada hora, cron o manual)
   Binance API (ccxt) → candles OHLCV → PostgreSQL [taula: candles]

2. FEATURES (en memòria, per candle)
   candles → TechnicalIndicators.compute_features() → DataFrame
   Features: EMA-9/21/50/200, RSI-14, MACD, Bollinger, ATR-14, Volume-SMA
   (no es persisteixen — es calculen a demanda)

3. DEMO RUNNER (cada 60s en paper trading)
   Binance API → preu actual
     Per a cada bot actiu:
       ObservationBuilder.build() → finestra de N candles + features
       bot.on_observation(obs)    → Signal(action, size, confidence, reason)
       PaperExchange.send_order() → Order (fees 0.1% + slippage 0.01%)
       DemoRepository.save_*()   → PostgreSQL (signals, orders, ticks, trades)
       TelegramNotifier          → notificació de trade / status horari

4. BACKTEST ENGINE (offline, sobre dades històriques)
   Mateix flux però sobre totes les candles [TEST_FROM .. avui]
   → BacktestMetrics → Sharpe, Drawdown, Win Rate, Calmar
```

---

## El pattern BaseBot — com funciona l'extensibilitat

Tot bot implementa dues funcions:

```python
class MyBot(BaseBot):
    def observation_schema(self) -> ObservationSchema:
        """Declara les necessitats de dades. Es crida una vegada a l'inici."""
        return ObservationSchema(
            features   = ["close", "rsi_14", "atr_14"],
            timeframes = ["4h", "1d"],   # multi-timeframe suportat
            lookback   = 300,
        )

    def on_observation(self, observation: dict) -> Signal:
        """Lògica del bot. Es crida cada N candles (selon timeframe)."""
        df = observation["4h"]["features"]
        price = observation["4h"]["current_price"]
        # ... lògica ...
        return Signal(action=Action.BUY, size=0.5, confidence=0.8, reason="...")
```

El `DemoRunner` i el `BacktestEngine` no saben quin bot executen. Treballen sempre contra la interfície `BaseBot`. Afegir un bot nou = crear la classe + el YAML. **Cap script del core necessita modificació.**

---

## Walk-Forward Validation

El mètode per evitar el *lookahead bias* (entrenar amb dades del futur):

```
Dades disponibles:
  [2019-01-01 ────────────────── 2024-12-31  |  2025-01-01 ──── avui]
        TRAINING (optimització i entrenament)       TEST (mai vist)

Regla d'or: cap bot va a demo fins que en el període TEST
            supera HoldBot (buy & hold) en Sharpe i Calmar.
```

Configurat a `config/settings.yaml`:
```yaml
backtesting:
  train_until: "2024-12-31"
  test_from:   "2025-01-01"
```

---

## Auto-discovery de models

El sistema descobreix automàticament tots els models llegint `config/models/*.yaml`:

```python
from core.config_utils import discover_configs
ml_configs = discover_configs("ML")   # {"xgboost": "config/models/xgboost.yaml", ...}
```

Cada YAML declara:
```yaml
module:     bots.gate.gate_bot   # importació Python
class_name: GateBot              # nom de la classe
```

**Afegir un model = crear el YAML + la classe. Zero canvis als scripts existents.**

---

## Tots els bots disponibles

| Família | Bot | Senyal | Config |
|---------|-----|--------|--------|
| classic | **HoldBot** | BUY/HOLD — *benchmark* | `hold.yaml` |
| classic | **DCABot** | BUY/HOLD | `dca.yaml` |
| classic | **TrendBot** | BUY/SELL/HOLD — EMA crossover | `trend.yaml` |
| classic | **GridBot** | BUY/SELL/HOLD — Bollinger | `grid.yaml` |
| classic | **MeanReversionBot** | BUY/SELL/HOLD — Z-score | `mean_reversion.yaml` |
| classic | **MomentumBot** | BUY/SELL/HOLD — ROC/MACD | `momentum.yaml` |
| classic | **EnsembleBot** | BUY/SELL/HOLD — majority vote | `ensemble.yaml` |
| ML | **MLBot (Random Forest)** | Discret | `random_forest.yaml` |
| ML | **MLBot (XGBoost)** | Discret | `xgboost.yaml` |
| ML | **MLBot (LightGBM)** | Discret | `lightgbm.yaml` |
| ML | **MLBot (CatBoost)** | Discret | `catboost.yaml` |
| ML | **MLBot (GRU)** | Discret | `gru.yaml` |
| ML | **MLBot (PatchTST)** | Discret | `patchtst.yaml` |
| ML | **MLBot (TFT)** | Discret | `tft.yaml` |
| RL | **RLBot (PPO)** | Discret | `ppo.yaml` |
| RL | **RLBot (SAC)** | Continu | `sac.yaml` |
| RL | **RLBot (PPO on-chain)** | Discret | `ppo_onchain.yaml` |
| RL | **RLBot (SAC on-chain)** | Continu | `sac_onchain.yaml` |
| RL | **RLBot (PPO professional)** | Discret-5 | `ppo_professional.yaml` |
| RL | **RLBot (SAC professional)** | Continu | `sac_professional.yaml` |
| RL | **RLBot (TD3 professional)** | Continu | `td3_professional.yaml` |
| RL | **RLBot (TD3 multiframe)** | Continu | `td3_multiframe.yaml` |
| gate | **GateBot** | BUY/SELL/HOLD — swing, 5 portes | `gate.yaml` |

---

## Scripts — referència ràpida

| Script | Quan usar-lo | Comanda típica |
|--------|-------------|----------------|
| `download_data.py` | Primer cop i actualitzacions | `python scripts/download_data.py` |
| `validate_data.py` | Verificar integritat de dades | `python scripts/validate_data.py` |
| `optimize_bots.py` | Optuna per a bots clàssics | `python scripts/optimize_bots.py --bots dca trend grid` |
| `optimize_models.py` | Optuna per a ML + RL | `python scripts/optimize_models.py --no-rl --trials 15` |
| `train_models.py` | Entrenar models ML | `python scripts/train_models.py` |
| `train_rl.py` | Entrenar agents RL | `python scripts/train_rl.py` |
| `train_gate_regime.py` | Entrenar P1 del Gate System | `python scripts/train_gate_regime.py` |
| `run_comparison.py` | Backtesta i compara tots | `python scripts/run_comparison.py --all` |
| `run_demo.py` | Paper trading 24/7 | `python scripts/run_demo.py` |

---

## Cicle de vida complet

```
Fase 0 — Dades
  python scripts/download_data.py
          ↓
Fase 1 — Optimitzar hiperparàmetres
  optimize_bots.py  →  YAML [best_params] (bots clàssics)
  optimize_models.py →  YAML [best_params] (ML + RL)
          ↓
Fase 2 — Entrenar
  train_models.py   →  models/*.pkl / *.pt
  train_rl.py       →  models/*.zip
  train_gate_regime.py → models/gate_hmm.pkl + gate_xgb_regime.pkl
          ↓
Fase 3 — Validar
  run_comparison.py →  BacktestMetrics
  (Criteri mínim: Sharpe > 1.0, MaxDrawdown > -25% en TEST)
          ↓
Fase 4 — Demo
  Editar config/demo.yaml → enabled: true per als bots aprovats
  python scripts/run_demo.py → paper trading 24/7
```

---

## Mètriques de backtesting

| Mètrica | Fórmula | Bon valor |
|---------|---------|-----------|
| **Sharpe Ratio** | `mean(returns) / std(returns) × √(252×24)` | > 1.0 |
| **Max Drawdown** | Pitjor caiguda pic → vall (%) | > −20% |
| **Calmar Ratio** | `retorn_anual / abs(max_drawdown)` | > 0.5 |
| **Win Rate** | % trades guanyadors (round-trips buy→sell) | > 50% |
| **Total Return** | Guany/pèrdua total (%) sobre capital inicial | Positiu |

> Les mètriques usen `√(252×24)` per anualitzar dades horàries.

---

## Estructura completa de carpetes

```
btc-trading-bot/
├── bots/
│   ├── classical/          DCABot, TrendBot, GridBot, HoldBot, MeanReversionBot, MomentumBot, EnsembleBot
│   ├── ml/                 MLBot + models (RF, XGBoost, LightGBM, CatBoost, GRU, PatchTST, TFT)
│   ├── rl/                 RLBot + agents (PPO, SAC, TD3) + environments + rewards + constants
│   └── gate/               GateBot + 5 portes (P1–P5) + regime_models (HMM, XGBoost) + near_miss_logger
├── config/
│   ├── models/             UN YAML per model (paràmetres Optuna dins best_params)
│   ├── exchanges/          Configuració del paper exchange (fees, slippage)
│   ├── settings.yaml       Globals (BD, exchange, dates walk-forward)
│   └── demo.yaml           Quins bots s'executen al DemoRunner
├── core/
│   ├── backtesting/        BacktestEngine, Comparator, Optimizers, agent_validator
│   ├── engine/             Runner (backtest), DemoRunner (paper trading 24/7)
│   ├── db/                 models.py (SQLAlchemy), session.py, repository.py
│   ├── interfaces/         BaseBot, BaseMLModel, BaseRLAgent, BaseExchange, BaseStrategy
│   ├── config_utils.py     apply_best_params, discover_configs (auto-discovery)
│   └── models.py           Candle, Signal, Order, Trade (Pydantic)
├── data/
│   ├── processing/         FeatureBuilder, DatasetBuilder, TechnicalIndicators, ExternalLoader
│   └── observation/        ObservationBuilder (construeix l'obs per a cada bot)
├── exchanges/
│   └── paper.py            PaperExchange (simulació amb fees + slippage)
├── monitoring/
│   └── dashboard.py        Streamlit dashboard de visualització
├── scripts/                Tots els scripts d'entrada (descrits a la taula anterior)
├── tests/
│   ├── smoke/              66 tests, zero deps externes
│   ├── unit/               57 tests amb dades mock (inclou 54 del Gate System)
│   └── integration/        5 tests (necessita PostgreSQL)
├── models/                 Models entrenats: *.pkl, *.pt, *.zip
└── docs/                   Documentació completa (estàs aquí)
```
