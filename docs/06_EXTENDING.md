# Com Estendre el Projecte — Receptes

> Instruccions per afegir nous bots, models, agents i fonts de dades.
> Per al schema complet dels YAMLs: veure **[04_CONFIGURATION.md](./04_CONFIGURATION.md)**.
> Per a l'arquitectura del BaseBot pattern: veure **[01_ARCHITECTURE.md](./01_ARCHITECTURE.md)**.

---

## Auto-discovery: el principi fonamental

El sistema usa **auto-discovery** per poblar tots els registres de models.
Els scripts `train_models.py`, `run_comparison.py` i `ml_bot.py` llegeixen
`config/models/*.yaml` a l'inici i construeixen els seus registres dinàmicament,
sense cap llista hardcodejada.

```python
# core/config_utils.discover_configs()
from core.config_utils import discover_configs
ml_configs  = discover_configs("ML")      # {"xgboost": "config/models/xgboost.yaml", ...}
all_configs = discover_configs()          # tots els bots i models
```

Cada YAML declara dos camps per a la càrrega dinàmica:

```yaml
module:     bots.ml.xgboost_model    # camí d'importació Python
class_name: XGBoostModel             # nom de la classe
```

**Afegir un model o bot = crear el YAML + la classe Python. Cap script necessita modificació.**

---

## 1. Afegir un Bot Clàssic

**Passos (2 essencials + 1 opcional):**

**1.1** Crea `bots/classical/my_bot.py`:

```python
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action
import pandas as pd

class MyBot(BaseBot):
    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=["close", "rsi_14", "ema_9"],
            timeframes=["1h"],
            lookback=50,
        )

    def on_observation(self, observation: dict) -> Signal:
        df = observation["1h"]["features"]
        param1 = self.config.get("param1", 10)
        # La teva lògica aquí
        return Signal(bot_id=self.bot_id, timestamp=..., action=Action.HOLD, ...)

    def on_start(self) -> None:
        pass
```

**1.2** Crea `config/models/my_bot.yaml`:

```yaml
category:   classic
model_type: my_bot
module:     bots.classical.my_bot    # ← auto-discovery
class_name: MyBot                    # ← auto-discovery

bot_id:    my_bot_v1
symbol:    BTC/USDT
timeframe: 1h
lookback:  50

param1: 10   # paràmetres llegits via self.config

optimization:
  n_trials: 50
  metric: sharpe_ratio
  direction: maximize
  search_space:
    param1: {type: int, low: 5, high: 50}
```

**1.3** (Opcional) Afegeix a `config/demo.yaml`:

```yaml
bots:
  - config_path: config/models/my_bot.yaml
    enabled: true
```

Ja apareix automàticament a `run_comparison.py` i `optimize_bots.py`.

---

## 2. Afegir un Model ML

**Passos (2 essencials + 1 opcional):**

**2.1** Crea `bots/ml/my_model.py`:

```python
# Opció A: model tree-based → hereta BaseTreeModel (implementa train/predict/save/load)
from bots.ml.base_tree_model import BaseTreeModel
from my_library import MyClassifier

class MyModel(BaseTreeModel):
    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.model = MyClassifier(n_estimators=n_estimators)

    @classmethod
    def from_config(cls, cfg: dict) -> "MyModel":
        return cls(**cfg.get("model", {}))

    def _get_model_label(self) -> str:       return "MY"


# Opció B: model DL → usa TimeSeriesDataset compartit
from data.processing.torch_dataset import TimeSeriesDataset
from core.interfaces.base_ml_model import BaseMLModel
from torch.utils.data import DataLoader

class MyDLModel(BaseMLModel):
    def train(self, X, y):
        ds = TimeSeriesDataset(X.values, y.values, seq_len=self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        ...
```

**2.2** Crea `config/models/my_model.yaml`:

```yaml
category:   ML
model_type: my_model
module:     bots.ml.my_model     # ← auto-discovery
class_name: MyModel              # ← auto-discovery

symbol:     BTC/USDT
timeframes: [1h]
forward_window: 24
threshold_pct:  0.005

features:
  select: null   # null = totes; llista = subset de columnes
  external: {}

training:
  experiment_name: my_model_v1
  model_path: models/my_model_v1.pkl
  model:
    n_estimators: 100

optimization:
  n_trials: 20
  metric: accuracy_mean
  direction: maximize
  search_space:
    n_estimators: {type: int, low: 50, high: 500}

bot:
  bot_id: ml_my_model_v1
  lookback: 200
  min_confidence: 0.4
  trade_size: 0.5
  prediction_threshold: 0.35
```

**2.3** (Opcional) Entrena i compara:

```bash
python scripts/train_models.py --models my_model
python scripts/run_comparison.py --bots hold my_model
```

---

### `BaseTreeModel` — scaffold compartit

`bots/ml/base_tree_model.py` proporciona el bucle d'entrenament complet per a models tree-based (RF, XGB, LGB, CB comparteixen la mateixa implementació):

- TimeSeriesSplit 5-fold CV amb tqdm
- StandardScaler + DataFrames amb feature names
- `save()` / `load()` via pickle `{model, scaler, feature_names}`
- `predict()` amb threshold configurable

Els models fills implementen **2 mètodes abstractes d'1 línia** + `__init__` + `from_config()`. Tot el bucle d'entrenament és heretat.

### `TimeSeriesDataset` — Dataset PyTorch compartit

`data/processing/torch_dataset.py` conté la classe que GRU, PatchTST i qualsevol model DL futur comparteixen:

```python
from data.processing.torch_dataset import TimeSeriesDataset

ds = TimeSeriesDataset(X.values, y.values, seq_len=self.seq_len)
# ds[i] → (tensor[seq_len, n_features], tensor[label])
```

---

## 3. Afegir un Agent RL

**Passos (4):**

**3.1** Crea `bots/rl/agents/my_agent.py`:

```python
from core.interfaces.base_rl_agent import BaseRLAgent
from core.backtesting.agent_validator import validate_agent

class MyAgent(BaseRLAgent):
    def train(self, env, total_timesteps: int = 500_000) -> None:
        self.model = A2C("MlpPolicy", env, learning_rate=self.learning_rate, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, observation, deterministic: bool = True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def _validate(self, env) -> dict:
        return validate_agent(self, env)   # ← cap duplicació, compartit amb PPO/SAC/TD3

    def save(self, path): self.model.save(path)
    def load(self, path): self.model = A2C.load(path)
```

**3.2** Afegeix als registres de `trainer.py` i `rl_bot.py` (2 fitxers, 1 línia cadascun):

```python
from bots.rl.agents.my_agent import MyAgent

_AGENT_REGISTRY = { ..., "my_agent": MyAgent }
_ENV_REGISTRY   = { ..., "my_agent": BtcTradingEnvDiscrete }  # o entorn custom
```

**3.3** Crea `config/models/my_agent.yaml`:

```yaml
category:   RL
model_type: my_agent
symbol:    BTC/USDT
timeframe: 1h

# CRÍTIC: obs_shape = len(select) × lookback — mai canviar sense re-entrenar
features:
  lookback: 96
  select: [close, rsi_14, macd, atr_14, ema_9, ema_21, bb_upper, bb_lower, volume]
  external: {}

training:
  experiment_name: my_agent_v1
  train_pct: 0.8
  model_path: models/my_agent_v1
  environment:
    initial_capital: 10000.0
    fee_rate: 0.001
    reward_scaling: 100.0
    reward_type: sharpe
  model:
    total_timesteps: 500000
    learning_rate: 0.0003

optimization:
  n_trials: 10
  probe_timesteps: 20000
  metric: val_return_pct
  direction: maximize
  search_space:
    learning_rate: {type: float, low: 0.0001, high: 0.001, log: true}

bot:
  bot_id: rl_my_agent_v1
  trade_size: 0.5
```

**3.4** Afegeix a `scripts/train_rl.py`:

```python
AVAILABLE_AGENTS = { ..., "my_agent": "config/models/my_agent.yaml" }
```

> **Nota:** els agents RL no usen `discover_configs()` per al registry de classes
> (`_AGENT_REGISTRY` / `_ENV_REGISTRY`) perquè el mapatge entorn ↔ agent és
> un disseny explícit que el sistema no pot inferir automàticament del YAML.
> Sí que hi apareixen automàticament a `run_comparison.py` (BOT_REGISTRY).

⚠️ **Regla crítica `obs_shape`:** si canvies `features.select` o `features.lookback`, **has de re-entrenar el model**. Mai modificar-los en un model ja entrenat.

---

### `validate_agent()` — validació compartida

`core/backtesting/agent_validator.py` centralitza la lògica de validació que PPO, SAC i TD3 comparteixen:

```python
from core.backtesting.agent_validator import validate_agent

def _validate(self, env) -> dict:
    return validate_agent(self, env)
# Retorna: {val_return_pct, val_max_drawdown_pct, val_trades, val_final_capital}
```

---

## 4. Afegir un Indicador Tècnic

**4.1** Afegeix a `TechnicalIndicators` (`data/processing/technical.py`):

```python
@staticmethod
def my_indicator(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return df["close"].rolling(period).mean()
```

**4.2** Afegeix-lo a `compute_features()` al mateix fitxer.

**4.3** Declara-la a `features.select` del YAML del bot que la necessita.

---

## 5. Afegir una Font de Dades Externa

Fonts ja implementades (referència per al patró):

| Font | Fitxer | Taula BD |
|------|--------|----------|
| Fear & Greed | `data/sources/fear_greed.py` | `fear_greed` |
| Funding Rate + OI | `data/sources/futures.py` | `funding_rates`, `open_interest` |
| Hash rate, on-chain | `data/sources/blockchain.py` | `blockchain_metrics` |
| Open Interest 5m | `data/sources/binance_vision.py` | `open_interest` |

**Passos (8):**

1. Model Pydantic a `core/models.py` (validació de dades entrants)
2. Model SQLAlchemy a `core/db/models.py` (esquema de taula)
3. Fetcher a `data/sources/my_source.py` (descàrrega + `update()` incremental)
4. Migració Alembic: `alembic revision --autogenerate -m "add_my_external"`
5. Loader a `data/processing/external.py` (retorna DataFrame UTC-indexed)
6. Integra a `FeatureBuilder` (`data/processing/feature_builder.py`)
7. Activa al YAML amb `features.external.my_source: true`
8. Crea `scripts/download_my_source.py` + `scripts/update_my_source.py`

> ⚠️ Fonts amb historial limitat (p. ex. `open_interest` REST = darrers 30 dies)
> provoquen un `dropna()` massiu. Usa la versió Binance Vision S3 (des de 2021)
> per a lookbacks llargs.

---

## 6. EnsembleBot — ✅ Implementat

`EnsembleBot` és a `bots/classical/ensemble_bot.py`. Funciona via auto-discovery
com qualsevol bot clàssic (camps `module` + `class_name` al YAML).

**Polítiques disponibles:**

| Política | Estat | Quan usar |
|---------|-------|-----------|
| `majority_vote` | ✅ implementada | Punt de partida — > 50% sub-bots han d'acordar |
| `weighted` | 🔜 futur | Quan alguns bots clarament millors (Sharpe sliding window) |
| `stacking` | 🔜 futur | Quan tens historial suficient per entrenar meta-capa ML |

**Afegir o treure sub-bots:** edita `config/models/ensemble.yaml` (secció `sub_bots`)
i reinicia el DemoRunner. No cal modificar cap fitxer Python.

```yaml
# config/models/ensemble.yaml
sub_bots:
  - config/models/trend.yaml         # classical
  - config/models/xgboost.yaml       # ML (ha d'estar entrenat)
  - config/models/ppo_professional.yaml  # RL (ha d'estar entrenat)
```

L'EnsembleBot detecta automàticament el tipus de cada sub-bot (classic / ML / RL)
pel camp `category` del seu YAML — sense hard-coding ni registres manuals.

**Criteris mínims per afegir un sub-bot a l'ensemble:**
Sharpe > 1.0 i Drawdown < −25% en backtest out-of-sample (period TEST_FROM en endavant).

---

---

## 7. Estendre el Gate System

El Gate System és un bot independent a `bots/gate/`. Pots provar variacions de portes individuals o implementar un GateBot v2 sense tocar res existent.

---

### 7.1 Canviar els paràmetres d'una porta sense re-entrenar

Tots els paràmetres de P3, P4 i P5 llegits del YAML (`gate.yaml`) es poden modificar directament:

```yaml
# config/models/gate.yaml

p3:
  min_level_strength: 0.5   # era 0.4 — més estricte
  fractal_n_4h: 3           # era 2 — fractals menys sensibles

p4:
  rsi2_oversold: 15         # era 10 — llindar RSI-2 menys extrem

p5:
  max_risk_pct: 0.005       # era 0.01 — 0.5% de risc per trade (més conservador)
  min_rr:
    STRONG_BULL: 2.0        # era 1.5 — R:R més estricte
```

P1 i P2 no llegeixen paràmetres crítics del YAML (els models ja estan entrenats). Reiniciar el DemoRunner aplica els nous valors.

⚠️ Si canvies `features_4h`, `features_1d`, `lookback` o `timeframe`, cal re-entrenar.

---

### 7.2 Afegir un nou senyal a P4

**Exemple:** afegir un 5è senyal (Stochastic RSI < 20).

**7.2.1** A `bots/gate/gates/p4_momentum.py`, afegeix el càlcul i inclou-lo als resultats:

```python
# En _compute_signals():
stoch_rsi = self._stoch_rsi(df_4h)
stoch_ok = bool(stoch_rsi.iloc[-1] < 20)

return P4Result(
    signals_detail={..., "stoch_rsi": stoch_ok},
    signals_total=5,   # era 4
    ...
)
```

**7.2.2** Actualitza `_MIN_SIGNALS` si cal:
```python
_MIN_SIGNALS = {
    "STRONG_BULL": 1,
    "WEAK_BULL": 2,    # potser ara exigeixes 3 de 5
    "RANGING": 3,
}
```

No cal canviar cap YAML ni re-entrenar.

---

### 7.3 Afegir un nou component a P2

P2 retorna un `position_multiplier [0,1]` com a producte de sub-scores. Per afegir un 3r factor (ex: news sentiment):

```python
# bots/gate/gates/p2_health.py

def _news_score(self, news_sentiment: float) -> float:
    """news_sentiment ∈ [-1, 1]. Negatiu → reduir; positiu → amplificar."""
    if news_sentiment < -0.5:
        return 0.5   # notícies molt negatives → reduir a la meitat
    elif news_sentiment > 0.5:
        return 1.0   # notícies positives → pes complet
    return 0.8

def evaluate(self, df_1d, regime, news_sentiment=0.0) -> float:
    fg_score      = self._fg_score(df_1d, regime)
    funding_score = self._funding_score(df_1d)
    news_score    = self._news_score(news_sentiment)
    return fg_score * funding_score * news_score
```

Actualitza `gate_bot.py` per passar el valor de sentiment a `p2.evaluate()`.

---

### 7.4 Implementar GateBot v2 (nova instància en paral·lel)

Per provar una variació important (ex: incluure shorts, 3 timeframes) sense interferir amb gate_v1:

**7.4.1** Crea `bots/gate_v2/` amb la nova implementació (o hereta de `GateBot`).

**7.4.2** Crea `config/models/gate_v2.yaml`:
```yaml
category:   gate
model_type: gate_v2
module:     bots.gate_v2.gate_bot_v2
class_name: GateBotV2
bot_id:     gate_v2
...
model_paths:
  hmm: models/gate_v2_hmm.pkl
  xgb: models/gate_v2_xgb.pkl
```

**7.4.3** Entrena:
```bash
python scripts/train_gate_regime.py --config config/models/gate_v2.yaml
```

**7.4.4** Afegeix a `config/demo.yaml`:
```yaml
- config_path: config/models/gate_v2.yaml
  enabled: false   # activa quan estigui entrenat
```

Tots dos (gate_v1 i gate_v2) corren en paral·lel i les seves posicions van a taules independents gràcies al camp `bot_id`.

---

### 7.5 Canviar P1: provar un altre classificador de règim

P1 usa HMM per descobrir estats + XGBoost per classificar. Per provar LightGBM com a classificador:

**7.5.1** Crea `bots/gate/regime_models/lgbm_classifier.py` seguint el mateix patró que `xgb_classifier.py` (implementa `fit`, `predict_proba`, `save`, `load`).

**7.5.2** A `bots/gate/gates/p1_regime.py`, afegeix lògica per carregar el classificador alternatiu:
```python
classifier_type = self.config.get("p1", {}).get("classifier", "xgb")
if classifier_type == "lgbm":
    self._clf = LGBMRegimeClassifier()
    self._clf.load(paths["lgbm"])
```

**7.5.3** Actualitza `gate.yaml`:
```yaml
model_paths:
  hmm: models/gate_hmm.pkl
  lgbm: models/gate_lgbm_regime.pkl
p1:
  classifier: lgbm   # "xgb" per defecte
```

**7.5.4** Actualitza `train_gate_regime.py` per entrenar i guardar el model alternatiu.

---

*Última actualització: Març 2026 · Versió 4.0 (Gate System afegit)*
