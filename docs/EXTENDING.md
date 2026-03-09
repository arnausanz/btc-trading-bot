# Com Estendre el Projecte — Receptes

> Instruccions concretes per afegir nous bots, models, agents i fonts de dades.
> El sistema és dissenyat per ser extensible per herència i registre.
> Per al schema complet dels YAMLs: veure **[CONFIGURATION.md](./CONFIGURATION.md)**.

---

## 1. Afegir un Bot Clàssic

Un bot clàssic implementa regles deterministes (sense entrenament).
**Un sol YAML** conté la config base i el search space d'Optuna.

**Passos:**

**1.1** Crea el fitxer `bots/classical/my_bot.py`:
```python
from core.interfaces.base_bot import BaseBot
from core.interfaces.base_strategy import BaseStrategy
from core.models import Signal, Action
from core.interfaces.base_bot import ObservationSchema
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, param1: int = 10):
        self.param1 = param1

    def generate_signal(self, df: pd.DataFrame, bot_id: str) -> Signal:
        # La teva lògica aquí
        # df té columnes: open, high, low, close, volume, ema_9, rsi_14, etc.
        action = Action.HOLD
        reason = "No signal"
        return Signal(
            bot_id=bot_id,
            timestamp=df.iloc[-1]["timestamp"],
            action=action,
            size=0.5,
            confidence=0.7,
            reason=reason,
        )

class MyBot(BaseBot):
    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=["close", "rsi_14", "ema_9"],  # features que necessites
            timeframes=["1h"],
            lookback=50,
        )
```

**1.2** Crea `config/models/my_bot.yaml` (YAML unificat):
```yaml
category: classic
model_type: my_bot       # ← ha de coincidir amb la clau del diccionari _CLASSICAL
bot_id: my_bot_v1
symbol: BTC/USDT
timeframe: 1h
lookback: 50
features: [close, rsi_14, ema_9]

# Paràmetres de la estratègia (llegits directament pel bot)
param1: 10

# Search space per Optuna
optimization:
  n_trials: 50
  metric: sharpe_ratio
  direction: maximize
  search_space:
    param1:
      type: int
      low: 5
      high: 50
```

**1.3** Afegeix a `scripts/optimize_bots.py`:
```python
from bots.classical.my_bot import MyBot

ALL_BOTS = {
    "dca":            {"class": DCABot,           "config": "config/models/dca.yaml"},
    "trend":          {"class": TrendBot,          "config": "config/models/trend.yaml"},
    "mean_reversion": {"class": MeanReversionBot,  "config": "config/models/mean_reversion.yaml"},
    "momentum":       {"class": MomentumBot,       "config": "config/models/momentum.yaml"},
    "my_bot":         {"class": MyBot,             "config": "config/models/my_bot.yaml"},  # ← nou
}
```

**1.4** Afegeix a `scripts/run_comparison.py`:
```python
BOT_REGISTRY = {
    ...
    "my_bot": "config/models/my_bot.yaml",   # ← nou
}

# I al diccionari _CLASSICAL dins _instantiate_bot():
_CLASSICAL = {
    "hold": HoldBot, "dca": DCABot,
    "trend": TrendBot, "grid": GridBot,
    "mean_reversion": MeanReversionBot,
    "momentum": MomentumBot,
    "my_bot": MyBot,                         # ← nou
}
```

**1.5** Afegeix a `config/demo.yaml` (si vols al demo):
```yaml
bots:
  - config_path: config/models/my_bot.yaml
    enabled: true
```

---

## 2. Afegir un Model ML

Un model ML s'integra via `MLBot` com a backend intercambiable.
**Un sol YAML** conté tota la configuració: features, entrenament, Optuna i desplegament.

**Passos:**

**2.1** Crea `bots/ml/my_model.py`:
```python
from core.interfaces.base_ml_model import BaseMLModel
import pandas as pd
import numpy as np

class MyModel(BaseMLModel):
    def __init__(self, param1: int = 100, **kwargs):
        self.param1 = param1
        self.model = None

    @classmethod
    def from_config(cls, training_cfg: dict) -> "MyModel":
        return cls(**training_cfg.get("model", {}))

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        # Entrena el model, retorna mètriques
        # {"accuracy_mean": ..., "precision_mean": ..., "recall_mean": ...}
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Retorna probabilitats [0.0 – 1.0]
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(path)
```

**2.2** Afegeix al registry de `bots/ml/ml_bot.py`:
```python
from bots.ml.my_model import MyModel

_MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost":       XGBoostModel,
    ...
    "my_model": MyModel,   # ← model_type que posem al YAML
}
```

**2.3** Crea `config/models/my_model.yaml` (YAML unificat):
```yaml
category: ML
model_type: my_model     # ← clau al _MODEL_REGISTRY
symbol: BTC/USDT
timeframes: [1h]
forward_window: 24
threshold_pct: 0.005

features:
  select: null           # null = totes; llista = subset de columnes
  external: {}           # fonts externes (fear_greed, funding_rate, etc.)

training:
  experiment_name: my_model_v1
  model_path: models/my_model_v1.pkl
  model:
    param1: 100

optimization:
  n_trials: 20
  metric: accuracy_mean
  direction: maximize
  search_space:
    param1:
      type: int
      low: 50
      high: 300

bot:
  bot_id: ml_my_model_v1
  lookback: 200
  min_confidence: 0.4
  trade_size: 0.5
  prediction_threshold: 0.35
```

**2.4** Afegeix a `scripts/train_models.py`:
```python
ALL_CONFIGS = {
    ...
    "my_model": "config/models/my_model.yaml",   # ← nou
}
_KEY_TO_MODEL_TYPE = {
    ...
    "my_model": "my_model",                      # ← nou (clau → model_type al YAML)
}
```

**2.5** Afegeix a `scripts/optimize_models.py`:
```python
ALL_ML_CONFIGS = {
    ...
    "my_model": "config/models/my_model.yaml",   # ← nou
}
```

**2.6** Afegeix a `scripts/run_comparison.py`:
```python
BOT_REGISTRY = {
    ...
    "my_model": "config/models/my_model.yaml",   # ← nou
}
```

Ara `optimize_models.py`, `train_models.py` i `run_comparison.py` el detecten automàticament.

---

## 3. Afegir un Agent RL

**Un sol YAML** unificat per a entrenament, Optuna i desplegament.

**Passos:**

**3.1** Crea `bots/rl/agents/my_agent.py`:
```python
from core.interfaces.base_rl_agent import BaseRLAgent
from stable_baselines3 import A2C  # o qualsevol SB3 agent

class MyAgent(BaseRLAgent):
    def __init__(self, learning_rate: float = 3e-4, **kwargs):
        self.learning_rate = learning_rate
        self.model = None

    @classmethod
    def from_config(cls, training_cfg: dict) -> "MyAgent":
        return cls(**training_cfg.get("model", {}))

    def train(self, env, total_timesteps: int = 500_000) -> None:
        self.model = A2C("MlpPolicy", env, learning_rate=self.learning_rate, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation) -> tuple:
        return self.model.predict(observation, deterministic=True)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = A2C.load(path)
```

**3.2** Afegeix al registry de `bots/rl/rl_bot.py`:
```python
from bots.rl.agents.my_agent import MyAgent

_AGENT_REGISTRY = {
    "ppo":      PPOAgent,
    "sac":      SACAgent,
    "my_agent": MyAgent,   # ← nou
}
```

**3.3** Crea `config/models/my_agent.yaml` (YAML unificat):
```yaml
category: RL
model_type: my_agent    # ← clau al _AGENT_REGISTRY
symbol: BTC/USDT
timeframe: 1h

# CRÍTIC: obs_shape = len(select) × lookback
# Ha de ser idèntic entre entrenament i desplegament.
features:
  lookback: 96
  select:
    - close
    - rsi_14
    - macd
    - atr_14
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
    policy: MlpPolicy

optimization:
  n_trials: 10
  probe_timesteps: 20000
  metric: val_return_pct
  direction: maximize
  search_space:
    learning_rate: {type: float, low: 0.0001, high: 0.001, log: true}
    reward_type: {type: categorical, choices: [simple, sharpe, sortino]}

bot:
  bot_id: rl_my_agent_v1
  trade_size: 0.5
```

**3.4** Afegeix a `scripts/train_rl.py`:
```python
AVAILABLE_AGENTS = {
    ...
    "my_agent": "config/models/my_agent.yaml",   # ← nou
}
```

**3.5** Afegeix a `scripts/run_comparison.py`:
```python
BOT_REGISTRY = {
    ...
    "my_agent": "config/models/my_agent.yaml",   # ← nou
}
```

⚠️ **Regla crítica `obs_shape`:** si canvies `features.select` o `features.lookback` al YAML, has de re-entrenar el model. Mai canviar-los sense re-entrenar — causaria `ValueError: Unexpected observation shape`.

---

## 4. Afegir un Indicador Tècnic

Els indicadors es calculen a `data/processing/technical_indicators.py`.

**Passos:**

**4.1** Afegeix el mètode a `TechnicalIndicators`:
```python
@staticmethod
def my_indicator(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula el meu indicador."""
    return df["close"].rolling(period).mean()  # exemple
```

**4.2** Afegeix a `compute_features()`:
```python
def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
    ...
    df["my_indicator_14"] = self.my_indicator(df, 14)
    return df
```

**4.3** Declara-la a `features.select` del YAML del bot que la necessita:
```yaml
# config/models/my_bot.yaml
features:
  select:
    - close
    - rsi_14
    - my_indicator_14   # ← la nova feature
```

O per a bots clàssics, a l'`observation_schema()` del bot:
```python
def observation_schema(self):
    return ObservationSchema(
        features=["close", "rsi_14", "my_indicator_14"],
        ...
    )
```

---

## 5. Afegir una Font de Dades Externa

El sistema ja té infraestructura completa per a fonts externes. Segueix el patró establert.

**Fonts ja implementades** (com a referència):
- `data/sources/fear_greed.py` → Fear & Greed Index (diari)
- `data/sources/futures.py` → Funding Rate + Open Interest (8h / 1h)
- `data/sources/blockchain.py` → Hash rate, adreces, fees (diari)
- `data/sources/binance_vision.py` → Open Interest (5m, historial S3)

**Passos per afegir una nova font:**

**5.1** Crea el model Pydantic a `core/models.py` (validació):
```python
class MyExternalData(BaseModel):
    timestamp: datetime
    value: float

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
```

**5.2** Crea el model SQLAlchemy a `core/db/models.py`:
```python
class MyExternalDB(Base):
    __tablename__ = "my_external"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (UniqueConstraint("timestamp", name="uq_my_external_ts"),)
```

**5.3** Crea el fetcher `data/sources/my_source.py`:
```python
class MySourceFetcher:
    def fetch_and_store(self, since: datetime | None = None) -> int:
        """Descàrrega + validació + save. Idempotent."""
        ...

    def _save(self, entries: list[MyExternalData]) -> int:
        """Bulk insert, omiteix duplicats via set de timestamps existents."""
        ...

    def get_last_timestamp(self) -> datetime | None:
        ...

    def update(self) -> int:
        """Incremental: si BD buida → tot l'historial, sinó → darrers N registres."""
        ...
```

**5.4** Crea la migració Alembic:
```bash
alembic revision --autogenerate -m "add_my_external"
alembic upgrade head
```

**5.5** Afegeix el loader a `data/processing/external.py`:
```python
def load_my_external() -> pd.DataFrame:
    """Retorna DataFrame UTC-indexed amb columna 'my_value'."""
    with SessionLocal() as session:
        rows = session.query(MyExternalDB).order_by(MyExternalDB.timestamp).all()
    df = pd.DataFrame([{"timestamp": r.timestamp, "my_value": r.value} for r in rows])
    return df.set_index("timestamp")
```

**5.6** Integra a `FeatureBuilder` (a `data/processing/feature_builder.py`):
```python
if self.external_cfg.get("my_source"):
    ext = load_my_external()
    df = pd.merge_asof(df, ext, left_index=True, right_index=True, direction="backward")
```

**5.7** Activa al YAML del model que la necesita:
```yaml
features:
  external:
    my_source: true
  select:
    - close
    - rsi_14
    - my_value   # ← nom de la columna afegida pel loader
```

**5.8** Crea els scripts de càrrega inicial i update:
- `scripts/download_my_source.py` — descàrrega inicial (1 sola vegada)
- `scripts/update_my_source.py` — actualització incremental (cron)

---

## 6. Afegir un EnsembleBot (ROADMAP)

> Pendent d'implementar. Disseny previst:

```python
class EnsembleBot(BaseBot):
    """Combina senyals de múltiples sub-bots."""

    def __init__(self, bots: list[BaseBot], policy: str = "majority_vote"):
        self.bots = bots
        self.policy = policy  # majority_vote | weighted | unanimous | stacking

    def on_observation(self, obs) -> Signal:
        signals = [bot.on_observation(obs) for bot in self.bots]
        return self._aggregate(signals)

    def _aggregate(self, signals: list[Signal]) -> Signal:
        if self.policy == "majority_vote":
            actions = [s.action for s in signals]
            # Retorna l'acció majoritària
            ...
        elif self.policy == "weighted":
            # Pes proporcional al Sharpe dels últims N dies
            ...
```

**Polítiques d'ensemble:**

| Política | Quan usar |
|---------|-----------|
| `majority_vote` | Diversitat alta entre bots |
| `weighted` | Alguns bots clarament millors que altres |
| `unanimous` | Conservador; poques operacions |
| `stacking` | Quan tens prou dades de predictions per entrenar una meta-capa |
| `dynamic_routing` | Quan saps detectar el règim de mercat |

---

*Última actualització: Març 2026 · Versió 2.0*
