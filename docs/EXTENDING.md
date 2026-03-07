# Com Estendre el Projecte — Receptes

> Instruccions concretes per afegir nous bots, models, agents i fonts de dades.
> El sistema és dissenyat per ser extensible per herència i registre.

---

## 1. Afegir un Bot Clàssic

Un bot clàssic implementa regles deterministes (sense entrenament).

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

**1.2** Crea `config/bots/my_bot.yaml`:
```yaml
bot_id: my_bot_v1
bot_class: bots.classical.my_bot.MyBot
strategy_class: bots.classical.my_bot.MyStrategy
params:
  param1: 10
```

**1.3** Afegeix a `scripts/optimize_bots.py` (si vols optimitzar):
```python
ALL_BOTS = {
    "dca": "config/bots/dca.yaml",
    "trend": "config/bots/trend.yaml",
    "my_bot": "config/bots/my_bot.yaml",   # ← afegeix aquí
}
```

**1.4** Afegeix a `scripts/run_comparison.py`:
```python
BOT_CONFIGS = {
    ...
    "my_bot": get_best_bot_config_path("my_bot"),  # ← afegeix aquí
}
```

**1.5** Afegeix a `config/demo.yaml` (si vols al demo):
```yaml
bots:
  - config/bots/my_bot.yaml
```

---

## 2. Afegir un Model ML

Un model ML s'integra via `MLBot` com a backend intercambiable.

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

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Entrena el model aquí
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
_MODEL_REGISTRY = {
    "rf": RandomForestModel,
    "xgb": XGBoostModel,
    ...
    "my_model": MyModel,   # ← afegeix aquí
}
```

**2.3** Crea `config/training/my_model_experiment_1.yaml`:
```yaml
model_type: my_model
params:
  param1: 100
train_until: "2024-12-31"
target: price_up_1pct_in_24h
```

**2.4** Crea `config/optimization/my_model.yaml`:
```yaml
model_type: my_model
n_trials: 15
metric: sharpe_ratio
search_space:
  param1:
    type: int
    low: 50
    high: 300
```

**2.5** Afegeix a `optimize_models.py`:
```python
ALL_ML_CONFIGS = {
    ...
    "my_model": "config/optimization/my_model.yaml",
}
```

Ara `optimize_models.py`, `train_models.py` i `run_comparison.py` el detecten automàticament.

---

## 3. Afegir un Agent RL

**Passos:**

**3.1** Crea `bots/rl/agents/my_agent.py`:
```python
from core.interfaces.base_rl_agent import BaseRLAgent
from stable_baselines3 import A2C  # o qualsevol SB3 agent

class MyAgent(BaseRLAgent):
    def __init__(self, learning_rate: float = 3e-4, **kwargs):
        self.learning_rate = learning_rate
        self.model = None

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
_AGENT_REGISTRY = {
    "ppo": PPOAgent,
    "sac": SACAgent,
    "my_agent": MyAgent,   # ← afegeix aquí
}
```

**3.3** Crea `config/training/my_agent_experiment_1.yaml` i `config/optimization/my_agent.yaml` seguint el mateix patró que `ppo` o `sac`.

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

**4.3** Declara-la a l'`observation_schema()` del bot que la necessita:
```python
def observation_schema(self):
    return ObservationSchema(
        features=["close", "rsi_14", "my_indicator_14"],
        ...
    )
```

---

## 5. Afegir una Font de Dades Externa

> ⚠️ La infraestructura per a fonts externes és un **ROADMAP item** (pendent d'implementar).
> Instruccions provisionals per al que existeix ara.

**Situació actual:** L'`ObservationSchema` té un camp `extras: dict` però l'`ObservationBuilder` no el processa encara. Per ara, la manera d'afegir dades externes és com a columna addicional al DataFrame de features.

**Exemple Fear & Greed Index (provisional):**

**5.1** Crea `data/sources/fear_greed.py`:
```python
import requests
import pandas as pd

def fetch_fear_greed_history() -> pd.DataFrame:
    """Descàrrega l'historial del Fear & Greed Index (gratuït)."""
    url = "https://api.alternative.me/fng/?limit=365&format=json"
    resp = requests.get(url).json()
    records = [{"timestamp": pd.Timestamp(int(r["timestamp"]), unit="s", tz="UTC"),
                "fear_greed": int(r["value"])} for r in resp["data"]]
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

def fetch_fear_greed_current() -> int:
    """Retorna el valor actual del Fear & Greed Index (0-100)."""
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    resp = requests.get(url).json()
    return int(resp["data"][0]["value"])
```

**5.2** Crea la taula a la BD:
```sql
CREATE TABLE fear_greed (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL UNIQUE,
    value INTEGER NOT NULL,         -- 0 (Extreme Fear) → 100 (Extreme Greed)
    classification VARCHAR(50)      -- "Extreme Fear", "Fear", "Neutral", etc.
);
```

**5.3** Afegeix el model SQLAlchemy a `core/db/models.py`:
```python
class FearGreedDB(Base):
    __tablename__ = "fear_greed"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, unique=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(50))
```

**5.4** Incorpora com a feature al `DatasetBuilder` o `ObservationBuilder` via JOIN amb la taula de candles per `timestamp`.

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

*Última actualització: Març 2026 · Versió 1.1*
