# Roadmap — Tasques Pendents i Visió de Futur

> **Regla d'or:** cap bot va a demo fins que en backtest out-of-sample superi HoldBot en Sharpe i Calmar.
> Per a l'arquitectura del sistema: veure **[PROJECT.md](./PROJECT.md)**.

---

## Estat actual dels components

| Component | Estat | Notes |
|-----------|-------|-------|
| PaperExchange | ✅ Operatiu | Fees 0.1% + slippage 0.01% |
| OHLCVFetcher (Binance/ccxt) | ✅ Operatiu | 1h, 4h, 12h, 1d des de 2019 |
| ObservationBuilder | ✅ Operatiu | Cache en memòria per tick |
| TrendBot | ✅ Operatiu | EMA crossover + RSI |
| DCABot | ✅ Operatiu | Compra periòdica fixa |
| GridBot | ✅ Operatiu | Bollinger Bands |
| HoldBot | ✅ Operatiu | Benchmark buy & hold |
| MeanReversionBot | ✅ Operatiu | Z-score + RSI extrem + filtre volum |
| MomentumBot | ✅ Operatiu | ROC + volum confirmat + MACD |
| MLBot (RF, XGB, LGBM, CB, GRU, PatchTST) | ✅ Operatiu | 6 backends |
| RLBot (PPO, SAC) | ✅ Operatiu | Discret + continu |
| RLBot on-chain (PPO, SAC) | ⚠️ Pendent entrenament | Configs creades, requereix BD externa + `train_rl.py --agents ppo_onchain sac_onchain` |
| RLBot Professional (PPO, SAC) | ⚠️ Pendent entrenament complet | Política professional implementada (12H, Discrete-5/Continuous, ATR stop, position state, reward professional). Smoke OK. Training complet + Optuna pendent. |
| BacktestEngine + MLflow | ✅ Operatiu | Registre automàtic |
| BotComparator | ✅ Operatiu | Ranking per Sharpe |
| DemoRunner (multi-bot) | ✅ Operatiu | Persistència + Telegram |
| TelegramNotifier | ✅ Operatiu | Trades, status horari, drawdown |
| Tests (smoke + unit) | ✅ 123 tests | Cobertura bàsica sense BD |
| Optimize workflow | ✅ Operatiu | `best_params` dins YAML base (1 sol fitxer per model) |
| Config YAML unificat | ✅ Operatiu | Un YAML per model (classic/ML/RL), search space integrat |
| Walk-forward split | ✅ Operatiu | TRAIN_UNTIL / TEST_FROM al settings |
| Documentació | ✅ Completa | PROJECT, MODELS, DB, EXTENDING, CONFIG (v2.0 YAML unificat) |
| Mètriques backtesting | ✅ Correctes | Sharpe/Calmar/WinRate correctament implementats |
| Dashboard (Streamlit) | ⚠️ Bàsic | Només preus de la BD |
| Tests d'integració | ⚠️ Esquelet | 5 tests, necessiten BD real |
| Dades externes (Fear&Greed + on-chain) | ✅ Operatiu | fear_greed, funding_rates, open_interest, blockchain_metrics — integrat a FeatureBuilder/DatasetBuilder/ObservationBuilder |
| EnsembleBot | ❌ Pendent | Meta-capa de combinació |
| Feature Store | ❌ Pendent | Placeholder reservat |
| Risk Manager | ❌ Pendent | Placeholder reservat |

---

## Tasques pendents

---

### ✅ B — Dades externes (completament implementat)

Fonts implementades i operatives. Scripts de descàrrega inicial + update incremental per a cron.

| Font | Taula BD | Cobertura | Script cron |
|------|----------|-----------|-------------|
| Fear & Greed Index | `fear_greed` | Des de feb. 2018, diari | `update_fear_greed.py` (diari) |
| Funding rates | `funding_rates` | Des de set. 2019, cada 8h | `update_futures.py` (horari) |
| Open Interest (Vision S3) | `open_interest` (5m) | Des de des. 2021, cada 5min | `update_binance_vision.py` (diari) |
| Open Interest (REST) | `open_interest` (1h) | Darrers 30 dies, cada 1h | `update_futures.py` (horari) |
| Hash rate | `blockchain_metrics` | Des de ~2009, diari | `update_blockchain.py` (diari) |
| Adreces actives | `blockchain_metrics` | Des de ~2009, diari | `update_blockchain.py` (diari) |
| Transaction fees | `blockchain_metrics` | Des de ~2009, diari | `update_blockchain.py` (diari) |

**Comprovació de completesa:** `python scripts/check_data_completeness.py`

✅ **Integració al pipeline:** `FeatureBuilder`, `DatasetBuilder` i `ObservationBuilder` ja suporten totes les fonts externes via config YAML. No cal tocar codi — s'activen amb `features.external` al YAML del model. Config de referència: `config/models/ppo_onchain.yaml`.

---

#### B_pendent. Anàlisi de notícies i sentiment (NLP)
La font més complexa però potencialment la més poderosa per anticipar moviments.

**Opcions per obtenir el senyal:**
- **Claude API:** enviar titulars/articles i demanar un score de sentiment BTC (-1.0 a 1.0) + classificació de rellevància. Flexible i d'alta qualitat, però té cost per token.
- **BERT/FinBERT fine-tuned:** model local entrenat en notícies financeres. Gratuït en inferència, però cal mantenir-lo.
- **CryptoPanic API:** agrega notícies de cripto amb votes (bullish/bearish) de la comunitat. Té pla gratuït.

**Fonts de notícies candidates:** CryptoPanic, CoinDesk RSS, Reuters cripto, tweets de comptes influents (X API).

**Historial per entrenament:** CryptoPanic té historial via API. Per a sentiment retroactiu amb Claude/BERT, caldria processar un arxiu de titulars antics.

**Estructura proposada:**
```
data/sources/
├── fear_greed.py        # B1
├── onchain.py           # B2
└── news_sentiment.py    # B3 (fetch + score via Claude API o FinBERT)
```

**Taula BD genèrica per a tot extern:**
```sql
CREATE TABLE external_signals (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),        -- 'fear_greed', 'sopr', 'news_sentiment', etc.
    timestamp TIMESTAMPTZ,
    value FLOAT,               -- el valor numèric principal
    metadata JSON              -- dades extra (classificació, text, etc.)
);
```

---

### 🟡 C — Nous bots

Cada bot nou segueix el cicle complet: recerca → implementació → configuració → optimització Optuna → backtest → (si supera HoldBot) → demo.

#### ✅ C1. Bots clàssics (implementats)

MeanReversionBot i MomentumBot implementats seguint estat de l'art per BTC.
**Pas següent:** executar backtest, optimitzar amb Optuna i, si superen HoldBot, activar al demo.

```bash
# Optimitzar:
python scripts/optimize_bots.py --bots mean_reversion momentum

# Comparar amb la resta de bots:
python scripts/run_comparison.py --bots hold mean_reversion momentum trend grid
```

| Bot | Lògica | Estat |
|-----|--------|-------|
| MeanReversionBot | Z-score del preu + RSI extrem + filtre de volum | ✅ Implementat — pendent backtest + Optuna |
| MomentumBot | Rate of Change (ROC) + volum confirmat + MACD | ✅ Implementat — pendent backtest + Optuna |
| BreakoutBot | Suports/resistències (pivot points) + ATR per confirmar | ❌ Pendent |

---

#### C2. Nous models ML

Cada model nou necessita: implementació `BaseMLModel`, entrada al `_MODEL_REGISTRY`, config d'entrenament + Optuna, backtest complet.

| Model | Tipus | Notes |
|-------|-------|-------|
| Temporal Fusion Transformer (TFT) | Deep Learning | Estat de l'art per séries temporals financeres amb covarites externes; ideal per incorporar dades de B1/B2/B3 |
| N-BEATS / N-HiTS | Deep Learning | Arquitectures pures sense RNN; molt eficients |
| TabNet | Tabular | Competitiu amb XGBoost, interpretable via atenció |

---

#### ✅ C_onchain. PPO + SAC amb features on-chain (implementat, pendent entrenament)

Variants on-chain dels agents RL baseline. Mateixa política i reward que els baseline, però amb 5 features addicionals: Fear & Greed, funding rate, Open Interest (1h) i hash rate.

| Agent | Config | obs_shape | Estat |
|-------|--------|-----------|-------|
| PPO on-chain | `config/models/ppo_onchain.yaml` | 14×96=1344 | ✅ Config validada — pendent entrenament |
| SAC on-chain | `config/models/sac_onchain.yaml` | 14×96=1344 | ✅ Config creada — pendent entrenament |

```bash
# Entrenar (30-90 min per agent):
python scripts/train_rl.py --agents ppo_onchain sac_onchain

# Optimitzar primer (opcional, ~3h):
python scripts/optimize_models.py --agents ppo_onchain sac_onchain

# Comparar amb baseline:
python scripts/run_comparison.py --bots hold ppo sac ppo_onchain sac_onchain
```

**Pre-requisit:** dades externes a la BD (`python scripts/download_fear_greed.py`, `download_blockchain.py`, `update_futures.py`).

---

#### ✅ C3. Investigació RL professional + nova política (implementat, pendent entrenament complet)

**Investigació realitzada:** `docs/decisions/trading_policy_reference.md` — recerca profunda sobre gestió de risc professional, regime detection, position sizing, stop-loss i reward shaping per a BTC swing trading sense apalancament. Document de disseny final: `docs/nova_politica/DISSENY.md` (v2.0).

**Política implementada:** `ppo_professional` + `sac_professional` (timeframe 12H, cicles de 5–7 dies).

| Element del pla | Implementació |
|-----------------|--------------|
| Reward shaping (drawdown + overtrading) | ✅ `professional_risk_adjusted` — quadràtic progressiu + penalització per trades prematurs |
| Position sizing [0.0–1.0] | ✅ SAC continuous Box([0,1]) + deadband ±0.05; PPO Discrete-5 (HOLD/BUY_FULL/BUY_PARTIAL/SELL_PARTIAL/SELL_FULL) |
| Regime detection | ✅ ADX-14 (força tendència) + vol_ratio = ATR-5/ATR-14 (expansió/compressió) en observació |
| Portfolio state en observació | ✅ 4 dims afegides: pnl_pct, position_fraction, steps_in_position, drawdown_pct |
| Stop-loss | ✅ Stop ATR dur forçat per l'entorn (agent no pot moure'l); `stop_atr_multiplier=2.0` |
| Multi-timeframe obs | 🔄 Substituït per 12H × 90 lookback (45 dies) — menys soroll, millor per a swing |

**Fitxers nous:** `bots/rl/environment_professional.py`, `bots/rl/rewards/professional.py`, `config/models/ppo_professional.yaml`, `config/models/sac_professional.yaml`.

**Estat:** Smoke OK (-8.7% SAC, -39.7% PPO amb 50k steps aleatoris). Training complet (500k steps) + Optuna pendent.

```bash
# Optimitzar (en curs):
python scripts/optimize_models.py --agents ppo_professional sac_professional

# Entrenar amb millors hiperparàmetres:
python scripts/train_rl.py --agents ppo_professional sac_professional

# Comparar:
python scripts/run_comparison.py --bots hold ppo sac ppo_professional sac_professional
```

**Extensions futures de C3 (pendent):**

| Agent | Notes |
|-------|-------|
| TD3 (Twin Delayed DDPG) | Millor que SAC en entorns molt sorollosos; estable |
| Curriculum learning (PPO/SAC) | Entrenar en fases: tendències clares → rangs → alta volatilitat |
| DreamerV3 | Model-based RL; aprèn un model intern del mercat i simula internament — molt eficient en mostres |

---

### 🟡 D — EnsembleBot

La capa de meta-decisió: combinar els senyals de tots els bots per produir un sol senyal més robust.

**Més allà del propi bot, caldrà:**
- **Historial de prediccions:** guardar els senyals de cada bot a la BD per poder calcular pesos dinàmics
- **Mètriques en finestra lliscant:** calcular el Sharpe dels últims N dies per bot per ponderar el vote
- **Backtest de l'ensemble:** sistema per backtesta combinacions de bots (no un bot sol)
- **Selecció de components:** assegurar diversitat baixa correlació entre els bots seleccionats

**Polítiques a implementar (per ordre de complexitat):**

| Política | Descripció | Quan usar |
|---------|-----------|-----------|
| `majority_vote` | Si >50% de bots diuen BUY → BUY | Punt de partida |
| `weighted` | Pes proporcional al Sharpe dels últims N dies | Quan alguns bots son clarament millors |
| `unanimous` | Només actua si TOTS coincideixen | Molt conservador, poques operacions |
| `regime_routing` | Selecciona el millor bot per règim de mercat | Quan tenim detecció de règim |
| `stacking` | Model ML de 2a capa entrena sobre les prediccions | Quan tenim prou historial de predictions |

---

### 🟢 E — Millores del sistema

#### E1. Telegram millorat per al demo
Quan hi hagi mesos de dades de demo reals, millorar les notificacions:

- **Resum diari:** PnL de cada bot, millor/pitjor del dia, comparativa vs. BTC spot
- **Alertes de drawdown:** configurables per bot (ex: alerta si drawdown > 10%)
- **Rànquing setmanal:** quin bot va millor la setmana
- **Comandes interactives:** `/status`, `/portfolio`, `/trades`, `/ranking` via Telegram bot
- **Gràfics:** enviar una imatge amb el portfolio de cada bot (matplotlib inline)

---

#### E2. Correccions del DemoRunner

- **Sincronització de candles:** el bot ha d'actuar quan tanca una candle (1 cop/hora), no cada 60s sobre la mateixa candle oberta
- **Persistir estat intern:** `_in_position` i `_tick_count` han de persistir a la BD i restaurar-se en reiniciar

---

#### E3. Dashboard complet
Quan hi hagi mesos de dades de demo reals:

- Portfolios en temps real per bot (gràfic PnL acumulat)
- Drawdowns visuals i alertes configurables
- Trade log amb filtre per bot i data
- Comparativa vs. BTC spot (benchmark)
- Matriu de correlació de retorns entre bots
- Ranking per Sharpe + Calmar dels últims 30/90/180 dies

---

### 🔵 F — Desplegament i demo llarg

#### F1. Migrar a servidor
Explorar opcions (queda lluny, però cal tenir-ho en ment):

| Opció | Cost | Notes |
|-------|------|-------|
| Oracle Cloud Free Tier | Gratuït | 4 vCPU, 24GB RAM; suficient per ara |
| Hetzner Cloud CX22 | ~4€/mes | Millor latència a Europa |
| Raspberry Pi 4/5 | Hardware únic | Opció local, sense cost mensual |

**Infraestructura mínima per al desplegament:**
- Cron per `update_data.py` + `update_fear_greed.py` cada hora
- Systemd (o Docker) per restart automàtic del DemoRunner
- Backup automàtic de la BD (pg_dump diari)
- Watchdog extern amb alerta Telegram per caigudes

---

#### F2. Demo 24/7 durant mesos
L'objectiu final d'aquesta fase del projecte:

- Mínimament **3-6 mesos** de paper trading en temps real
- Tots els bots que hagin superat el criteri de backtest (Sharpe > 1.5, Drawdown < -20%, supera HoldBot)
- L'EnsembleBot actiu des del primer dia del demo llarg
- Registre complet a la BD de tots els ticks i trades
- Anàlisi mensual: ajustar pesos de l'ensemble si cal

---

## Seqüència recomanada

```
[B ✅] Fear & Greed + on-chain (dades, scripts, FeatureBuilder/DatasetBuilder/ObservationBuilder)
              ↓
[C1 ✅] MeanReversionBot + MomentumBot ← implementats, pendent backtest + Optuna
              ↓
[C_onchain ✅] PPO/SAC on-chain ← configs creades i validades, PENDENT ENTRENAMENT
              ↓
[C3 ✅] RL professional → ppo_professional + sac_professional (12H, ATR stop, position state, reward professional) — pendent entrenament complet + Optuna
              ↓
[B_pendent/NLP] NLP sentiment ← quan els models ML ja consumeixin les noves features
              ↓
[C2] TFT / N-BEATS ← quan les dades externes estiguin llestes
              ↓
[D] EnsembleBot (majority vote → weighted → stacking)
              ↓
[E1] Telegram millorat
              ↓
[F1] Migrar a servidor
              ↓
[F2] Demo 24/7 — 3-6 mesos
```

---

*Última actualització: Març 2026 · Versió 1.5*

---

## 🔧 COSES A RETOCAR ABANS DE FER RES MES

> **Auditoria realitzada:** Març 2026 · Estat del projecte: 148 fitxers, ~15.000 LOC, 9 bots, 37 YAML configs.
> **Objectiu:** Consolidar la base tècnica abans de seguir afegint models i funcionalitats.
> 
> Les correccions petites (**✅ Aplicades**) ja estan fetes.
> Les propostes grans requereixen la teva validació (**⏳ Pendent aprovació**).

---

### PART 1 — COSES A ELIMINAR

#### 1.1 Bots poco óptims: **val la pena guardar-los tots** ⏳ Pendent decisió

**La pregunta:** té sentit tenir `HoldBot`, `DCABot`, `GridBot` si els RL professionals els superaran?

**La resposta recomanada: SÍ, guardar-los tots.** Motiu:
- `HoldBot` és el benchmark indispensable (la "regla d'or" del roadmap el menciona explícitament).
- `DCABot` representa una estratègia que molts inversors reals fan servir — útil per contextualitzar els resultats.
- Els clàssics actuen com a **floor floor de comparació**: si un RL no supera TrendBot o MeanReversionBot, el model no és bo.
- El cost de mantenir-los és mínim (cap BD, cap entrenament, cap reentrenament).

**Recomanació:** Mantenir tots. No eliminar cap bot. Eliminar un bot només si falles de forma consistent en 6+ mesos de demo real.

---

#### 1.2 Configs duplicades (base + optimitzat): **50% redundància de YAML** ✅ Fet (Març 2026)

**Solució implementada — Proposta A:** `best_params` dins el YAML base (in-place).

Quan Optuna troba els millors paràmetres, `save_best_config()` escriu una secció `best_params` directament al YAML base en lloc de crear un `*_optimized.yaml` separat:
```yaml
# config/models/ppo.yaml — UN sol fitxer, conté base + millors params
training:
  model:
    learning_rate: 0.0003        # valor base
    batch_size: 64
    ...

# Secció afegida per Optuna (generada automàticament, no editar)
best_params:
  learning_rate: 0.00087
  batch_size: 256
```
`apply_best_params()` (`core/config_utils.py`) aplica els overrides en temps d'entrenament/inferència. Routing automàtic: paràmetres d'entorn (`lookback`, `reward_type`, etc.) → `training.environment`; la resta → `training.model`; bots clàssics → top-level flat.

**Fitxers modificats:** `core/config_utils.py` (nou), `core/backtesting/{rl,ml,}_optimizer.py`, `bots/rl/trainer.py`, `bots/rl/rl_bot.py`, `core/interfaces/base_bot.py`, `core/engine/demo_runner.py`, `scripts/{train_rl,train_models,run_comparison,optimize_models,optimize_bots}.py`.

**Impacte real:** 0 fitxers `_optimized.yaml` existents eliminats (cap havia estat generat encara). Reducció potencial de 42 → 21 fitxers quan es faci una optimització massiva.

---

#### 1.3 Tests d'integració esquelet: **netejar o completar** ⏳ Pendent decisió

Els 5 tests a `tests/integration/test_full_backtests.py` estan tots marcats amb `pytest.skip("requires real PostgreSQL")`. Ocupen codi sense donar valor.

**Opcions:**
- **Opció A:** Eliminar el fitxer fins que es tingui la infra per a tests reals.
- **Opció B:** Convertir-los en tests amb SQLite mock (no necessiten BD real).
- **Opció C:** Mantenir com a "esquelet documentat" per al futur.

**Recomanació:** Opció B — implementar amb SQLite in-memory. És 2-3 hores de feina i dona cobertura real als fluxos crítics.

---

### PART 2 — COSES A OPTIMITZAR

#### 2.1 Duplicació: `ProgressCallback` als 3 agents RL ✅ **Aplicat**

Era la mateixa classe copiada 3 vegades en `ppo_agent.py`, `sac_agent.py` i `td3_agent.py`.

**Solució aplicada:** Extreta a `bots/rl/callbacks.py`. Els 3 agents ara fan:
```python
from bots.rl.callbacks import ProgressCallback
```
**Estalvi:** 45 línies eliminades. Ara qualsevol nou agent importa directament sense copiar res.

---

#### 2.2 Duplicació: `GridBot` hardcodejava BB window i std ✅ **Aplicat**

`grid_bot.py` tenia `rolling(window=20)` i `2 * std` hardcodejats. Ara llegeix del YAML:
```python
bb_window = self.config.get("bb_window", 20)
bb_std    = self.config.get("bb_std", 2.0)
```
El `config/models/grid.yaml` ja inclou `bb_window: 20`, `bb_std: 2.0` i els dos al search_space d'Optuna.

---

#### 2.3 Duplicació major: 4 models ML tree-based gairebé idèntics ⏳ Pendent aprovació

`xgboost_model.py`, `lightgbm_model.py`, `catboost_model.py` i `random_forest_model.py` comparteixen:
- Estructura `train()` idèntica (TimeSeriesSplit CV 5-fold, MLflow logging, tqdm progress)
- Estructura `save()`/`load()` idèntica (pickle amb `{model, scaler, feature_names}`)
- Estructura `predict()` gairebé idèntica

**Proposta:** Crear `bots/ml/base_tree_model.py` amb el patró compartit:
```python
class BaseTreeModel(BaseMLModel):
    """Shared training loop for tree-based classifiers (XGB, LGB, CB, RF)."""
    
    @abstractmethod
    def _build_model(self, params: dict): ...        # cada fill construeix el seu model
    
    def train(self, X, y) -> dict: ...               # TimeSeriesSplit CV idèntic
    def predict(self, X, threshold=0.35): ...        # idèntic
    def save(self, path): ...                        # idèntic
    def load(self, path): ...                        # idèntic
```
Cada model fill reduiria a ~30 LOC (ara ~130 LOC cadascun).

**Impacte:** ~300 LOC eliminades, manteniment molt més fàcil. Si cal canviar el CV o el logging, es fa en un sol lloc.
**Risc:** Baix. La lògica de cada model és independent; només el scaffolding és compartit.

---

#### 2.4 Duplicació: `_validate()` als 3 agents RL ⏳ Pendent aprovació

El mètode `_validate(env)` és gairebé idèntic als 3 agents (PPO/SAC/TD3). Calcula retorn, drawdown màxim i nombre de trades.

**Proposta A (simple):** Moure la lògica a `BaseRLAgent._validate(env, agent_name)` — el mètode ja existeix a la interfície però no implementa la lògica.

**Proposta B (més explícita):** Funció utilitària a `core/backtesting/agent_validator.py`:
```python
def validate_agent(agent, env) -> dict:
    """Runs validation episode, returns standard metrics dict."""
    ...
```

**Impacte:** ~80 LOC eliminades. Tots els agents futurs heretarien la validació automàticament.

---

#### 2.5 Duplicació: `TimeSeriesDataset` en GRU i PatchTST ⏳ Pendent aprovació

La mateixa classe PyTorch Dataset està copiada en `gru_model.py` i `patchtst_model.py`.

**Proposta:** Extreure a `data/processing/torch_dataset.py`:
```python
class TimeSeriesDataset(Dataset):
    """Shared PyTorch Dataset for time-series window sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray): ...
    def __len__(self): ...
    def __getitem__(self, idx): ...
```

**Impacte:** ~25 LOC eliminades. Qualsevol model DL futur (TFT, N-BEATS) reutilitza directament.

---

#### 2.6 Hardcode `"12h"` al `download_data.py` ⏳ Pendent decisió

El timeframe `"12h"` es va afegir manualment durant C2. Idealment, la llista de timeframes a descarregar s'hauria de derivar automàticament de les configs YAML actives.

**Proposta:** Afegir una funció a `scripts/download_data.py`:
```python
def _collect_required_timeframes() -> set[str]:
    """Reads all config/models/*.yaml and collects all timeframes mentioned."""
    import glob, yaml
    tfs = set()
    for f in glob.glob("config/models/*.yaml"):
        cfg = yaml.safe_load(open(f))
        if tf := cfg.get("timeframe"):
            tfs.add(tf)
        tfs.update(cfg.get("aux_timeframes", []))
    return tfs
```
Així, quan s'afegeixi un nou agent amb un nou timeframe, el download l'agafa automàticament.

---

#### 2.7 Magic numbers en constants compartides ⏳ Pendent decisió

Hi ha valors numèrics repetits entre environments i rewards que haurien de viure en constants:

```python
# Ara estan duplicats a environment.py, environment_professional.py, rewards/
_ATR_REFERENCE     = 0.02    # referència 2% volatilitat BTC 12H
_CLIP_OBS_BOUNDS   = 10.0    # clipping de l'observació normalitzada
_NUMERICAL_EPSILON = 1e-8    # evitar divisió per zero
_DEADBAND          = 0.05    # fracció mínima de canvi per SAC/TD3
```

**Proposta:** Crear `bots/rl/constants.py` amb totes les constants RL compartides.
**Risc:** Molt baix. Les constants actuals funcionen bé; és una millora de mantenibilitat.

---

### PART 3 — ESCALABILITAT

#### 3.1 Diagnòstic: el core s'ha de tocar ZERO per a nous agents RL ✅ **Ja funciona**

El patró registry (`_AGENT_REGISTRY`, `_ENV_REGISTRY`, `@register("name")`) funciona. Afegir `TD3` ha requerit tocar exactament 7 fitxers (cap al core), tots de forma addictiva. La guia `C3_RL_ADVANCED.md § 9` documenta el procés amb checklist.

**Estat actual:** Excellent. Cap canvi al core per a nous agents RL.

---

#### 3.2 Diagnòstic: afegir nous ML models requereix 5 fitxers ⚠️ Acceptable però millorable

Afegir un model ML nou (e.g. TFT) requereix:
1. `bots/ml/tft_model.py` (nou)
2. `bots/ml/ml_bot.py` → `_MODEL_REGISTRY` (modificar)
3. `scripts/train_models.py` → llista de models (modificar)
4. `config/models/tft.yaml` (nou)
5. `scripts/run_comparison.py` → `BOT_REGISTRY` (modificar)

**Millora proposta:** Fer que `_MODEL_REGISTRY` i `BOT_REGISTRY` s'auto-poblïn llegint les configs YAML (`category: ML`) en lloc d'estar hardcodejats. Així afegir un model seria 0 canvis de codi.

```python
# Proposta: auto-discovery de models via YAML
def _discover_bots(category: str = None) -> dict[str, str]:
    """Returns {model_type: yaml_path} for all models matching category."""
    import glob, yaml
    registry = {}
    for path in sorted(glob.glob("config/models/*.yaml")):
        cfg = yaml.safe_load(open(path))
        if category is None or cfg.get("category") == category:
            registry[cfg["model_type"]] = path
    return registry
```

**Impacte:** Afegir un model nou = crear YAML + model Python. Zero tocar scripts.
**Risc:** Mitjà — cal assegurar-se que no s'agafin configs _optimized automàticament.

---

#### 3.3 Diagnòstic: afegir nous bots clàssics requereix 4 fitxers ⚠️ Acceptable

Segueix el mateix patró que ML. La mateixa solució d'auto-discovery aplicaria.

---

#### 3.4 Falta README.md arrel ⏳ Pendent

No hi ha cap `README.md` a l'arrel del projecte. Qualsevol persona nova (o tu d'aquí a 6 mesos) no sap per on començar.

**Contingut mínim recomanat:**
- Una frase de QUÈ és el projecte
- Prerequisits (Python, PostgreSQL, ccxt)
- Quick start (3-4 comandos)
- Estructura de directoris resumida
- Llista de scripts principals amb descripció d'1 línia
- Referència als docs detallats (PROJECT.md, EXTENDING.md, MODELS.md)

---

### PART 4 — DADES

#### 4.1 Diagnòstic: la BD és sòlida ✅

Totes les dades crítiques estan a PostgreSQL:

| Taula | Dades | Cobertura | Backup |
|-------|-------|-----------|--------|
| `candles` | OHLCV 1H/4H/12H/1D BTC/USDT | Des de gen. 2019 | ⚠️ Cal configurar pg_dump |
| `fear_greed` | Índex 0-100 diari | Des de feb. 2018 | ⚠️ Cal configurar pg_dump |
| `funding_rates` | Binance perpetual 8H | Des de set. 2019 | ⚠️ Cal configurar pg_dump |
| `open_interest` | Binance Vision 5m + REST 1H | 5m des de des. 2021 | ⚠️ Cal configurar pg_dump |
| `blockchain_metrics` | Hash rate, adreces, fees | Des de 2009 | ⚠️ Cal configurar pg_dump |

**Risc crític detectat:** **No hi ha backup automàtic configurat**. Si el disc falla o el PostgreSQL es corromp, es perd tot l'historial. Per a dades que han trigat hores a descarregar (OHLCV, blockchain), seria molt costós recuperar.

**Accions recomanades (prioritat ALTA):**
```bash
# 1. pg_dump diari via cron (minim)
0 3 * * * pg_dump btc_trading > /backups/btc_trading_$(date +%Y%m%d).sql

# 2. Retenir últims 30 dies
find /backups -name "*.sql" -mtime +30 -delete

# 3. Copiar a ubicació remota (S3, Google Drive, etc.)
```

---

#### 4.2 Diagnòstic: `open_interest` 1H té cobertura limitada ✅ Documentat

Ja documentat al codi i als docs: les dades REST cobreixen els **darrers 30 dies** → no es pot usar per entrenar models amb lookback llarg. Les configs professionals ja exclouen `oi_btc_1h`. Les dades 5m de Binance Vision cobreixen des de des. 2021.

**Estat:** Correctament gestionat. No requereix acció.

---

#### 4.3 Dades de sentiment NLP: **pendents de valorar** ⏳ Pendent

Veure secció `B_pendent` del ROADMAP. El Fear & Greed ja cobreix el sentiment de mercat bàsic. NLP seria el pas següent.

---

### RESUM EXECUTIU: Per ordre de prioritat

| Prioritat | Tasca | Esforç | Impacte |
|-----------|-------|--------|---------|
| 🔴 **P0** | Backup automàtic BD (pg_dump + cron) | 30 min | Evitar pèrdua de dades irreversible |
| 🟠 **P1** | `BaseTreeModel` per XGB/LGB/CB/RF | 2h | -300 LOC, manteniment fàcil |
| 🟠 **P1** | `_validate()` a `BaseRLAgent` | 1h | -80 LOC, nous agents ho hereten |
| 🟠 **P1** | `TimeSeriesDataset` a `data/processing/` | 30 min | -25 LOC, reutilitzable per TFT/N-BEATS |
| 🟡 **P2** | README.md arrel | 1h | Orientació per a tu d'aquí 6 mesos |
| 🟡 **P2** | Auto-discovery de models via YAML | 3h | Zero canvis de codi per nous models |
| 🟡 **P2** | `bots/rl/constants.py` compartit | 30 min | Consistència de constants RL |
| 🟢 **P3** | Config YAML: unificar base + optimitzat | 2h | -16 fitxers YAML |
| 🟢 **P3** | Tests integració amb SQLite mock | 3h | Cobertura real sense BD externa |
| 🟢 **P3** | Auto-discovery timeframes al download | 1h | Zero tocar download_data.py per nous TFs |

> **Ja aplicat (✅):** ProgressCallback extret a `bots/rl/callbacks.py` · GridBot BB params llegits del YAML

---

*Auditoria: Març 2026 · Properes accions recomanades: P0 (backup BD) + P1 (BaseTreeModel)*

