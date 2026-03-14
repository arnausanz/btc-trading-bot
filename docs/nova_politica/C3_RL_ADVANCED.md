# C3 — Investigació i Implementació: RL Avançat per a Swing Trading BTC

> **Versió:** 1.0 · Març 2026
> **Prerequisits:** Llegir primer `DISSENY.md` (C2 — PPO/SAC Professional)

---

## Índex

1. [Context i motivació](#1-context-i-motivació)
2. [Estat de l'art 2024–2025](#2-estat-de-lart-20242025)
3. [Per què TD3 per a mercats financers sorollosos](#3-per-què-td3-per-a-mercats-financers-sorollosos)
4. [Dades multi-timeframe: teoria i implementació](#4-dades-multi-timeframe-teoria-i-implementació)
5. [Dades de sentiment: Fear & Greed i Funding Rate](#5-dades-de-sentiment-fear--greed-i-funding-rate)
6. [Reward `regime_adaptive`: detecció de règim](#6-reward-regime_adaptive-detecció-de-règim)
7. [Models implementats](#7-models-implementats)
8. [Com entrenar i optimitzar](#8-com-entrenar-i-optimitzar)
9. [Com afegir nous agents (guia per a futurs contribuïdors)](#9-com-afegir-nous-agents)
10. [Agents futurs recomanats](#10-agents-futurs-recomanats)

---

## 1. Context i motivació

Els agents C2 (`ppo_professional`, `sac_professional`) van demostrar que:
- Timeframe 12H redueix el soroll d'alta freqüència i minimitza comissions
- Position state (+4 dimensions) permet a l'agent veure on és en el trade
- ATR stop loss mecànic evita el comportament "moure el stop" (error clàssic)
- Reward professional (ATR-scaled + drawdown penalty) genera decisions més conservadores

**Limitació de C2:** L'agent veu cada timeframe en aïllament. Un agent que opera a 12H no sap si el mercat està en tendència a escala diària, ni quin és el sentiment del mercat en el moment de la decisió.

**Objectiu de C3:**
- Introduir **TD3** com a algoritme principal (millor que SAC en entorns sorollosos)
- Integrar **dades multi-timeframe** (1H + 4H context)
- Integrar **dades de sentiment** (Fear & Greed, funding rate)
- Implementar **reward adaptatiu al règim** (trending/ranging/chaotic)

---

## 2. Estat de l'art 2024–2025

### Algoritmes de RL per a trading (en ordre de maduresa)

| Algoritme | Tipus | Pros | Contres | Ús recomanat |
|-----------|-------|------|---------|--------------|
| **PPO** | On-policy | Estable, interpretatble | Menys eficient en mostra | Prototipat ràpid |
| **SAC** | Off-policy | Bona exploració (entropy) | Sensible a lr; overestima Q | Continu, exploració crítica |
| **TD3** | Off-policy | Twin critics estables, menys soroll | Sense exploració automàtica | **Continu, soroll elevat** |
| DreamerV3 | Model-based | Molt eficient en mostra | Molt complex | Pocs timesteps disponibles |
| Curriculum RL | Meta-estratègia | Aprenentatge progressiu | Requereix múltiples envs | Agents no convergents |

**Conclusió 2024:** Per a trading de BTC amb recompenses sorolloses i no estacionàries, **TD3 és l'algoritme off-policy més estable**. SAC és preferible quan la diversitat d'exploració és crítica; TD3 quan la variance del gradient és el problema principal.

### Per què NO DreamerV3 ara

DreamerV3 (Hafner et al., 2023) és teòricament superior però requereix:
- Implementació pròpia (no disponible a SB3)
- Models latents de 200M+ paràmetres
- Inferència molt més lenta que PPO/SAC/TD3
- Necessitat de dades d'imatge o seqüències molt riques

Per a un dataset de candles 1D–12H, TD3 és la millor elecció pragmàtica.

---

## 3. Per què TD3 per a mercats financers sorollosos

### El problema de l'overestimació en Q-learning

SAC i DDPG utilitzen un sol crític (xarxa Q). En entorns amb recompenses sorolloses (com BTC on hi ha jumps de preu inesperats), el crític tendeix a **sobreestimar** els Q-values màxims, cosa que porta a polítiques massa agressives.

### Twin Q-networks (la solució de TD3)

TD3 manté **dos crítics independents** (Q1 i Q2) i utilitza el **mínim** dels dos per a l'estimació del target:

```
TD3 Bellman target = r + γ · min(Q1(s', a'), Q2(s', a'))
```

Això elimina el biaix d'overestimació: si un crític sobreestima, l'altre actua com a fre. En pràctica, genera polítiques més conservadores i estables en mercats volàtils.

### Delayed policy updates

L'actor (política) s'actualitza cada `policy_delay=2` steps de crític. Aixó:
- Permet al crític convergir una mica abans de canviar la política
- Redueix la variança dels gradients de política
- Especialment útil en sèries temporals financeres on la stationarity és baixa

### Target policy smoothing

El target de Bellman s'avalua amb soroll afegit a les accions target:

```
ã = clip(π(s') + clip(N(0, σ), -c, c), a_min, a_max)
TD3 target = r + γ · min(Q1(s', ã), Q2(s', ã))
```

Això **suavitza el paisatge de valor** i evita que l'agent exploti pics d'alta recompensa que podrien ser artefactes del soroll del mercat.

### Comparació directa SAC vs TD3

```
                    SAC                     TD3
───────────────────────────────────────────────────────
Critics             1 (doble estoc.)        2 (determinístic)
Exploració          Entropia automàtica     Soroll gaussià explícit
Overestimació       Possible                Molt reduïda
Millor per          Exploració crítica       Entorns sorollosos
Convergència        Lenta però estable      Ràpida en ambients estables
BTC swing (12H)     Bona                    Millor (menys spikes)
```

---

## 4. Dades multi-timeframe: teoria i implementació

### Per què múltiples timeframes

Un operador professional llegeix el mercat en diverses escales simultàniament:
- **1H (entrada):** Patrons de breakout, MACD crossovers, RSI oversold/overbought
- **4H (tendència):** Direcció de la tendència intermèdia, suports/resistències clau
- **1D (estructura):** Tendència macro, nivells de Fibonacci majors

Un agent que opera a 1H sense veure el 4H pot comprar just quan la tendència 4H és bajista — un error que un trader humà evitaria immediatament.

### Implementació: `MultiFrameFeatureBuilder`

```
data/processing/multiframe_builder.py
```

**Funcionament:**

```
1H OHLCV  ──► compute_features() ──► df_1h (close, rsi_14, macd, ...)
4H OHLCV  ──► compute_features() ──► df_4h (rsi_14, ema_20, adx_14, ...)
                                          │
                                   rename_suffix(_4h)
                                          │
                                   (rsi_14_4h, ema_20_4h, adx_14_4h, ...)
                                          │
df_1h ◄── merge_asof(backward) ◄── df_4h_renamed
                                          │
                               + external (fear_greed, funding_rate)
                                          │
                                     df_merged
                                          │
                             features.select filter + dropna()
                                          │
                              DataFrame llest per l'entorn
```

**merge_asof backward:** Cada fila 1H hereta el darrer valor disponible del 4H. Això és equivalent a forward-fill: `14:00 1H` hereta el valor del `12:00 4H` (el darrer candle 4H disponible). No hi ha data leakage perquè el valor 4H `t` és available des que comença el candle, i només s'aplica a les hores posteriors.

### Configuració YAML

```yaml
model_type: td3_multiframe
timeframe: 1h              # base (environment steps)
aux_timeframes: [4h]       # auxiliary (merged via MultiFrameFeatureBuilder)

features:
  lookback: 60
  select:
    # 1H features
    - close
    - rsi_14
    ...
    # 4H features (suffix _4h automàtic)
    - rsi_14_4h
    - ema_20_4h
    - adx_14_4h
```

### Tamaño de l'observació

```
obs_shape = (n_features_1h + n_features_4h) × lookback_1h + 4 (position state)
          = (14 + 11) × 60 + 4
          = 1.504
```

---

## 5. Dades de sentiment: Fear & Greed i Funding Rate

### Fear & Greed Index

**Font:** alternative.me (disponible des de 2018)
**Granularitat:** Diari
**Rang:** 0 (Extreme Fear) → 100 (Extreme Greed)

**Per què és útil:**
- Extremes (< 20, > 80) sovint marquen punts de gir importants
- BTC tendeix a pujar quan passa de Fear → Greed, i a baixar en l'inrevés
- Correlació negativa amb drawdowns severes: puntuacions < 25 precedeixen molts mínims locals

**Com s'integra:**
- Es guarda a la taula `fear_greed` de la BD (actualitzada diàriament)
- `FeatureBuilder` la carrega amb `merge_asof(backward)` → forward-fill per hores
- Feature resultant: `fear_greed_value` (float 0–100)

### Funding Rate

**Font:** Binance perpetual BTC/USDT:USDT (cada 8H, des de 2019)
**Rang:** Típicament -0.1% a +0.3% per cada 8H

**Per què és útil:**
- **Funding positiu elevat** (> 0.1%): longs paguen shorts → mercat sobrecomprat, possible retracement
- **Funding negatiu** (< -0.05%): shorts paguen longs → mercat sobrvenut, possible rebote
- És un proxy directe del sentiment del mercat de derivats (on hi ha molt volum)

**Com s'integra:**
- Taula `funding_rates` a la BD (actualitzada per cron horari)
- Feature resultant: `funding_rate` (float, valors típics ±0.001)

### Per què sentiment + TD3 > sentiment + SAC

SAC afegeix entropia per explorar, cosa que pot diluir l'efecte de senyals de sentiment forts. TD3 és més determinista i, un cop aprèn la correlació funding_rate → direcció, l'aplica de forma consistent.

---

## 6. Reward `regime_adaptive`: detecció de règim

### El problema del reward únic

El reward `professional` és excel·lent per al cas general, però no distingeix entre:
- Mercat en **tendència clara** (ADX > 25): hauries de seguir la tendència i aguantar
- Mercat en **rang** (ADX < 20, vol_ratio < 0.8): hauries d'esperar breakouts
- Mercat **caòtic** (vol_ratio > 1.5): millor estar flat, no entrar

Amb un reward uniforme, l'agent aprèn un comportament promig que no és òptim en cap règim.

### Detecció de règim

```python
# Paràmetres de detecció
ADX_TREND_MIN = 25    # ADX sobre aquest llindar = tendència
ADX_RANGE_MAX = 20    # ADX sota aquest = rang
VOL_TREND_MAX = 1.5   # vol_ratio (ATR_5/ATR_14) màxim per tendència
VOL_RANGE_MAX = 0.8   # vol_ratio màxim per rang pur
VOL_CHAOS_MIN = 1.5   # vol_ratio sobre aquest = caos

is_trending = (ADX >= 25) AND (0.8 <= vol_ratio <= 1.5)
is_ranging  = (ADX <= 20) AND (vol_ratio < 0.8)
is_chaotic  = (vol_ratio > 1.5)
```

### Ajusts de reward per règim

| Règim | Base PnL | Penalització chaos | Bonus paciència | Bonus aguantar |
|-------|----------|--------------------|-----------------|----------------|
| Trending | ×1.5 | — | — | Petit (+0.002×scaling) |
| Ranging | ×0.7 | — | Gran (+0.0015×scaling) | — |
| Chaotic | ×1.0 | Fort (×0.4×(vol-1.5)) | Petit (+0.001×scaling si flat) | — |
| Neutral | ×1.0 | Moderat | — | — |

**Lògica:** En tendència, cada % de retorn val 1.5× més, perquè és consistent i defensable. En rang, el base és reduït per evitar over-trading amb oscil·lacions petites. En caos, s'amplifica el penalty d'entrar.

### Nota sobre `adx` en `regime_adaptive`

El reward `regime_adaptive` accepta un paràmetre extra `adx` (default 20.0). Quan s'usa amb `BtcTradingEnvProfessionalContinuous`, el paràmetre `adx` no s'injecta automàticament com `atr_pct` i `vol_ratio`.

Per fer-ho completament funcional, `_compute_reward()` hauria de llegir `adx_14` del row del DataFrame i passar-lo al reward. La implementació actual del `_ProfessionalBase._compute_reward()` no passa `adx` — per tant `regime_adaptive` usa el default (20.0, = règim neutral) tret que es subclassifiqui l'entorn.

**Solució a curt termini:** Usar `td3_multiframe` amb `reward_type: professional` per als primers entrenaments, i canviar a `regime_adaptive` un cop verificat que l'entorn injecta `adx`.

**Solució completa (futura extensió):** Afegir `adx` al bloc de context de `_ProfessionalBase._compute_reward()` de `environment_professional.py`, de la mateixa manera que s'hi injecta `vol_ratio`.

---

## 7. Models implementats

### `td3_professional` — TD3 + Sentiment 12H

**Fitxer:** `config/models/td3_professional.yaml`
**Algoritme:** TD3 (Twin Delayed DDPG)
**Entorn:** `BtcTradingEnvProfessionalContinuous` (idèntic a `sac_professional`)
**Reward:** `professional` (ATR-scaled, drawdown penalty, overtrading)

**Diferències clau vs `sac_professional`:**

| Paràmetre | SAC Professional | TD3 Professional |
|-----------|-----------------|-----------------|
| Algoritme | SAC | TD3 |
| `buffer_size` | 100.000 | 200.000 |
| `learning_rate` | 0.0003 | 0.001 |
| `policy_delay` | N/A | 2 |
| `target_policy_noise` | N/A | 0.2 |
| `action_noise_sigma` | N/A | 0.1 |
| Features noves | — | `fear_greed_value`, `funding_rate` |

**Features (18 total):**
```
close, rsi_14, macd, macd_signal, macd_hist, atr_14, atr_5, vol_ratio,
bb_upper_20, bb_lower_20, bb_middle_20, ema_20, ema_50, adx_14,
volume, volume_norm_20,
fear_greed_value, funding_rate          ← NOVES vs C2
```

**obs_shape:** 18 × 90 + 4 = **1.624** (idèntic a sac_professional)

---

### `td3_multiframe` — TD3 + Multi-Timeframe 1H+4H

**Fitxer:** `config/models/td3_multiframe.yaml`
**Algoritme:** TD3
**Entorn:** `BtcTradingEnvProfessionalContinuous`
**Feature builder:** `MultiFrameFeatureBuilder` (1H base + 4H context)
**Reward:** `regime_adaptive` (per defecte) o `professional`

**Features (25 total):**
```
# 1H features (14)
close, rsi_14, macd, macd_signal, macd_hist, atr_14, atr_5, vol_ratio,
bb_upper_20, bb_lower_20, adx_14, volume_norm_20,
fear_greed_value, funding_rate

# 4H features (12) — suffix _4h
close_4h, rsi_14_4h, ema_20_4h, ema_50_4h, atr_14_4h, adx_14_4h,
vol_ratio_4h, macd_4h, macd_signal_4h, bb_upper_20_4h, bb_lower_20_4h
```

**obs_shape:** 25 × 60 + 4 = **1.504**

---

## 8. Com entrenar i optimitzar

### Prerequisits de dades

```bash
# 1. Descarregar candles (assegura 1h, 4h i 12h)
python scripts/download_data.py

# 2. Fear & Greed Index
python scripts/download_fear_greed.py

# 3. Funding rates
python scripts/update_futures.py

# 4. Verificar completesa
python scripts/check_data_completeness.py
```

### Smoke test (5 minuts — verifica que tot funciona)

```bash
python scripts/train_rl.py --agents td3_professional --smoke
python scripts/train_rl.py --agents td3_multiframe --smoke
```

### Optimitzar hiperparàmetres (Optuna)

```bash
# Optimitza tots els nous agents
python scripts/optimize_models.py --agents td3_professional td3_multiframe

# Només td3_professional (15 trials × ~25k steps = ~2h)
python scripts/optimize_models.py --agents td3_professional --trials 15
```

### Entrenament complet

```bash
# Entrena amb el config optimitzat (si existeix) o base
python scripts/train_rl.py --agents td3_professional td3_multiframe

# Comparativa final (incloent tots els agents RL)
python scripts/run_comparison.py --bots ppo_professional sac_professional td3_professional td3_multiframe
```

### Comanda per incluir TD3 a comparativa completa

```bash
python scripts/run_comparison.py --all
```

---

## 9. Com afegir nous agents

El projecte segueix el **patró registry** per mantenir totes les modificacions localitzades. Afegir un nou agent RL requereix tocar exactament **5 fitxers** (sempre els mateixos):

### Pas 1: Crear l'agent

```python
# bots/rl/agents/my_agent.py
from stable_baselines3 import MyAlgo
from core.interfaces.base_rl_agent import BaseRLAgent

class MyAgent(BaseRLAgent):
    @classmethod
    def from_config(cls, config): ...
    def train(self, train_env, val_env, total_timesteps): ...
    def act(self, observation, deterministic=True): ...
    def save(self, path): ...
    def load(self, path): ...
    def _validate(self, env): ...
```

### Pas 2: Afegir a `agents/__init__.py`

```python
from bots.rl.agents.my_agent import MyAgent
__all__ = ["SACAgent", "PPOAgent", "TD3Agent", "MyAgent"]
```

### Pas 3: Registrar en els 3 fitxers de runtime

**`bots/rl/trainer.py`** — `_AGENT_REGISTRY` i `_ENV_REGISTRY`
**`bots/rl/rl_bot.py`** — `_AGENT_REGISTRY`
**`core/backtesting/rl_optimizer.py`** — dins `_objective()`: `_AGENT_REGISTRY` i `_ENV_REGISTRY`

```python
# En cada un dels 3 fitxers:
_AGENT_REGISTRY = {
    ...
    "my_agent": MyAgent,          # ← afegir aquí
}
_ENV_REGISTRY = {
    ...
    "my_agent": BtcTradingEnvProfessionalContinuous,  # ← entorn que correspon
}
```

### Pas 4: Registrar en els scripts CLI

**`scripts/train_rl.py`** — `AVAILABLE_AGENTS`
**`scripts/run_comparison.py`** — `BOT_REGISTRY` i `_RL_KEYS`

```python
AVAILABLE_AGENTS = {
    ...
    "my_agent": "config/models/my_agent.yaml",
}
```

### Pas 5: Crear el YAML de configuració

```bash
cp config/models/td3_professional.yaml config/models/my_agent.yaml
# Editar: model_type, experiment_name, model_path, model hyperparams, bot.bot_id
```

### Checklist ràpid

```
[ ] bots/rl/agents/my_agent.py         (nova classe)
[ ] bots/rl/agents/__init__.py          (afegir import + __all__)
[ ] bots/rl/trainer.py                  (_AGENT_REGISTRY + _ENV_REGISTRY)
[ ] bots/rl/rl_bot.py                   (_AGENT_REGISTRY + on_observation branching)
[ ] core/backtesting/rl_optimizer.py    (_AGENT_REGISTRY + _ENV_REGISTRY dins _objective)
[ ] scripts/train_rl.py                 (AVAILABLE_AGENTS)
[ ] scripts/run_comparison.py           (BOT_REGISTRY + _RL_KEYS)
[ ] config/models/my_agent.yaml         (YAML unificat)
```

---

## 10. Agents futurs recomanats

### DreamerV3 (model-based RL)

**Quan implementar:** Quan tingueu més de 2M timesteps de dades (o un simulador de BTC ràpid).
**Avantatge:** Aprèn un model intern del mercat → pot planificar sense executar trades reals.
**Implementació suggerida:** Usar `dreamer-pytorch` o implementar des de zero seguint Hafner et al. 2023.

### Curriculum Learning

**Idea:** Entrenar en phases progressives:
1. **Fase 1 (Easy):** Dades 2019–2021 (tendències clares, bull market)
2. **Fase 2 (Medium):** Dades 2021–2022 (ATH + crash, alta volatilitat)
3. **Fase 3 (Hard):** Dades 2022–2024 (bear market + range)

**Com implementar:** Crear `CurriculumTrainer` que canvia el `df_train` en cada fase, recarregant el model de la fase anterior com a punt de partida.

### Multi-agent RL (ensemble)

**Idea:** Entrenar N agents amb seeds/configs diverses i fer voting/averaging de les seves accions.
**Avantatge:** Robustesa — si un agent falla, els altres compensen.
**Implementació suggerida:** `EnsembleRLBot` que agrega N `RLBot`.

### PPO amb LSTM (temporal memory)

**Idea:** Substituir MlpPolicy per RecurrentPPO (SB3-contrib).
**Avantatge:** L'agent pot recordar patrons multi-step sense necessitar un lookback fix.
**Prerequisit:** `pip install sb3-contrib`

### Long-Short amb marge simulat

**Extensió de l'entorn:** Ampliar l'action space de [0,1] a [-1,1] per permetre posicions curtes.
**Requeriment:** Implementar `short_fee_rate` i `margin_call_threshold` en `_ProfessionalBase`.

---

## Resum dels canvis de codi (C3)

### Fitxers nous

| Fitxer | Descripció |
|--------|-----------|
| `bots/rl/agents/td3_agent.py` | TD3Agent — wraps SB3 TD3 |
| `data/processing/multiframe_builder.py` | MultiFrameFeatureBuilder — 1H+4H merge |
| `bots/rl/rewards/advanced.py` | regime_adaptive reward function |
| `config/models/td3_professional.yaml` | TD3 + sentiment, 12H |
| `config/models/td3_multiframe.yaml` | TD3 + 1H+4H, multi-frame |

### Fitxers modificats

| Fitxer | Canvi |
|--------|-------|
| `bots/rl/agents/__init__.py` | + TD3Agent |
| `bots/rl/trainer.py` | + TD3Agent, + MultiFrameFeatureBuilder routing, + advanced rewards |
| `bots/rl/rl_bot.py` | + TD3 registry + continuous action parsing |
| `core/backtesting/rl_optimizer.py` | + TD3 registry + multiframe routing |
| `scripts/train_rl.py` | + td3_professional, td3_multiframe |
| `scripts/run_comparison.py` | + td3_professional, td3_multiframe |

### Cap canvi a la BD ni al schema d'entorn

Tots els canvis de C3 són **addictius**. Els agents C1 (ppo, sac) i C2 (ppo_professional, sac_professional) no es veuen afectats per cap canvi.

---

*Última actualització: Març 2026 · Versió 1.0*
