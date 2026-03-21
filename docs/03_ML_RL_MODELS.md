# Models ML i RL — Referència Completa

> Referència de tots els models d'aprenentatge automàtic (supervisat i per reforç) del projecte.
> Per al Gate System, veure **[02_GATE_SYSTEM.md](./02_GATE_SYSTEM.md)**.
> Per a les comandes de posada en marxa, veure **[07_OPERATIONS.md](./07_OPERATIONS.md)**.

**Última actualització:** Març 2026
**Versió pipeline:** Post-audit v2 (bug fix `sharpe_ratio → precision_mean`, regularització afegida)

---

## Índex

1. [Visió general i arquitectura](#1-visió-general-i-arquitectura)
2. [Paràmetres comuns a tots els models ML](#2-paràmetres-comuns-a-tots-els-models-ml)
3. [Models ML supervisats — Arbres de decisió](#3-models-ml-supervisats--arbres-de-decisió)
   - 3.1 [XGBoost](#31-xgboost)
   - 3.2 [LightGBM](#32-lightgbm)
   - 3.3 [CatBoost](#33-catboost)
   - 3.4 [Random Forest](#34-random-forest)
4. [Models ML supervisats — Xarxes neuronals](#4-models-ml-supervisats--xarxes-neuronals)
   - 4.1 [GRU Bidireccional](#41-gru-bidireccional)
   - 4.2 [PatchTST](#42-patchtst)
   - 4.3 [TFT (Temporal Fusion Transformer)](#43-tft-temporal-fusion-transformer)
5. [Models d'Aprenentatge per Reforç (RL)](#5-models-daprenentatge-per-reforç-rl)
   - 5.1 [PPO Professional](#51-ppo-professional)
   - 5.2 [SAC Professional](#52-sac-professional)
   - 5.3 [TD3 Professional](#53-td3-professional)
   - 5.4 [TD3 Multiframe](#54-td3-multiframe)
6. [Models clàssics (referència)](#6-models-clàssics-referència)
7. [EnsembleBot](#7-ensemblebot)
8. [Decisions de disseny globals](#8-decisions-de-disseny-globals)
9. [Comandes d'entrenament](#9-comandes-dentrenament)

---

## 1. Visió general i arquitectura

El pipeline té tres capes de models:

```
Capa 1: Models clàssics       — regles deterministes, sense entrenament
Capa 2: Models ML supervisats — aprenen patrons de price action
Capa 3: Models RL             — aprenen política de trading per reward
                                └─ EnsembleBot: meta-bot que combina senyals de múltiples sub-bots
```

### Flux de vida d'un model ML/RL

```
1. optimize_models.py / (--no-rl per saltar RL)  → cerca d'hiperparàmetres (Optuna, k-fold)
   Resultat: secció best_params al YAML

2. train_models.py / train_rl.py                  → entrenament final amb best_params + dades completes
   Resultat: fitxer .pkl / .pt / .zip al directori models/

3. run_comparison.py                              → backtest i validació walk-forward
   Resultat: mètriques Sharpe/Calmar a MLflow

4. demo.yaml                                      → activa el bot en paper trading
   Resultat: senyals en temps real registrats a la BD
```

### Mètrica d'optimització: per què `precision_mean` per ML?

Els models ML retornen `{accuracy_mean, precision_mean, recall_mean}`.

- **`accuracy_mean`** (descartada): en dades desbalancejades (més candles HOLD que BUY), un model que sempre prediu HOLD obté accuracy ~60%. No és útil per trading.
- **`precision_mean`** (triada): `TP / (TP + FP)`. Maximitza que quan el model diu BUY, tingui raó. Redueix falsos positius → menys trades innecessaris → menys comissions perdudes.
- **`recall_mean`** (descartada): maximitzaria captures de tots els BUYs però produiria molts falsos positius, el pitjor error en trading.

> **Bug corregit (Febr 2026):** Totes les YAML tenien `metric: sharpe_ratio`. Problema: `sharpe_ratio` no és al dict de retorn de `BaseTreeModel.train()`, per tant `metrics.get("sharpe_ratio", 0.0)` retornava 0.0 per a TOTS els trials d'Optuna, que seleccionava hiperparàmetres aleatòriament. Corregit a `precision_mean`.

---

## 2. Paràmetres comuns a tots els models ML

Aquests paràmetres apareixen a totes les YAML de models ML i tenen el mateix significat:

```yaml
# ── Etiquetatge ──────────────────────────────────────────────────────────────
threshold_pct: 0.01
# Moviment mínim de preu (1%) per considerar una candle com a BUY positiu.
# Valor <0.005 genera massa soroll. Valor >0.02 genera massa pocs positius.
# Literatura: Sezer et al. (2020) recomanen 0.5-2% per a BTC 1H.
# Triat: 0.01 (1%) → equilibri entre senyal i soroll per a BTC/USDT 1H.

forward_window: 24
# Nombre de candles futures per calcular si el preu puja threshold_pct.
# 24 candles × 1H = mirar 24h endavant.
# Captura swing moves de 1-2 dies, no micromoments.

# ── Inferència ───────────────────────────────────────────────────────────────
prediction_threshold: 0.55
# Probabilitat mínima per emetre senyal BUY. Default sklearn és 0.5.
# Augmentat a 0.55 per reduir falsos positius i millorar precision en producció.

min_confidence: 0.65
# Segon filtre: el bot no obre posicions si la probabilitat és <0.65.
# Filtre conservador per mercats incerts.

# ── Features ─────────────────────────────────────────────────────────────────
bot:
  lookback: 200
# Candles de context que el bot llegeix per construir la finestra de features.
# 200 × 1H = 8.3 dies de context.

features:
  select: null
# null = usar totes les columnes de FeatureBuilder.
# Alternativa: llista explícita per reduir dimensionalitat.

features:
  external: {}
# Fonts de dades externes. Exemple:
#   fear_greed: true       # Fear & Greed Index (CNN)
#   funding_rate: true     # Funding rate de futures perpetuals
```

---

## 3. Models ML supervisats — Arbres de decisió

### 3.1 XGBoost

**Fitxer:** `config/models/xgboost.yaml`
**Classe:** `bots.ml.xgboost_model.XGBoostModel`
**Tipus:** Gradient Boosting (XGBoost library, Chen & Guestrin 2016)

#### Descripció

XGBoost és un gradient boosting sobre arbres de decisió amb regularització L1/L2 nativa. Combina molts arbres febles de manera seqüencial, on cada arbre corregeix els errors del precedent. És el model de referència de la indústria per a dades tabulars.

**Avantatges:** velocitat, suport MPS/GPU, regularització robusta, interpretable via SHAP
**Limitacions:** no captura dependències temporals llargues (cal features manuals de context)

#### Paràmetres YAML comentats

```yaml
training:
  model:
    n_estimators: 1000
    # Nombre d'arbres. Valor elevat amb early_stopping és la pràctica recomanada.
    # Chen & Guestrin (2016): n_estimators alt + lr baix > n_estimators baix + lr alt.

    max_depth: 4
    # Profunditat màxima de cada arbre. Controla overfitting directament.
    # Per crypto: Sezer et al. (2020) recomanen depth ≤6. Triat: 4 (conservador).

    learning_rate: 0.01
    # Shrinkage factor. Valors baixos (0.01-0.05) amb molts arbres > valors alts.
    # Friedman (2001): lr<0.1 sempre superior asimptòticament.

    scale_pos_weight: 1.0
    # Pes de la classe positiva (BUY). Valor 1.0 = sense biaix.
    # BUG CORREGIT: era 2.0 → forçava el model a predir BUY massa sovint.

    subsample: 0.8
    # Fracció de mostres per construir cada arbre. Equivalent a bagging.
    # Friedman (2002): subsample=0.5-0.8 redueix overfitting significativament.

    colsample_bytree: 0.8
    # Fracció de features per construir cada arbre.

    min_child_weight: 5
    # Suma mínima de pesos de les instàncies en un node fill.
    # Evita splits en nodes molt petits.

    reg_alpha: 0.1    # Regularització L1 (Lasso)
    reg_lambda: 1.0   # Regularització L2 (Ridge)
```

#### Espai de cerca Optuna

```yaml
optimization:
  search_space:
    n_estimators:     [500, 1500]
    max_depth:        [3, 6]        # No busquem >6 per crypto
    subsample:        [0.6, 0.95]
    colsample_bytree: [0.6, 0.95]
    min_child_weight: [1, 20]
    reg_alpha:        [0.0, 1.0]
    reg_lambda:       [0.5, 5.0]   # L2 mínim 0.5 per sempre tenir regularització Ridge
```

---

### 3.2 LightGBM

**Fitxer:** `config/models/lightgbm.yaml`
**Classe:** `bots.ml.lightgbm_model.LightGBMModel`
**Tipus:** Gradient Boosting leaf-wise (Ke et al., Microsoft Research 2017)

#### Descripció

LightGBM utilitza creixement leaf-wise (best-first) en lloc del level-wise d'XGBoost. Resulta en arbres asimètrics que capturen interaccions no lineals complexes de manera molt més eficient. Típicament 3-10x més ràpid que XGBoost en datasets grans.

**Avantatges:** velocitat, eficiència de memòria, suport natiu per a features categòriques
**Limitacions:** més propens a overfitting si `num_leaves` és massa gran

#### Paràmetres YAML comentats

```yaml
training:
  model:
    n_estimators: 1000
    max_depth: 6        # Hard cap (el paràmetre principal de LGBM és num_leaves)
    num_leaves: 31
    # Principal control de complexitat en LGBM. Ha de ser ≤ 2^max_depth.
    # Ke et al. (2017): 31 és el default segur. Per >10k mostres, es pot augmentar.

    learning_rate: 0.01
    scale_pos_weight: 1.0   # Bug de 2.0 corregit

    min_child_samples: 20
    # Nombre mínim d'instàncies en una fulla. Paràmetre clau anti-overfitting de LGBM.
    # Equivalent a min_child_weight d'XGBoost però en nombre absolut.

    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 1.0     # LGBM default és 0.0, augmentat a 1.0
```

**Per què LightGBM sobre XGBoost?** Per datasets grans (>50k candles), LGBM és sistemàticament més ràpid. Per BTC 1H (52k+ candles), LGBM és el model preferit per iteració ràpida d'Optuna. Tenir ambdós és la millor pràctica per ensembling.

---

### 3.3 CatBoost

**Fitxer:** `config/models/catboost.yaml`
**Classe:** `bots.ml.catboost_model.CatBoostModel`
**Tipus:** Gradient Boosting simètric (Prokhorenkova et al., Yandex 2018)

#### Descripció

CatBoost utilitza arbres simètrics (oblivious trees) on cada nivell aplica la mateixa condició a totes les branques. Implementa "ordered boosting" que redueix el target leakage durant l'entrenament. Molt robust a overfitting en datasets petits.

**Avantatges:** ordered boosting (menys leakage), molt robust, no cal normalitzar
**Limitacions:** més lent en entrenament que LGBM

#### Paràmetres YAML comentats

```yaml
training:
  model:
    iterations: 1000    # Equivalent a n_estimators
    depth: 6
    # Profunditat dels oblivious trees. depth=6 en CatBoost ≈ depth=4-5 asimètric.
    # Prokhorenkova et al. (2018): depth 6-8 optimal per dades tabulars.

    learning_rate: 0.01
    l2_leaf_reg: 3.0
    # Regularització L2 a les fulles. CatBoost default. No canviar sense motiu.

    min_data_in_leaf: 10
    # Nombre mínim de mostres a cada fulla. Default de CatBoost és 1 (massa permissiu).
    # Augmentat a 10 per crypto.
```

**Quan preferir CatBoost?** Quan el dataset té features categòriques, és relativament petit (<20k mostres), o es vol el model més robust sense tuning extensiu. Per BTC 1H amb features numèriques, és la tercera opció (darrere XGB i LGBM), però bon candidat per ensembling.

---

### 3.4 Random Forest

**Fitxer:** `config/models/random_forest.yaml`
**Classe:** `bots.ml.random_forest.RandomForestModel`
**Tipus:** Bagging d'arbres de decisió independents (Breiman 2001)

#### Descripció

Random Forest construeix molts arbres de manera independent i fa voting de les prediccions. A diferència dels GBMs, no és seqüencial: cada arbre aprèn de manera independent sobre un subset aleatori de dades i features.

**Avantatges:** molt estable, resistent a overfitting per construcció, perfectament paral·lelitzable, `class_weight="balanced"` incorporat
**Limitacions:** menys precís que GBMs en dades tabulars grans

#### Paràmetres YAML comentats

```yaml
training:
  model:
    n_estimators: 1000
    # RF no sobreajusta per n_estimators (a diferència de GBMs sense early stopping).
    # Breiman (2001): afegir més arbres sempre ajuda fins a ~500-1000.

    max_depth: 15
    # RF pot créixer profund sense overfitting gràcies al bagging.
    # Triat: 15 com a límit raonable per BTC 1H.

    min_samples_leaf: 5
    # Paràmetre anti-overfitting principal per RF.
    # Literatura financera: 5-20. Triat: 5.

    max_features: sqrt
    # sqrt(n_features) és l'òptim teòric per classificació (Breiman 2001).

    # class_weight="balanced" ← configurat al codi, no al YAML
    # Pesa les classes inversament proporcional a la seva freqüència.
```

---

## 4. Models ML supervisats — Xarxes neuronals

### 4.1 GRU Bidireccional

**Fitxer:** `config/models/gru.yaml`
**Classe:** `bots.ml.gru_model.GRUModel` (arquitectura: `BidirectionalGRU`)
**Tipus:** Gated Recurrent Unit bidireccional (Cho et al. 2014)

#### Descripció

GRU és una simplificació de LSTM que elimina la cell state, reduint els paràmetres a 2/3 dels de LSTM amb rendiment comparable. La versió bidireccional processa la seqüència en les dues direccions i concatena les representacions.

**Arquitectura:**
```
Input [batch, seq_len, n_features]
  → BiGRU (num_layers capes, hidden_size, dropout entre capes)
  → Últim timestep [batch, hidden_size*2]   ← *2 per bidireccional
  → Dropout → FC(hidden_size*2 → hidden_size) → ReLU
  → Dropout → FC(hidden_size → 1) → Sigmoid
Output: probabilitat BUY [0,1]
```

#### Paràmetres YAML comentats

```yaml
training:
  model:
    seq_len: 96
    # 96 candles × 1H = 4 dies de context.
    # Livieris et al. (2020): 50-100 per BTC. Triat: 96 com a equilibri.
    # Optuna busca: 72 (3d), 96 (4d), 168 (7d).

    hidden_size: 256
    # Dimensionalitat de l'estat ocult. Bidireccional → sortida 512.
    # Optuna busca: 128, 256, 512.

    num_layers: 2
    # Stacked GRUs. Cho et al.: 2 capes supera 1 capa.
    # >3 capes rarament ajuden en series temporals financeres.
    # NOTA: dropout entre capes s'activa automàticament si num_layers > 1.

    dropout: 0.3
    # Zaremba et al. (2015) "RNN regularization": 0.3 recomanat per RNNs.

    epochs: 100         # Amb early_stopping patience=15
    batch_size: 64
    learning_rate: 0.0003
    patience: 15
```

**Notes d'implementació importants:**
- `_evaluate()` usa `threshold=0.5` (canonical binary). **Bug corregit: era 0.35.**
- MPS (Apple M-series) és ~3-5x més ràpid que CPU per GRU amb batch_size=64-128.
- `torch.set_num_threads(1)` evita deadlock BLAS/OpenMP conegut a macOS.

---

### 4.2 PatchTST

**Fitxer:** `config/models/patchtst.yaml`
**Classe:** `bots.ml.patchtst_model.PatchTSTModel`
**Tipus:** Transformer basat en patches (Nie et al. 2023, "A Time Series is Worth 64 Words")

#### Descripció

PatchTST divideix la seqüència temporal en patches (segments) i tracta cada patch com un token del Transformer. Inspirat en ViT (Vision Transformer). Avantatges clau:

1. **Reducció de la longitud de seqüència:** patches de mida 16 redueixen 168 timesteps a ~10 tokens. Self-attention és O(n²) → molt més eficient.
2. **Captura de patrons locals:** cada patch representa un segment temporal coherent.
3. **Channel independence:** cada feature tractada independentment, evitant overfitting.

**Arquitectura:**
```
Input [batch, n_features, seq_len]   ← channel-independent
  → Patch embedding: seq_len → n_patches (patch_len=16, stride=8)
  → Positional encoding
  → Transformer encoder (n_layers blocs, d_model, n_heads)
  → [CLS] token → FC → Sigmoid
Output: probabilitat BUY [0,1]
```

#### Paràmetres YAML comentats

```yaml
training:
  model:
    seq_len: 168
    # 168 candles × 1H = 7 dies. PatchTST és especialment eficient amb seqüències llargues.

    patch_len: 16
    # Mida de cada patch. 168/16 = ~10 patches.
    # Nie et al. (2023): patch_len=16 és el valor per defecte recomanat.

    stride: 8
    # Pas entre patches. stride=patch_len/2 = overlapping al 50% (recomanació de l'article).

    d_model: 256
    # Dimensió del model Transformer. Optuna busca 128, 256, 512.

    n_heads: 8
    # Ha de dividir d_model exactament. d_model=256, n_heads=8 → head_dim=32.

    n_layers: 3
    # Nie et al. (2023): 3 capes supera 1-2 en la majoria de tasques TS.

    dropout: 0.2
    epochs: 100
    batch_size: 64
    learning_rate: 0.0001
    patience: 15
```

---

### 4.3 TFT (Temporal Fusion Transformer)

**Fitxer:** `config/models/tft.yaml`
**Classe:** `bots.ml.tft_model.TFTModel`
**Tipus:** Transformer especialitzat per TS (Lim et al., Google 2021)

> ⚠️ **Estat: NO optimitzat ni entrenat.** El TFT és extremadament pesat en temps de computació (8-16h d'optimització, 8-16h d'entrenament). Pendent d'optimitzar i entrenar en un futur quan el hardware o el temps disponible ho permeti. Veure [ROADMAP.md](./ROADMAP.md) per a la planificació.

#### Descripció

TFT incorpora mecanismes especialitzats que PatchTST no té:

- **Variable Selection Networks (VSN):** selecció automàtica de les features rellevants.
- **Gated Residual Networks (GRN):** gates per ignorar informació irrellevant.
- **Interpretable Multi-Head Attention:** atenció sobre el temps (més interpretable que PatchTST).

**Avantatges:** el model més potent per series temporals tabulars amb context temporal
**Limitacions:** el més lent dels tres (GRU/PatchTST/TFT); requereix més dades per convergir

#### Paràmetres YAML comentats

```yaml
training:
  model:
    seq_len: 168      # 7 dies de context (igual que PatchTST)
    d_model: 256      # Lim et al. usen 160 en el paper; 256 per paritat amb PatchTST
    n_heads: 8        # head_dim=32, consistent
    n_layers: 3       # Punt de saturació per la majoria de tasques TS financeres
    dropout: 0.2
    epochs: 100
    batch_size: 64
    learning_rate: 0.0003
    # TFT tolera LRs lleument més alts (0.0003 vs 0.0001) perquè els GRN
    # actuen com a regularitzadors addicionals.
    patience: 15
```

#### Comparativa GRU vs PatchTST vs TFT

| Criteri | GRU | PatchTST | TFT |
|---------|-----|----------|-----|
| Velocitat entrenament | ★★★ | ★★ | ★ |
| Capacitat temporal | ★★ | ★★★ | ★★★ |
| Robustesa overfitting | ★★ | ★★★ | ★★ |
| Interpretabilitat | ★ | ★★ | ★★★ |
| Dataset petit (<10k) | ★★★ | ★★ | ★ |
| Factibilitat M4 Pro | ★★★ | ★★★ | ★★ |
| **Estat actual** | ✅ | ✅ | ⏳ Pendent |

En general per BTC 1H: **GRU és el millor compromís**. PatchTST si tens paciència per optimitzar `seq_len`. TFT si vols el model més expressiu i tens temps suficient.

---

## 5. Models d'Aprenentatge per Reforç (RL)

### Arquitectura RL Professional

Tots els agents RL professionals comparteixen:

```
Dades: BTC/USDT 12H (no 1H) — swing trading, no scalping
Features: 18 features tècnics + Fear&Greed + Funding Rate = 20 features
Entorn: BtcTradingEnvProfessionalContinuous
  - Acció contínua: float [-1, 1] (posició del portafoli)
  - Observació: lookback × n_features + 4 (position state: pos, entry_price, unrealized_pnl, drawdown)
  - Reward: professional o regime_adaptive
  - Stop Loss automàtic: ATR × stop_atr_multiplier
Timestep: 1 pas = 1 candle 12H
Total timesteps training: 500k (≈ 2000 episodis sobre dades de training)
```

**Per què 12H en lloc de 1H?**

1H genera massa soroll micro-estructural: els agents aprenen patrons sense valor predictiu. PPO/SAC baseline en 1H feia 1 trade cada 2 candles (overtrading), destruint el retorn via comissions. 12H captura swing moves reals (2-5 dies) → menys trades, comissions menors, molt més senyal. Sezer et al. (2020): RL per crypto millora clarament en timeframes >4H.

**Pre-requisits** (executar abans d'optimitzar/entrenar RL professional):
```bash
python scripts/download_data.py        # Candles 1H + 4H + 12H a la BD
python scripts/download_fear_greed.py  # Fear & Greed Index
python scripts/update_futures.py       # Funding rate de perpetuals
```

---

### 5.1 PPO Professional

**Fitxer:** `config/models/ppo_professional.yaml`
**Agent:** `bots.rl.agents.PPOAgent` (Stable-Baselines3 PPO)
**Algorisme:** Proximal Policy Optimization (Schulman et al., OpenAI 2017)

#### Descripció

PPO és un algorisme on-policy que actualitza la política directament i usa un clip ratio per evitar updates massa grans. És el més estable de tots tres i el millor punt de partida per a nous entorns.

**Avantatges:** molt estable, poca sensibilitat als hiperparàmetres, bon default
**Limitacions:** menys eficient en dades que SAC/TD3 (off-policy)

#### Paràmetres YAML comentats

```yaml
training:
  model:
    policy: MlpPolicy   # Multi-Layer Perceptron (observació és un vector pla)

    n_steps: 2048
    # Passos per episode d'actualització (rollout buffer).
    # Schulman et al. (2017): 2048 és el default recomanat per envs continus.

    batch_size: 128
    # n_steps ha de ser divisible: 2048/128 = 16 minibatches.

    n_epochs: 10
    # Epochs sobre el rollout buffer per actualització. PPO default: 10.

    gamma: 0.99
    # Discount factor. 0.99 → l'agent mira ~100 passos endavant.
    # Per 12H, 100 passos = 50 dies. Adequat per swing trading de setmanes.

    learning_rate: 0.0003
    ent_coef: 0.01
    # Coeficient d'entropia. Incentiva l'exploració: policy_loss += ent_coef * entropy.
    # Schulman et al. (2017): 0.01 és l'estàndard. Optuna busca 0.001-0.05.

    clip_range: 0.2
    # Paràmetre clau de PPO. Schulman et al.: 0.2 és el default ben calibrat.
    # Optuna busca 0.1-0.3.

    net_arch: [256, 256]
    # Actor i crítica. SB3 default [64, 64] insuficient per observació gran (~1624 dims).

    total_timesteps: 500000

  environment:
    reward_type: professional
    stop_atr_multiplier: 2.0
    fee_rate: 0.001    # Comissió 0.1% — inclosa en l'entrenament per evitar overtrading

optimization:
  probe_timesteps: 50000   # Passos per trial d'Optuna
  n_trials: 15
```

---

### 5.2 SAC Professional

**Fitxer:** `config/models/sac_professional.yaml`
**Agent:** `bots.rl.agents.SACAgent` (Stable-Baselines3 SAC)
**Algorisme:** Soft Actor-Critic (Haarnoja et al., UC Berkeley 2018)

#### Descripció

SAC és off-policy i es basa en el principi de Maximum Entropy RL: maximitza tant el reward com l'entropia de la política. Utilitza experience replay → molt més eficient en mostres que PPO.

**Avantatges:** el més eficient en mostres; exploració implícita per max-entropy
**Limitacions:** menys estable amb reward shaping molt agressiu

#### Paràmetres YAML comentats

```yaml
training:
  model:
    buffer_size: 100000
    # Replay buffer. Haarnoja et al. (2018): buffer gran millora diversitat d'experiències.

    batch_size: 256     # Haarnoja et al.: batch_size=256 és el default recomanat

    gamma: 0.99
    gradient_steps: -1
    # -1 = "tan ràpid com sigui possible". Aprofita al màxim el replay buffer.
    # CANVIAT DE 1 → -1 per millorar l'eficiència d'aprenentatge.

    learning_rate: 0.0003
    learning_starts: 1000    # Passos aleatoris inicials per omplir el buffer
    tau: 0.005               # Soft update per target networks. No canviar.
    net_arch: [256, 256]
    total_timesteps: 500000
```

#### SAC vs PPO: quan triar?

| Situació | Recomanació |
|----------|-------------|
| Primer entrenament, entorn nou | PPO (més estable) |
| Voleu eficiència màxima | SAC (off-policy, gradient_steps=-1) |
| Reward shaping agressiu | PPO (menys sensible) |
| Optimització Optuna ràpida | SAC (aprèn de les mateixes dades més vegades) |

---

### 5.3 TD3 Professional

**Fitxer:** `config/models/td3_professional.yaml`
**Agent:** `bots.rl.agents.TD3Agent` (Stable-Baselines3 TD3)
**Algorisme:** Twin Delayed DDPG (Fujimoto et al., McGill University 2018)

#### Descripció

TD3 millora sobre DDPG i SAC per a espais d'acció continus mitjançant tres tècniques:

1. **Twin Critics:** dos crítics paral·lels → s'usa el mínim per evitar overestimació del Q-valor.
2. **Delayed Policy Updates:** l'actor s'actualitza menys freqüentment que el crític (`policy_delay`).
3. **Target Policy Smoothing:** afegeix soroll a les accions del target → regularitza l'actor.

**Avantatges:** el més estable per a espais d'acció continus; `policy_delay` és un regularitzador natural
**Limitacions:** menys explorador que SAC per default

#### Paràmetres YAML comentats

```yaml
training:
  model:
    buffer_size: 200000     # Fujimoto et al. (2018) usen 1M per MuJoCo; 200k per BTC
    batch_size: 256
    gamma: 0.99
    gradient_steps: 1       # TD3 usa 1 (a diferència de SAC que pot usar -1)
    learning_rate: 0.001    # TD3 tolera LRs lleugerament superiors
    learning_starts: 1000
    policy_delay: 2         # Actor s'actualitza cada 2 steps del crític (paper: 2)
    target_policy_noise: 0.2   # Soroll afegit a accions del target. Optuna: 0.1-0.4
    target_noise_clip: 0.5
    action_noise_sigma: 0.1    # Soroll d'exploració. Optuna: 0.05-0.3
    tau: 0.005
    net_arch: [400, 300]    # Fujimoto et al. usen [400, 300] per DDPG/TD3
    total_timesteps: 500000
```

---

### 5.4 TD3 Multiframe

**Fitxer:** `config/models/td3_multiframe.yaml`
**Agent:** `bots.rl.agents.TD3Agent`
**Entorn:** `BtcTradingEnvProfessionalContinuous`

#### Descripció

Variant de TD3 Professional que afegeix features de timeframe 4H a les features 1H. L'objectiu és donar a l'agent context macro (tendència de 4H) mentre opera en 1H.

- **Features 1H (14):** close, rsi_14, macd, macd_signal, macd_hist, atr_14, atr_5, vol_ratio, bb_upper_20, bb_lower_20, adx_14, volume_norm_20, fear_greed_value, funding_rate
- **Features 4H (11, suffix `_4h`):** close_4h, rsi_14_4h, ema_20_4h, ema_50_4h, atr_14_4h, adx_14_4h, vol_ratio_4h, macd_4h, macd_signal_4h, bb_upper_20_4h, bb_lower_20_4h
- **Total observació:** 25 features × lookback=60 + 4 position state = 1504 dimensions

#### Paràmetres YAML clau

```yaml
aux_timeframes: [4h]
# El builder baixa dades 1H i 4H, fa merge per timestamp, retorna un DataFrame
# amb features del timeframe primari + sufixos _4h.

features:
  lookback: 60     # 60H = 2.5 dies de context 1H. Optuna busca 40, 60, 90.

training:
  model:
    batch_size: 256
    buffer_size: 200000
    net_arch: [400, 300]
    total_timesteps: 600000   # 100k addicionals per la major complexitat multiframe

  environment:
    reward_type: regime_adaptive
    # Adapta la penalització/premi al règim de mercat actual.
    # Millor que "professional" per multiframe perquè el règim de 4H orienta el comportament.
```

**Per què `regime_adaptive` en Multiframe i `professional` en els altres?**

`regime_adaptive` utilitza el context de 4H per detectar si estem en:
- **Tendència forta (ADX > 25):** premia seguir la tendència, penalitza contra-tendència.
- **Lateral (ADX < 20):** premia mean-reversion.
- **Alta volatilitat (ATR elevat):** redueix la mida efectiva de la posició.

Amb features multi-timeframe, l'agent té prou informació per aprofitar el règim. En els agents single-timeframe, el `professional` és més senzill i estable.

---

## 6. Models clàssics (referència)

Els models clàssics no tenen entrenament ni optimització. Es documenten aquí per completesa.

| Bot | Lògica | Senyal BUY | Senyal SELL | Freq. senyals |
|-----|--------|-----------|------------|---------------|
| **TrendBot** | EMA20 > EMA50, ADX > 25 | Creuament alcista | Creuament baixista | <0.1% candles |
| **MomentumBot** | RSI + MACD | RSI < 30 + MACD cross | RSI > 70 + MACD cross | ~0.5% candles |
| **MeanReversionBot** | Bollinger Bands | Tancament sota BB_lower | Tancament sobre BB_upper | ~1-2% candles |
| **GridBot** | Grid de preus | Compra en cada nivell baix | Ven en cada nivell alt | Continu |
| **DCABot** | Compra periòdica | Cada N candles | — | Continu |
| **HoldBot** | Buy & Hold | Una sola compra | Mai ven | ~0% |

**Problema de l'EnsembleBot amb clàssics purs:** TrendBot (<0.1%) + MomentumBot (~0.5%) + MeanReversionBot (~2%) → probabilitat de coincidència en la mateixa candle ≈ 0.001% → pràcticament 0 trades. La solució és afegir models ML com a sub-bots de l'ensemble.

---

## 7. EnsembleBot

**Fitxer:** `config/models/ensemble.yaml`
**Classe:** `bots.classical.ensemble_bot.EnsembleBot`

#### Descripció

Meta-bot que combina senyals de múltiples sub-bots via majority vote. Un senyal s'emet quan >50% dels sub-bots actius coincideixen en BUY o SELL a la mateixa candle.

#### Configuració actual i problemes

```yaml
policy: majority_vote
# Únic policy implementat. Futurs: weighted (pes per Sharpe), stacking (ML de 2a capa).

sub_bots:  # Configuració actual: NOMÉS clàssics → quasi 0 trades
  - config/models/trend.yaml
  - config/models/mean_reversion.yaml
  - config/models/momentum.yaml
```

#### Pla per fer funcionar l'Ensemble

Un cop retrenats els models ML amb `metric: precision_mean`:

```yaml
# Configuració recomanada post-reentrenament:
sub_bots:
  # Clàssics (esparsos — necessiten suport ML per disparar el majority vote)
  - config/models/trend.yaml
  - config/models/mean_reversion.yaml
  - config/models/momentum.yaml
  # ML reentrenats (clau per desbloquejar el majority vote)
  - config/models/xgboost.yaml      # ~900 senyals / 52k candles
  - config/models/lightgbm.yaml     # ~900 senyals / 52k candles
  # Opcionals si superen el criteri Sharpe/Drawdown:
  # - config/models/catboost.yaml
  # - config/models/random_forest.yaml
  # RL professional un cop validats:
  # - config/models/ppo_professional.yaml
```

**Criteri mínim per entrar a l'ensemble:** Sharpe > 1.0 + Drawdown > -25% en backtest out-of-sample.

---

## 8. Decisions de disseny globals

### 8.1 Mètrica d'optimització

| Categoria | Mètrica | Justificació |
|-----------|---------|--------------|
| Tots els ML | `precision_mean` | Redueix falsos positius (BUY incorrectes), minimitza pèrdues per overtrading |
| Tots els RL | `val_return_pct` | Objectiu directe: maximitzar retorn en dades de validació |

### 8.2 Cadena de filtres de confiança (ML)

Tots els models ML usen una cadena de filtres:

```
Raw probability (model.predict_proba())
  → prediction_threshold: 0.55  (filtre 1: elimina senyals febles)
  → min_confidence: 0.65         (filtre 2: filtre addicional conservador en el bot)
```

**Bug corregit (v2):** `_evaluate()` en GRU/PatchTST/TFT usava `threshold=0.35` durant cross-validation, donant una estimació excessivament optimista. Corregit a `threshold=0.5` (canònic binari, consistent amb sklearn).

### 8.3 Etiquetatge supervisat

```
BUY label = 1  si  max(close[t+1:t+24]) / close[t] - 1 > 0.01 (1%)
BUY label = 0  en qualsevol altre cas (HOLD implícit)
```

`forward_window=24` i `threshold_pct=0.01` van junts. No canviar un sense l'altre.

### 8.4 Hardware M4 Pro 24GB RAM — guia pràctica

| Model | RAM estimada | Temps estimat (optimize+train) | MPS? |
|-------|-------------|-------------------------------|------|
| XGBoost | <1 GB | 2-4h (50 trials) | No |
| LightGBM | <1 GB | 1-3h (50 trials) | No |
| CatBoost | <1 GB | 2-5h (50 trials) | No |
| Random Forest | <2 GB | 1-2h (30 trials) | No |
| GRU (hidden=512, seq=168) | ~4 GB | 4-8h (20 trials) | ✅ MPS |
| PatchTST (d=512, seq=336) | ~6 GB | 6-12h (20 trials) | ✅ MPS |
| TFT (d=512, seq=336) | ~8 GB | 8-16h (20 trials) | ✅ MPS |
| PPO Professional | ~2 GB | 2-4h (optimize) + 4-8h (train) | No (CPU) |
| SAC Professional | ~2 GB | 2-4h + 4-8h | No (CPU) |
| TD3 Professional | ~2 GB | 2-4h + 4-8h | No (CPU) |
| TD3 Multiframe | ~3 GB | 3-5h + 5-10h | No (CPU) |
| Gate HMM+XGBoost | <1 GB | ~30-45 min total | No |
| **Total seqüencial** | — | **~50-100h** | — |

> **Nota MPS:** PyTorch MPS (Apple Silicon) funciona per a GRU/PatchTST/TFT però Stable-Baselines3 (RL) usa CPU. SB3 no suporta MPS de forma estable a data 2026.

---

## 9. Comandes d'entrenament

### Ordre recomanat

```bash
# ── Pre-requisits ─────────────────────────────────────────────────────────────
python scripts/download_data.py
python scripts/download_fear_greed.py
python scripts/update_futures.py

# ── FASE 1: Optimitzar models ML (Optuna) ────────────────────────────────────
python scripts/optimize_models.py --no-rl              # tots els ML
python scripts/optimize_models.py --no-rl --trials 15  # versió ràpida

# ── FASE 2: Entrenar models ML amb best_params ────────────────────────────────
python scripts/train_models.py

# ── FASE 3: Optimitzar RL professional ───────────────────────────────────────
# NOTA: executar seqüencialment (no en paral·lel per RAM i CPU)
python scripts/optimize_models.py --model ppo_professional
python scripts/optimize_models.py --model sac_professional
python scripts/optimize_models.py --model td3_professional
python scripts/optimize_models.py --model td3_multiframe

# ── FASE 4: Entrenar RL professional amb best_params ─────────────────────────
python scripts/train_rl.py

# ── FASE 5: Backtest complet ──────────────────────────────────────────────────
python scripts/run_comparison.py --all

# ── FASE 5.5: Entrenar Gate System (pipeline independent) ─────────────────────
alembic upgrade head
python scripts/train_gate_regime.py   # ~30-45 min

# ── FASE 6: Activar a demo ────────────────────────────────────────────────────
# Edita config/demo.yaml: enabled: false → enabled: true per cada bot que superi:
# Criteri ML/RL: Sharpe > 1.0 + MaxDrawdown > -25% en test out-of-sample
# Gate System: activar quan models/*.pkl existeixin
```

### Flags útils

```bash
# Smoke test RL (comprova que l'entorn funciona sense entrenar del tot)
python scripts/train_rl.py --smoke

# Veure resultats a MLflow
mlflow ui --port 5000
# Obre http://localhost:5000
```

### Notes sobre re-entrenament

**No cal canviar paràmetres manualment.** El pipeline és autònom: `optimize_models.py` actualitza la secció `best_params` al YAML, i `train_models.py` llegeix el YAML i aplica els `best_params` automàticament via `apply_best_params()`.

Els models ML es guarden a `models/<nom>.pkl` o `models/<nom>.pt`. Els RL a `models/<nom>.zip`. Pots re-entrenar sense esborrar: els fitxers es sobreescriuen.

El Gate System (`train_gate_regime.py`) és un pipeline independent que no toca els YAMLs dels altres models. Pot re-entrenar-se en qualsevol moment sense afectar la resta.
