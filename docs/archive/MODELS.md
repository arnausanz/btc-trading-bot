# MODELS.md — BTC Trading Bot: Model Reference

> **Per què existeix aquest document?**
> Els fitxers YAML perden tots els comentaris quan `MLOptimizer.save_best_config()` els reescriu
> via `yaml.dump()`. Tota la documentació, justificació de paràmetres i decisions de disseny es
> manté aquí per garantir que no es perdi res entre cicles d'optimització.

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
8. [Gate System](#8-gate-system)
9. [Decisions de disseny globals](#9-decisions-de-disseny-globals)
10. [Comandes d'entrenament](#10-comandes-dentrenament)

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
1. optimize_ml.py / optimize_rl.py   → cerca d'hiperparàmetres (Optuna, k-fold)
   Resultat: secció best_params al YAML
2. train_ml.py / train_rl.py         → entrenament final amb best_params + dades completes
   Resultat: fitxer .pkl / .pt / directori de model
3. demo.yaml                         → activa el bot en paper trading
   Resultat: senyals en temps real registrats a MLflow / logs
```

### Mètrica d'optimització: per què `precision_mean` per ML?

Els models ML retornen `{accuracy_mean, precision_mean, recall_mean}`.

- **`accuracy_mean`** (descartada): en dades desbalancejades (més candles HOLD que BUY), un model
  que sempre prediu HOLD obté accuracy ~60%. No és útil per trading.
- **`precision_mean`** (triada): `TP / (TP + FP)`. Maximitza que quan el model diu BUY, tingui raó.
  Redueix falsos positius → menys trades innecessaris → menys comissions perdudes.
- **`recall_mean`** (descartada com a principal): maximitzaria captures de tots els BUYs però
  produiria molts falsos positius, el pitjor error en trading apalancat.

> **Bug corregit (Febr 2026):** Totes les YAML tenien `metric: sharpe_ratio`. Problema: `sharpe_ratio`
> no és al dict de retorn de `BaseTreeModel.train()`, per tant `metrics.get("sharpe_ratio", 0.0)`
> retornava 0.0 per a TOTS els trials d'Optuna, que seleccionava hiperparàmetres aleatòriament.
> Corregit a `precision_mean`.

---

## 2. Paràmetres comuns a tots els models ML

Aquests paràmetres apareixen a totes les YAML de models ML i tenen el mateix significat:

```yaml
# ── Etiquetatge ──────────────────────────────────────────────────────────────
threshold_pct: 0.01
# Moviment mínim de preu (1%) per considerar una candle com a BUY positiu.
# Valor <0.005 genera massa soroll (moviments trivials). Valor >0.02 genera massa
# pocs positius i el model no aprèn prou.
# Literatura: Sezer et al. (2020) recomanen 0.5-2% per a BTC 1H.
# Triat: 0.01 (1%) → equilibri entre senyal i soroll per a BTC/USDT 1H.

forward_window: 24
# Nombre de candles futures per calcular si el preu puja threshold_pct.
# 24 candles × 1H = mirar 24h endavant.
# Triat perquè captures swing moves de 1-2 dies, no micromoments de 1-2h.

# ── Inferència ───────────────────────────────────────────────────────────────
prediction_threshold: 0.55
# Probabilitat mínima per emetre senyal BUY. Default sklearn és 0.5.
# Augmentat a 0.55 per reduir falsos positius i millorar precision en producció.
# IMPORTANT: ha de ser consistent amb la mètrica d'optimització (precision_mean).

min_confidence: 0.65
# Segon filtre: el bot no obre posicions si la probabilitat és <0.65, fins i tot
# si supera prediction_threshold. Filtre conservador per mercats incerts.
# En paper trading es pot abaixar a 0.60 per observar més senyals.

# ── Features ─────────────────────────────────────────────────────────────────
bot:
  lookback: 200
# Candles de context que el bot llegeix per construir la finestra de features.
# Ha de ser > seq_len (neural nets) o > max_depth effectiu (arbres).
# 200 × 1H = 8.3 dies de context.

features:
  select: null
# null = usar totes les columnes de FeatureBuilder.
# Alternativa: llista explícita de columnes per reduir dimensionalitat.
# Per a arbres, null és acceptable. Per a neural nets, considerar selecció.

features:
  external: {}
# Fonts de dades externes. Exemple:
#   fear_greed: true           # Fear & Greed Index (CNN)
#   funding_rate: true         # Funding rate de futures perpetuals
#   funding_rate_symbol: BTC/USDT:USDT
# Deixar buit ({}) per models base sense dades on-chain.
```

---

## 3. Models ML supervisats — Arbres de decisió

### 3.1 XGBoost

**Fitxer:** `config/models/xgboost.yaml`
**Classe:** `bots.ml.xgboost_model.XGBoostModel`
**Tipus:** Gradient Boosting (XGBoost library, Chen & Guestrin 2016)

#### Descripció

XGBoost és un gradient boosting sobre arbres de decisió amb regularització L1/L2 nativa.
Combina molts arbres febles de manera seqüencial, on cada arbre corregeix els errors del
precedent. És el model de referència de la indústria per a dades tabulars i series temporals
amb features explícites.

**Avantatges:** velocitat, suport MPS/GPU, regularització robusta, interpretable via SHAP
**Limitacions:** no captura dependències temporals llargues (cal features manuals de context)

#### Comentaris YAML complets

```yaml
# ── Arquitectura ─────────────────────────────────────────────────────────────
training:
  model:
    n_estimators: 1000
    # Nombre d'arbres. Valor elevat amb early_stopping és la pràctica recomanada.
    # Chen & Guestrin (2016): n_estimators alt + lr baix > n_estimators baix + lr alt.
    # Amb 1000 arbres i lr=0.01 el model és estable. Optuna pot trobar menys.

    max_depth: 4
    # Profunditat màxima de cada arbre. Controla overfitting directament.
    # Literatura: Hastie et al. "Elements of Statistical Learning": max_depth=4-6
    # és l'òptim per a la majoria de problemes. Valors >8 sobreajusten.
    # Per a crypto: Sezer et al. (2020) recomanen depth ≤6. Triat: 4 (conservador).

    learning_rate: 0.01
    # Shrinkage factor. Valors baixos (0.01-0.05) amb molts arbres > valors alts
    # amb pocs arbres. Friedman (2001): lr<0.1 sempre superior asimptòticament.
    # NOTA: Optuna pot trobar valors millors entre 0.01-0.3.

    scale_pos_weight: 1.0
    # Pes de la classe positiva (BUY). Valor 1.0 = sense biaix.
    # BUG CORREGIT: era 2.0 → forçava el model a predir BUY massa sovint.
    # Resultava en 0 prediccions SELL en backtest. Tornat a 1.0.
    # Optuna pot triar valors 1.0-5.0 si les dades estan molt desbalancejades.

    subsample: 0.8
    # Fracció de mostres per construir cada arbre. Equivalent a bagging.
    # Friedman (2002): subsample=0.5-0.8 redueix overfitting significativament.
    # Triat: 0.8 (conserva la majoria de les dades però afegeix estocasticitat).

    colsample_bytree: 0.8
    # Fracció de features per construir cada arbre. Feature bagging.
    # Chen et al. recomanat: 0.6-0.9. Valor 0.8 evita que un sol feature domini.

    min_child_weight: 5
    # Suma mínima de pesos de les instàncies en un node fill.
    # Evita splits en nodes molt petits (memoritzar outliers).
    # Rang recomanat per finançament: 5-20. Augmentar si overfitting persisteix.

    reg_alpha: 0.1
    # Regularització L1 (Lasso). Força alguns pesos d'arbres a zero.
    # Útil per selecció de features implícita. Valor baix per no eliminar massa.

    reg_lambda: 1.0
    # Regularització L2 (Ridge). Suavitza els pesos sense eliminar-los.
    # XGBoost default és 1.0. Augmentar a 2-5 si hi ha overfitting.
```

#### Justificació de l'espai de cerca Optuna

```yaml
optimization:
  search_space:
    n_estimators: [500, 1500]
    # Rang ample: Optuna decidirà quants arbres calen amb lr variable.
    # Amb early_stopping implícit via CV, valors alts no sobreajusten.

    max_depth: [3, 6]
    # No busquem >6 per crypto: massa depth = overfitting garantit.
    # Zeng et al. (2023, "Crypto ML Survey"): depth 4-5 optimal per BTC.

    subsample / colsample_bytree: [0.6, 0.95]
    # Rang estàndard de la literatura. Optuna trobarà l'equilibri.

    min_child_weight: [1, 20]
    # Rang ampli. Valors alts (>10) útils si el dataset és petit.

    reg_alpha: [0.0, 1.0] / reg_lambda: [0.5, 5.0]
    # L2 mínim 0.5 per sempre tenir alguna regularització Ridge.
```

---

### 3.2 LightGBM

**Fitxer:** `config/models/lightgbm.yaml`
**Classe:** `bots.ml.lightgbm_model.LightGBMModel`
**Tipus:** Gradient Boosting leaf-wise (Ke et al., Microsoft Research 2017)

#### Descripció

LightGBM utilitza creixement leaf-wise (best-first) en lloc del level-wise d'XGBoost.
Resulta en arbres asimètrics que capturen interaccions no lineals complexes de forma
molt més eficient. És típicament 3-10x més ràpid que XGBoost i millora en datasets grans.

**Avantatges:** velocitat, eficiència de memòria, suport natiu per a features categòriques
**Limitacions:** més propens a overfitting que XGBoost si `num_leaves` és massa gran;
cal calibrar `min_child_samples` per controlar la mida mínima de fulles

#### Comentaris YAML complets

```yaml
training:
  model:
    n_estimators: 1000
    # Equivalent al d'XGBoost. LightGBM amb leaf-wise pot arribar al mateix
    # rendiment amb menys arbres, però mantenim 1000 + lr baix per seguretat.

    max_depth: 6
    # LightGBM pot créixer molt profund amb leaf-wise. max_depth=-1 seria
    # sense límit, però volem cap a 6 per trading crypto.
    # IMPORTANT: max_depth NO és el paràmetre principal de LightGBM,
    # num_leaves sí. max_depth és un hard cap.

    num_leaves: 31
    # Nombre màxim de fulles per arbre. Principal control de complexitat en LGBM.
    # Relació amb max_depth: num_leaves ≤ 2^max_depth.
    # Ke et al. (2017): num_leaves=31 és el default segur. Per >10k mostres
    # es pot augmentar a 63-127. Triat: 31 com a base conservadora.
    # NOTA: Optuna pot buscar 15-127 per trobar l'òptim.

    learning_rate: 0.01
    # Idèntica lògica que XGBoost.

    scale_pos_weight: 1.0
    # Ídem XGBoost. Bug de 2.0 corregit.

    min_child_samples: 20
    # Nombre mínim d'instàncies en una fulla. Paràmetre clau anti-overfitting de LGBM.
    # Equival a min_child_weight d'XGBoost però en nombre absolut de mostres.
    # LightGBM docs: augmentar a 50-100 per datasets grans o molt sorollosos.
    # Triat: 20 per BTC 1H (suficient per evitar memoritzar soroll).

    subsample: 0.8
    # Bagging fraction. Ke et al. recomanen 0.8-0.9 per LGBM.

    colsample_bytree: 0.8
    # Feature fraction. Ídem XGBoost.

    reg_alpha: 0.1
    # L1 regularization.

    reg_lambda: 1.0
    # L2 regularization. LGBM default és 0.0, augmentat a 1.0.
```

#### Per què LightGBM sobre XGBoost?

Per datasets grans (>50k candles amb moltes features), LGBM és sistemàticament més ràpid.
Per crypto 1H (52k+ candles), LGBM és el model preferit per iteració ràpida d'Optuna.
Chen et al. (Kaggle survey 2022): LGBM guanya en velocitat; XGBoost en estabilitat.
Tenir tots dos és la millor pràctica per ensembling.

---

### 3.3 CatBoost

**Fitxer:** `config/models/catboost.yaml`
**Classe:** `bots.ml.catboost_model.CatBoostModel`
**Tipus:** Gradient Boosting simètric (Prokhorenkova et al., Yandex 2018)

#### Descripció

CatBoost utilitza arbres simètrics (oblivious trees) on cada nivell de l'arbre aplica la
mateixa condició a totes les branques. Això el fa molt robust a overfitting en datasets
petits i molt eficient en inferència (les prediccions es fan com a lookup de taula).
CatBoost implementa "ordered boosting" que redueix el target leakage durant l'entrenament.

**Avantatges:** ordered boosting (menys leakage), molt robust a overfitting, no cal normalitzar
**Limitacions:** més lent en entrenament que LGBM; menys flexible en arquitectura

#### Comentaris YAML complets

```yaml
training:
  model:
    iterations: 1000
    # Equivalent a n_estimators. CatBoost utilitza "iterations" com a nomenclatura.

    depth: 6
    # Profunditat dels oblivious trees. Per la naturalesa simètrica, depth=6 és
    # equivalent a depth~4-5 d'un arbre asimètric en expressivitat pràctica.
    # Prokhorenkova et al. (2018): depth 6-8 optimal per dades tabulars.
    # Triat: 6 com a equilibri entre capacitat i regularització.

    learning_rate: 0.01
    # Ídem altres GBMs.

    scale_pos_weight: 1.0
    # CatBoost utilitza class_weights internament. Equivalent al d'XGBoost.

    l2_leaf_reg: 3.0
    # Regularització L2 a les fulles. Paràmetre principal de regularització de CatBoost.
    # Prokhorenkova et al.: valors 1-10. Default de CatBoost és 3.0.
    # Mantenim el default: és ben calibrat per la majoria de problemes.
    # Augmentar a 5-10 si hi ha overfitting. Abaixar a 1-2 si underfitting.

    min_data_in_leaf: 10
    # Nombre mínim de mostres a cada fulla. Anti-overfitting principal.
    # CatBoost default és 1 (massa permissiu). Augmentat a 10 per crypto.
    # Literatura de CatBoost: valors 5-50 en problemes de predicció financera.
```

#### Quan preferir CatBoost?

CatBoost brilla quan: (1) el dataset té features categòriques, (2) el dataset és relativament
petit (<20k mostres), (3) es vol el model més robust sense tuning extensiu.
Per BTC 1H amb features numèriques, CatBoost és la tercera opció (darrere XGB i LGBM) però
és un bon candidat per a l'ensemble.

---

### 3.4 Random Forest

**Fitxer:** `config/models/random_forest.yaml`
**Classe:** `bots.ml.random_forest.RandomForestModel`
**Tipus:** Bagging d'arbres de decisió independents (Breiman 2001)

#### Descripció

Random Forest construeix molts arbres de decisió de manera independent (paral·lela) i fa
voting de les prediccions. A diferència dels GBMs, no és seqüencial: cada arbre aprèn
de manera independent sobre un subset aleatori de dades i features. Menys potent que els
GBMs en datasets grans però molt estable i resistent a overfitting per construcció.

**Avantatges:** molt estable, resistent a overfitting, perfectament paral·lelitzable, sense
tuning crític, `class_weight="balanced"` incorporat
**Limitacions:** menys precís que GBMs en dades tabulars grans; no aprofita dependències
seqüencials

#### Comentaris YAML complets

```yaml
training:
  model:
    n_estimators: 1000
    # Nombre d'arbres. Breiman (2001): afegir més arbres sempre ajuda fins a un
    # punt de saturació (~500-1000 arbres). No hi ha overfitting per n_estimators
    # en Random Forest (a diferència de GBMs sense early stopping).

    max_depth: 15
    # Profunditat màxima. RF pot créixer arbres molt profunds sense overfitting
    # gràcies al bagging. max_depth=None (sense límit) és comú a la literatura.
    # Triat: 15 com a límit raonable per BTC 1H. Optuna buscarà 5-30.

    min_samples_leaf: 5
    # Mínim de mostres a cada fulla. Paràmetre anti-overfitting principal per RF.
    # Breiman (2001): min_samples_leaf=1 és el default però genera overfitting.
    # Recomanació literatura financera: 5-20. Triat: 5.

    max_features: sqrt
    # Nombre de features a considerar per split. "sqrt" = sqrt(n_features).
    # Breiman (2001): sqrt(n_features) és l'òptim teòric per classificació.
    # Alternativa: "log2" per datasets amb moltes features correlacionades.

    # class_weight="balanced"  ← configurat al codi (random_forest.py), no al YAML
    # Pesa les classes inversament proporcional a la seva freqüència.
    # Substitueix scale_pos_weight i és l'approach recomanat per RF.
```

---

## 4. Models ML supervisats — Xarxes neuronals

### 4.1 GRU Bidireccional

**Fitxer:** `config/models/gru.yaml`
**Classe:** `bots.ml.gru_model.GRUModel` (arquitectura: `BidirectionalGRU`)
**Tipus:** Gated Recurrent Unit bidireccional (Cho et al. 2014 + Schuster & Paliwal 1997)

#### Descripció

GRU és una simplificació de LSTM que elimina la cell state, reduint els paràmetres a
2/3 dels de LSTM amb rendiment comparable. La versió bidireccional processa la seqüència
temporal en les dues direccions (forward i backward) i concatena les representacions,
capturant dependències causals i de context futur (dins de la finestra).

**Arquitectura actual:**
```
Input [batch, seq_len, n_features]
  → BiGRU (num_layers capes, hidden_size, dropout entre capes)
  → Últim timestep [batch, hidden_size*2]  ← *2 per bidireccional
  → Dropout → FC(hidden_size*2 → hidden_size) → ReLU
  → Dropout → FC(hidden_size → 1) → Sigmoid
Output: probabilitat BUY [0,1]
```

**Avantatges:** capta dependències temporals llargues, bidireccional millora context
**Limitacions:** més lent de entrenar que arbres; requereix GPU/MPS per ser pràctic

#### Comentaris YAML complets

```yaml
training:
  model:
    seq_len: 96
    # Longitud de la finestra temporal d'entrada (en candles).
    # 96 candles × 1H = 4 dies de context.
    # Literatura GRU crypto: Livieris et al. (2020) usen 50-100 per BTC.
    # Gers et al. (2001): finestres llargues ajuden per tendències però
    # augmenten el temps de training. Triat: 96 com a equilibri.
    # Optuna busca: 72 (3d), 96 (4d), 168 (7d).

    hidden_size: 256
    # Dimensionalitat de l'estat ocult de la GRU. Bidireccional → sortida 512.
    # Cho et al. (2014): hidden_size=128-256 per tasques de seqüència moderada.
    # Per BTC amb moltes features, 256 és el punt de partida estàndard.
    # Optuna busca: 128, 256, 512. En un M4 Pro 256 és còmode, 512 és factible.

    num_layers: 2
    # Nombre de capes GRU apilades. Stacked GRUs milloren la capacitat.
    # Cho et al.: 2 capes supera 1 capa en la majoria de tasques.
    # 3+ capes rarament ajuden en series temporals financeres (Sezer 2020).
    # Triat: 2 capes. Optuna busca 1 o 2.
    # NOTA: dropout entre capes s'activa automàticament si num_layers > 1.

    dropout: 0.3
    # Regularització per evitar overfitting. S'aplica entre capes i als FC layers.
    # Srivastava et al. (2014): 0.2-0.5 per xarxes recurrents.
    # Zaremba et al. (2015) "RNN regularization": 0.3 recomanat per RNNs.
    # Triat: 0.3. Optuna busca 0.1-0.5.

    epochs: 100
    # Màxim d'epochs. Amb early_stopping (patience=15), rarament s'arriba a 100.
    # El training s'atura quan la val_loss no millora en 15 epochs consecutius.

    batch_size: 64
    # Mida del mini-batch. Valors petits (32-64) generen gradients més sorollosos
    # però actuen com a regularitzador implícit (Keskar et al. 2017).
    # Per M4 Pro amb MPS: 64-128 és l'òptim pràctic.

    learning_rate: 0.0003
    # Adam optimizer learning rate. Kingma & Ba (2015): 0.001 és el default.
    # Per series temporals financeres, lr conservador (0.0001-0.001) és millor.
    # Triat: 0.0003 (3×10^-4), valor estàndard de la literatura per RNNs.

    patience: 15
    # Early stopping patience. Si val_loss no millora en 15 epochs, s'atura.
    # Amb lr_scheduler (ReduceLROnPlateau, patience=2), el lr es redueix abans.
    # 15 epochs donen marge per sortir de mínims locals sense overfitting.
```

#### Notes d'implementació importants

- `_evaluate()` usa `threshold=0.5` (canonical binary threshold). **Bug corregit: era 0.35.**
- `predict()` default `threshold=0.5`. El bot afegeix una capa addicional via
  `prediction_threshold: 0.55` i `min_confidence: 0.65`.
- El device es detecta automàticament: MPS (M4 Pro) > CPU. MPS és ~3-5x més ràpid que CPU
  per GRU amb batch_size=64-128.
- `torch.set_num_threads(1)` evita deadlock BLAS/OpenMP conegut a macOS.

---

### 4.2 PatchTST

**Fitxer:** `config/models/patchtst.yaml`
**Classe:** `bots.ml.patchtst_model.PatchTSTModel`
**Tipus:** Transformer basat en patches (Nie et al., 2023, "A Time Series is Worth 64 Words")

#### Descripció

PatchTST és una arquitectura Transformer per a series temporals que divideix la seqüència
en patches (segments) i tracta cada patch com un "token" del Transformer. Inspirat en
ViT (Vision Transformer), on els patches d'imatge són tokens. Avantatges clau:

1. **Reducció de la longitud de seqüència**: patches de mida 16 redueixen 168 timesteps
   a ~10 tokens. Self-attention és O(n²) → molt més eficient.
2. **Captura de patrons locals**: cada patch representa un segment temporal coherent.
3. **Channel independence**: cada feature és tractada independentment (no cross-feature
   attention), evitant overfitting en datasets petits.

**Arquitectura actual:**
```
Input [batch, n_features, seq_len]  ← channel-independent
  → Patch embedding: seq_len → n_patches (patch_len=16, stride=8)
  → Positional encoding
  → Transformer encoder (n_layers blocs, d_model, n_heads)
  → [CLS] token → FC → Sigmoid
Output: probabilitat BUY [0,1]
```

#### Comentaris YAML complets

```yaml
training:
  model:
    seq_len: 168
    # 168 candles × 1H = 7 dies de context. PatchTST és especialment eficient
    # amb seqüències llargues gràcies als patches.
    # Nie et al. (2023): seq_len=336-512 per forecasting. Per classificació
    # binària, 168-336 és suficient. Triat: 168 (7 dies de BTC = cicle setmanal).

    patch_len: 16
    # Mida de cada patch en timesteps. 168/16 = ~10 patches.
    # Nie et al. (2023): patch_len=16 és el valor per defecte recomanat.
    # patch_len massa petit → perd context local. Massa gran → perd resolució.

    stride: 8
    # Pas entre patches (overlapping patches). stride=8 amb patch_len=16
    # genera patches solapats al 50% → millor coverage temporal.
    # Nie et al.: stride = patch_len/2 és la recomanació estàndard.

    d_model: 256
    # Dimensió del model Transformer (embedding dimension).
    # Vaswani et al. (2017) "Attention is All You Need": d_model=512 per NLP.
    # Per series temporals amb dades limitades, 128-256 és suficient.
    # Triat: 256. Optuna busca 128, 256, 512.
    # En M4 Pro 24GB: 512 és factible però triga el doble sense millora garantida.

    n_heads: 8
    # Nombre de capçals d'atenció. Ha de dividir d_model exactament.
    # d_model=256, n_heads=8 → head_dim=32 (compacte però expressiu).
    # Vaswani et al.: n_heads = d_model/64 és la regla empírica.

    n_layers: 3
    # Nombre de blocs Transformer encoder.
    # Nie et al. (2023): 3 capes supera 1-2 en la majoria de tasques de TS.
    # >4 capes rarament ajuden en datasets petits (overfitting).
    # Optuna busca 2, 3, 4.

    dropout: 0.2
    # Regularització. Vaswani et al.: 0.1-0.3 per Transformers.
    # Per TS financeres amb dades limitades, 0.2-0.3 és conservador i segur.

    epochs: 100    # Amb early_stopping patience=15
    batch_size: 64
    learning_rate: 0.0001
    # Adam LR. Transformers solen necessitar LR baixa (1e-4 a 5e-4).
    # LR conservadora compensa l'absència de warmup schedule.

    patience: 15
```

---

### 4.3 TFT (Temporal Fusion Transformer)

**Fitxer:** `config/models/tft.yaml`
**Classe:** `bots.ml.tft_model.TFTModel`
**Tipus:** Transformer amb mecanismes especialitzats per TS (Lim et al., Google 2021)

#### Descripció

TFT (Temporal Fusion Transformer) és l'estat de l'art en series temporals tabulars fins 2023.
Incorpora mecanismes especialitzats que PatchTST no té:

- **Variable Selection Networks (VSN)**: selecció automàtica de les features rellevants
- **Gated Residual Networks (GRN)**: gates per ignorar informació irrellevant
- **Interpretable Multi-Head Attention**: atenció sobre el temps (més interpretable)

Per al nostre cas (classificació binària), TFT actua com un encoder potent amb
self-attention temporal sobre la finestra de context.

**Avantatges:** el model més potent per series temporals tabulars amb context temporal
**Limitacions:** el més lent dels tres (GRU/PatchTST/TFT); requereix més dades per converger

#### Comentaris YAML complets

```yaml
training:
  model:
    seq_len: 168
    # Igual que PatchTST: 7 dies de context.
    # Lim et al. (2021): TFT millora amb seqüències llargues gràcies a l'atenció.

    d_model: 256
    # Dimensió interna. Lim et al. usen 160 en el paper original amb dataset gran.
    # Per BTC 1H, 128-256 és suficient. Triat: 256 (parity amb PatchTST).

    n_heads: 8
    # Capçals d'atenció. Lim et al.: n_heads=4 en el paper. Augmentat a 8 per
    # d_model=256 → head_dim=32, consistent amb PatchTST.

    n_layers: 3
    # Nombre de capes. Lim et al. (2021): valors 1-4 en experiments.
    # 3 capes és el punt de saturació per la majoria de tasques de TS financera.

    dropout: 0.2
    # Regularització. Igual que PatchTST.

    epochs: 100    # Amb early_stopping patience=15
    batch_size: 64

    learning_rate: 0.0003
    # TFT tolera LRs lleument més alts que PatchTST (0.0003 vs 0.0001)
    # perquè els GRN actuen com a regularitzadors addicionals.

    patience: 15
```

#### TFT vs PatchTST vs GRU: quan triar cada un?

| Criteri | GRU | PatchTST | TFT |
|---------|-----|----------|-----|
| Velocitat entrenament | ★★★ | ★★ | ★ |
| Capacitat temporal | ★★ | ★★★ | ★★★ |
| Robustesa overfitting | ★★ | ★★★ | ★★ |
| Interpretabilitat | ★ | ★★ | ★★★ |
| Dataset petit (<10k) | ★★★ | ★★ | ★ |
| M4 Pro factibilitat | ★★★ | ★★★ | ★★ |

En general per a BTC 1H: GRU és el millor compromís. PatchTST si tens paciència per
optimitzar seq_len. TFT si vols el model més expressiu i tens dades suficients.

---

## 5. Models d'Aprenentatge per Reforç (RL)

### Arquitectura RL Professional

Tots els agents RL professionals comparteixen:

```
Dades: BTC/USDT 12H (not 1H) — swing trading, no scalping
Features: 18 features tècnics + Fear&Greed + Funding Rate = 20 features
Entorn: BtcTradingEnvProfessionalContinuous
  - Acció contínua: float [-1, 1] (posició del portafoli)
  - Observació: lookback × n_features + 4 (position state: pos, entry_price, unrealized_pnl, drawdown)
  - Reward: professional o regime_adaptive
  - Stop Loss automàtic: ATR × stop_atr_multiplier
Timestep: 1 pas = 1 candle 12H
Total timesteps training: 500k (≈ 250 dies de 12H candles × ~2000 episodis)
```

**Per què 12H en lloc de 1H?**

1H genera massa soroll micro-estructural: els agents aprenen patrons de microestructura
sense valor predictiu. PPO/SAC baseline en 1H feia 1 trade cada 2 candles (overtrading),
destruint el retorn via comissions. 12H captura swing moves reals (2-5 dies) → menys
trades, comissions menors, molt més senyal. Sezer et al. (2020): RL per crypto millora
clarament en timeframes >4H.

**Pre-requisits** (executar abans d'optimitzar/entrenar RL professional):
```bash
python scripts/download_data.py        # Candles 1H + 4H + 12H a la DB
python scripts/download_fear_greed.py  # Fear & Greed Index
python scripts/update_futures.py       # Funding rate de perpetuals
```

---

### 5.1 PPO Professional

**Fitxer:** `config/models/ppo_professional.yaml`
**Agent:** `bots.rl.agents.PPOAgent` (Stable-Baselines3 PPO)
**Algorisme:** Proximal Policy Optimization (Schulman et al., OpenAI 2017)

#### Descripció

PPO és un algorisme on-policy que actualitza la política directament i usa un clip ratio
per evitar updates massa grans (destabilizadors). És el més estable de tots tres i el
millor punt de partida per a nous entorns. No requereix buffer de replay: cada experiència
s'usa una sola vegada i es descarta.

**Avantatges:** molt estable, poca sensibilitat als hiperparàmetres, bon default
**Limitacions:** menys eficient en dades que SAC/TD3 (off-policy); convergeix més lent

#### Comentaris YAML complets

```yaml
training:
  model:
    policy: MlpPolicy
    # Multi-Layer Perceptron Policy. L'observació és un vector pla (no imatge).

    n_steps: 2048
    # Nombre de passos per episode d'actualització (rollout buffer size).
    # Schulman et al. (2017): n_steps=2048 és el default recomanat per envs
    # continus. Valors grans (4096) milloren l'estimació del gradient però
    # alenteixen l'aprenentatge inicial.

    batch_size: 128
    # Mini-batch per a les actualitzacions de la política.
    # n_steps ha de ser divisible per batch_size: 2048/128 = 16 minibatches.

    n_epochs: 10
    # Nombre d'epochs sobre el rollout buffer per actualització.
    # PPO default: 10. Augmentar si la política tarda a convergir.

    gamma: 0.99
    # Factor de descompte (discount factor). Governa l'horitzó temporal.
    # gamma=0.99 → l'agent "mira" uns 100 passos endavant. Per 12H, 100 passos
    # = 50 dies. Adequat per swing trading de setmanes.

    learning_rate: 0.0003
    # Adam LR per PPO. SB3 default: 3e-4. Bon valor de partida.

    ent_coef: 0.01
    # Coeficient d'entropia. Afegeix un terme de regularització que incentiva
    # l'exploració: policy_loss += ent_coef * entropy.
    # Schulman et al. (2017): ent_coef=0.01 és l'estàndard.
    # Valors alts (0.05): més exploració → útil al principi de l'entrenament.
    # Valors baixos (0.001): convergència més ràpida però risc d'optimum local.
    # AFEGIT específicament per millorar exploració en RL de trading.
    # Optuna busca 0.001-0.05.

    clip_range: 0.2
    # Clip del ratio política nova/vella. El paràmetre clau de PPO.
    # Schulman et al. (2017): clip_range=0.2 és el default ben calibrat.
    # Valors menors (0.1): updates molt conservadors → aprenentatge lent.
    # Valors majors (0.3): més agressiu → inestabilitat possible.
    # Optuna busca 0.1-0.3.

    net_arch: [256, 256]
    # Arquitectura de les xarxes actor i crítica (capes FC).
    # SB3 default: [64, 64]. Augmentat a [256, 256] per l'espai d'observació gran
    # (lookback × 18 features + 4 = ~1624 dimensions).
    # Literatura RL per trading (Zhang et al. 2020): [256, 256] o [400, 300].

    total_timesteps: 500000
    # 500k passos és suficient per aprendre swing trading bàsic en 12H.
    # 1M passos milloraria els resultats si el temps d'entrenament ho permet (~4-8h M4 Pro).

  environment:
    reward_type: professional
    # Reward shaping professional: penalitza drawdowns, premia Sharpe, inclou
    # cost de transacció explícit. Definit a bots/rl/rewards/professional.py.

    stop_atr_multiplier: 2.0
    # Stop loss automàtic a entry_price ± 2×ATR14. Controla el risc màxim per
    # trade. Optuna pot optimitzar entre 1.5-3.0.
    # Zhang et al. (2020): ATR-based stops milloren Sharpe en crypto RL.

    fee_rate: 0.001
    # Comissió per trade (0.1%). Binance Futures taker fee estàndard.
    # IMPORTANT: inclou el cost en l'entrenament → evita aprendre overtrading.

optimization:
  probe_timesteps: 50000
  # Passos per trial d'Optuna. 50k ≈ 3 episodis complets sobre dades de training.
  # Augmentat de 25k (era massa poc: ~6 episodis totals).

  n_trials: 15
  # Nombre de configuracions que Optuna probarà.
  # 15 és el mínim pràctic. Augmentar a 25-30 si el temps ho permet.
```

---

### 5.2 SAC Professional

**Fitxer:** `config/models/sac_professional.yaml`
**Agent:** `bots.rl.agents.SACAgent` (Stable-Baselines3 SAC)
**Algorisme:** Soft Actor-Critic (Haarnoja et al., UC Berkeley 2018)

#### Descripció

SAC és un algorisme off-policy basat en el principi de Maximum Entropy RL: aprèn una
política que maximitza tant el reward com l'entropia de la política (exploració implícita).
Utilitza un experience replay buffer → molt més eficient en dades que PPO.

**Avantatges:** el més eficient en mostres dels tres; exploració implícita per max-entropy;
`gradient_steps=-1` actualitza a màxima velocitat
**Limitacions:** menys estable que PPO en entorns nous; off-policy pot ser inestable
amb reward shaping molt agressiu

#### Comentaris YAML complets

```yaml
training:
  model:
    policy: MlpPolicy

    buffer_size: 100000
    # Mida del replay buffer (experience replay). SAC és off-policy → aprèn de
    # experiències passades emmagatzemades.
    # Haarnoja et al. (2018): buffer gran (100k-1M) millora la diversitat
    # d'experiències. Per 500k timesteps, 100k és suficient (rotació cada 5 eps.).

    batch_size: 256
    # Mini-batch del replay buffer per actualització.
    # Haarnoja et al.: batch_size=256 és el default recomanat per SAC.

    gamma: 0.99

    gradient_steps: -1
    # Nombre d'actualitzacions del gradient per timestep.
    # -1 = "tan ràpid com sigui possible" (SB3: actualitza tantes vegades com
    # timesteps han passat des de l'última actualització).
    # Haarnoja et al.: gradient_steps=1 és conservador; -1 aprofita al màxim
    # el replay buffer per a una major eficiència en mostres.
    # CANVIAT DE 1 → -1 per millorar l'eficiència d'aprenentatge.

    learning_rate: 0.0003

    learning_starts: 1000
    # Nombre de passos aleatoris inicials per omplir el buffer.
    # 1000 passos garanteix una diversitat inicial suficient.

    tau: 0.005
    # Soft update parameter per les target networks.
    # Haarnoja et al.: tau=0.005 és el default estàndard. No canviar.

    net_arch: [256, 256]
    # Igual que PPO Professional.

    train_freq: 1
    # Freqüència d'actualització del model (cada N passos).

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

1. **Twin Critics**: dos crítics paral·lels → s'usa el mínim per evitar overestimació del Q-valor.
2. **Delayed Policy Updates**: l'actor s'actualitza menys freqüentment que el crític
   (`policy_delay`). Redueix la variança en l'estimació del gradient.
3. **Target Policy Smoothing**: afegeix soroll a les accions del target → regularitza
   l'actor evitant que exploiti regions estrets del Q-valor.

**Avantatges:** el més estable per a espais d'acció continus; menys hiperparàmetres crítics
que SAC; `policy_delay` és un regularitzador natural
**Limitacions:** menys explorador que SAC per default; necessita `action_noise` ben calibrat

#### Comentaris YAML complets

```yaml
training:
  model:
    policy: MlpPolicy

    buffer_size: 200000
    # Buffer gran (200k) per TD3. Fujimoto et al. (2018) usen 1M per MuJoCo.
    # Per 500k timesteps de BTC, 200k és el compromís (rotació 2.5x).

    batch_size: 256
    # Fujimoto et al.: batch_size=100-256.

    gamma: 0.99

    gradient_steps: 1
    # TD3 típicament usa gradient_steps=1 (a diferència de SAC que pot usar -1).
    # Fujimoto et al.: 1 gradient step per timestep per estabilitat amb twin critics.

    learning_rate: 0.001
    # TD3 tolera LRs lleugerament superiors que SAC/PPO.
    # Fujimoto et al.: 1e-3 per actor i crítics.

    learning_starts: 1000

    policy_delay: 2
    # L'actor s'actualitza cada policy_delay passos del crític.
    # Fujimoto et al. (2018): policy_delay=2 és el valor recomanat al paper.
    # Valors majors (3): l'actor s'actualitza menys → menys inestable però
    # aprèn més lentament. Optuna busca 1, 2, 3.

    target_policy_noise: 0.2
    # Soroll afegit a les accions del target per suavitzar el Q-valor.
    # Fujimoto et al.: 0.2 (Gaussian std). Optuna busca 0.1-0.4.

    target_noise_clip: 0.5
    # Clip del soroll de target. [-0.5, 0.5].
    # Fujimoto et al.: clip=0.5 com a valor estàndard. No cal canviar.

    action_noise_sigma: 0.1
    # Soroll d'exploració afegit a les accions durant l'entrenament.
    # Gaussian noise std. Fujimoto et al.: 0.1 és el default recomanat.
    # Optuna busca 0.05-0.3: valors alts = més exploració inicial.

    tau: 0.005
    # Soft update per target networks. Ídem SAC.

    net_arch: [400, 300]
    # Fujimoto et al. (2018) usen [400, 300] per DDPG/TD3 en MuJoCo.
    # Canviat de [256, 256] a [400, 300] per seguir el paper original.

    total_timesteps: 500000
```

---

### 5.4 TD3 Multiframe

**Fitxer:** `config/models/td3_multiframe.yaml`
**Agent:** `bots.rl.agents.TD3Agent`
**Tipus:** TD3 amb features multi-timeframe (1H + 4H)
**Entorn:** `BtcTradingEnvProfessionalContinuous`

#### Descripció

Variant de TD3 Professional que afegeix features de timeframe 4H a les features 1H.
L'objectiu és donar a l'agent context macro (tendència de 4H) mentre opera en 1H.

**Features 1H (14):** close, rsi_14, macd, macd_signal, macd_hist, atr_14, atr_5, vol_ratio,
bb_upper_20, bb_lower_20, adx_14, volume_norm_20, fear_greed_value, funding_rate

**Features 4H (11, suffix _4h):** close_4h, rsi_14_4h, ema_20_4h, ema_50_4h, atr_14_4h,
adx_14_4h, vol_ratio_4h, macd_4h, macd_signal_4h, bb_upper_20_4h, bb_lower_20_4h

**Total observació:** 25 features × lookback=60 + 4 position state = 1504 dimensions

#### Comentaris YAML clau

```yaml
aux_timeframes: [4h]
# Timeframes auxiliars per MultiFrameFeatureBuilder.
# El builder baixa dades 1H i 4H, fa merge per timestamp, i retorna un
# DataFrame amb features del timeframe primari + sufixos _4h.

features:
  lookback: 60
  # Finestra temporal en candles 1H. 60H = 2.5 dies de context 1H.
  # El timeframe principal és 1H (a diferència dels altres RL que operen en 12H).
  # Optuna busca 40, 60, 90.

optimization:
  probe_timesteps: 60000
  # Lleugerament superior (60k vs 50k) perquè la observació és més gran
  # (25 features vs 18) → l'agent necessita una mica més de passos per
  # aprendre la representació.

training:
  model:
    batch_size: 256
    buffer_size: 200000
    net_arch: [400, 300]
    total_timesteps: 600000
    # 100k passos addicionals respecte TD3 Professional per l'observació
    # més gran i la major complexitat de la tasca multi-timeframe.

  environment:
    reward_type: regime_adaptive
    # Reward que adapta la penalització/premi al règim de mercat actual
    # (tendència, lateralitat, alta volatilitat). Definit a bots/rl/rewards/advanced.py.
    # Millor que "professional" per entorns multi-timeframe perquè el règim
    # de 4H orienta quin comportament penalitzar.
```

#### Per que `regime_adaptive` en Multiframe i `professional` en els altres?

`regime_adaptive` utilitza el context de 4H per detectar si estem en:

- **Tendència forta (ADX > 25):** premia seguir la tendència, penalitza contra-tendència
- **Lateral (ADX < 20):** premia mean-reversion, penalitza ruptures falses
- **Alta volatilitat (ATR elevat):** redueix la mida efectiva de la posició

Amb features multi-timeframe, l'agent té prou informació per aprofitar el règim.
En els agents single-timeframe (12H), el `professional` és més senzill i estable.

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
| **HoldBot** | Buy & Hold | Una sola compra | Mai ven | 0.001% candles |

**Problema de l'EnsembleBot amb clàssics:** TrendBot (<0.1%) + MomentumBot (~0.5%) +
MeanReversionBot (~2%) → probabilitat de coincidència en la mateixa candle ≈ 0.1% × 0.5%
× 2% = 0.001% → ~5-10 senyals en 52k candles → pràcticament 0 trades.

---

## 7. EnsembleBot

**Fitxer:** `config/models/ensemble.yaml`
**Classe:** `bots.classical.ensemble_bot.EnsembleBot`

#### Descripció

Meta-bot que combina senyals de múltiples sub-bots via majority vote. Un senyal s'emet
quan >50% dels sub-bots actius coincideixen en BUY o SELL a la mateixa candle.

#### Configuració actual i problemes

```yaml
policy: majority_vote
# Únic policy implementat. Futurs: weighted, stacking.

sub_bots:  # Configuració actual: NOMÉS clàssics → 0 trades
  - config/models/trend.yaml
  - config/models/mean_reversion.yaml
  - config/models/momentum.yaml
```

**El problema:** TrendBot senyalitza <0.07% de les candles. La probabilitat que 2 de 3
bots coincideixin en la mateixa candle és pràcticament zero. Cap trade.

#### Pla per fer funcionar l'Ensemble

Un cop retrenats els models ML amb `metric: precision_mean`:

1. **Afegir XGB i LGBM** com a primers sub-bots ML (generen ~1.7% de senyals BUY →
   molt més freqüent que els clàssics → desbloquejarà el majority vote).
2. **Criteris mínims** per entrar: Sharpe > 1.0 + Drawdown > -25% en backtest out-of-sample.
3. **Nombre mínim recomanat:** 5 sub-bots actius per a majority_vote efectiu.

```yaml
# Configuració recomanada post-reentrenament:
sub_bots:
  # Clàssics (esparsos — necessiten suport ML per disparar el majority vote)
  - config/models/trend.yaml
  - config/models/mean_reversion.yaml
  - config/models/momentum.yaml
  # ML reentrenats (clau per desbloquejar el majority vote)
  - config/models/xgboost.yaml     # ~900 senyals / 52k candles
  - config/models/lightgbm.yaml    # ~900 senyals / 52k candles
  # Opcionals si superen el criteri Sharpe/Drawdown:
  # - config/models/catboost.yaml
  # - config/models/random_forest.yaml
  # RL professional un cop validats amb backtest out-of-sample:
  # - config/models/ppo_professional.yaml
```

---

## 8. Gate System

**Fitxer:** `config/models/gate.yaml`
**Classe:** `bots.gate.gate_bot.GateBot`
**Categoria:** `gate` (pipeline d'entrenament propi, diferent de ML/RL)
**Paradigma:** Swing trading seqüencial — 5 portes, long-only v1, 4H primari + 1D diari

### La idea en 30 segons

Imagina't un trader professional que, abans d'obrir una posició, es fa 5 preguntes per ordre:

1. **"Com està el mercat?"** (P1 — Règim) → Si estem en bear market, no compro.
2. **"El mercat està sa?"** (P2 — Salut) → Si hi ha eufòria o pànic extrem, redueixo la mida.
3. **"Hi ha un bon preu?"** (P3 — Estructura) → Si no hi ha un nivell de suport clar a prop, no compro.
4. **"Hi ha momentum ara?"** (P4 — Momentum) → Si el preu no es mou en la meva direcció, espero.
5. **"Quant arriscar?"** (P5 — Risc) → Calculo la mida exacta i verifico que no em passo dels límits.

Cada pregunta és una "porta" (gate). Si qualsevol porta diu NO, no hi ha trade. Punt.
Això fa que el bot sigui molt selectiu (~10-15 trades/mes) però cada trade ha passat 5 filtres rigorosos.

### Flux visual complet

```
                          CADA 4 HORES (candle 4H nova)
                                    │
                    ┌───────────────┴───────────────┐
                    │   Hi ha posicions obertes?     │
                    └───────┬───────────────┬───────┘
                      SÍ    │               │   NO
                            ▼               │
                   Gestionar posicions      │
                   (trailing stop,          │
                    sortides, exits)        │
                            │               │
                            ▼               ▼
              ┌──────────────────────────────────────┐
              │  P1 — RÈGIM MACRO (1×/dia)           │
              │  "En quin tipus de mercat estem?"     │
              │  HMM + XGBoost → 6 règims possibles  │
              │  STRONG_BULL / WEAK_BULL / RANGING    │──→ Passen
              │  STRONG_BEAR / WEAK_BEAR / UNCERTAIN  │──→ HOLD (porta tancada)
              └───────────────┬──────────────────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │  P2 — SALUT DEL MERCAT (1×/dia)      │
              │  "El mercat està sa o sobreescalfat?" │
              │  Fear & Greed (contrarian) × Funding  │
              │  → multiplier [0.0 – 1.0]            │
              │  Si multiplier = 0 → HOLD             │
              └───────────────┬──────────────────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │  P3 — ESTRUCTURA DE PREU (cada 4H)   │
              │  "Hi ha un bon nivell on comprar?"    │
              │  Pivots fractals + Fibonacci + Volume │
              │  Profile → busca suports forts        │
              │  Filtre: l'acostament té volum? >0.8  │
              │  Si no hi ha nivell → HOLD            │
              └───────────────┬──────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │  NEAR-MISS LOG  │ ← registra l'oportunitat
                     │  (P1+P2+P3 OK)  │   per anàlisi posterior
                     └────────┬────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │  P4 — MOMENTUM TRIGGER (cada 4H)     │
              │  "El preu es mou a favor?"            │
              │  4 senyals: derivades, RSI-2, MACD   │
              │  Mínim adaptat al règim (1–2 senyals)│
              │  Si no hi ha prou momentum → HOLD     │
              └───────────────┬──────────────────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │  P5 — RISC I SIZING (cada 4H)        │
              │  "Puc entrar? Amb quanta pasta?"      │
              │  Kelly fraccionari → mida exacta      │
              │  VETOs: massa posicions, drawdown,    │
              │         R:R insuficient               │
              │  Si VETO → HOLD (near-miss loguejat)  │
              └───────────────┬──────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │   BUY            │
                     │   Obre posició   │
                     │   Persisteix a BD│
                     └─────────────────┘
```

### Per què seqüencial i no paral·lel?

L'ordre importa. Cada porta depèn del resultat de l'anterior: P1 determina quants senyals exigeix P4 (en STRONG_BULL en cal 1, en RANGING en calen 2). P2 modula la mida de la posició que calcula P5. P3 defineix els nivells de stop i target que avalua P5. P4 proporciona la confiança que entra a la fórmula de sizing de P5. Si avaluéssim en paral·lel, hauríem de fer comunicació entre portes. Seqüencialment, cada porta ja té tota la informació que necessita.

### Dos timeframes: per què 4H + 1D?

El bot opera en **4H** (decisions cada 4 hores) però necessita **context diari**: el 4H és on busca nivells (P3), momentum (P4) i gestiona posicions (P5), prou granular per swing trading de 2-5 dies. El 1D és on P1 classifica el règim i P2 avalua la salut — indicadors que canvien lentament i un cop al dia n'hi ha prou.

#### Descripció tècnica detallada

El Gate System és un bot de swing trading que avalua 5 portes de forma seqüencial. Si qualsevol porta es tanca, no hi ha trade. L'arquitectura imita el procés de decisió d'un trader discrecional professional: context macro → salut del mercat → estructura de preu → trigger d'entrada → gestió de risc.

**Avantatges:** rigorós per construcció (alt signal-to-noise ratio), interpretable (saps exactament per quina porta ha passat o no cada oportunitat), adaptatiu al règim (P1 canvia el comportament de totes les portes), posicions gestionades activament (trailing stop + sortides condicionals).

**Limitacions:** freqüència baixa (~10-15 trades/mes), requereix entrenament P1 separat (HMM+XGBoost), no compatible amb `optimize_bots.py`/`train_models.py`.

---

### P1 — Règim Macro

**Fitxers:** `bots/gate/regime_models/hmm_trainer.py` + `bots/gate/regime_models/xgb_classifier.py` + `bots/gate/gates/p1_regime.py`

**Problema que resol:** No vols comprar en un bear market. P1 classifica el mercat en 6 estats per evitar operar a contracorrent.

**Com funciona (2 models en cascada):**

**Pas 1 — HMM (offline, mensual):** Un Hidden Markov Model descobreix "règims" ocults a les dades diàries. No li diem quants n'hi ha: prova K=2 fins a K=6 i tria el que millor explica les dades (per BIC). Amb 10 inicialitzacions aleatòries per K (ja que HMM depèn del punt de partida). Li dona 3 observacions simples per dia: rendiment diari, volatilitat normalitzada (ATR-14) i ràtio de volum. Resultat: etiqueta cada dia històric amb un número d'estat.

**Pas 2 — XGBoost (online, diari):** Entrena un classificador supervisat sobre les etiquetes de l'HMM, però usant 14 features més riques. Validació walk-forward amb 5 folds per evitar overfitting. Optimització amb Optuna. En producció, rep les features del dia i retorna: "STRONG_BULL amb 0.82 de confiança".

**Per què 2 models?** L'HMM necessita tota la sèrie temporal (offline). XGBoost pot predir en temps real, però necessita etiquetes per entrenar → les agafa de l'HMM.

**14 features de P1:**
```
ema_ratio_50_200, ema_slope_50, ema_slope_200    ← tendència EMA
adx_14, atr_14_percentile                        ← força i volatilitat
funding_rate_3d, funding_rate_7d                 ← sentiment derivats
volume_sma_ratio, volume_trend                   ← volum
fear_greed, fear_greed_7d_change                 ← sentiment retail
rsi_14_daily, returns_5d, returns_20d            ← momentum i retorns
```

**6 règims (ordenats per rendiment mitjà de l'HMM):**

| Règim | Permet entrades? | Invalida posicions? |
|-------|:---:|:---:|
| STRONG_BULL | Sí | — |
| WEAK_BULL | Sí | — |
| RANGING | Sí (amb R:R més alt) | — |
| UNCERTAIN (confiança < 0.60) | No | Sí, tanca posicions |
| WEAK_BEAR | No | Sí, tanca posicions |
| STRONG_BEAR | No | Sí, tanca posicions |

```yaml
p1:
  min_regime_confidence: 0.60
  # Si cap règim assoleix 0.60 de probabilitat → UNCERTAIN → porta tancada.
  # Protegeix contra mercats sense tendència clara on l'HMM oscil·la.
```

---

### P2 — Salut Qualitativa del Mercat

**Fitxer:** `bots/gate/gates/p2_health.py`

**Problema que resol:** Fins i tot en bull market, hi ha moments on el mercat està sobreescalfat o massa apalancat. P2 detecta aquests moments i redueix la mida de la posició (o la veta completament).

**Com funciona:** Calcula un `position_multiplier` entre 0.0 i 1.0 com a producte de dues puntuacions. Si qualsevol és 0, el multiplicador és 0 (veto total).

**Sub-score 1 — Fear & Greed Index (contrarian):**

| Fear & Greed | En Bull | En Bear | Per què |
|:---:|:---:|:---:|---|
| > 75 (Greed extrem) | 0.5 | 0.9 | Bull + greed = bombolla → reduir |
| 50-75 (Normal-Greed) | 0.8 | 0.8 | Neutral |
| 25-50 (Normal-Fear) | 0.9 | 0.7 | Bull + fear = oportunitat |
| < 25 (Fear extrem) | 1.0 | 0.5 | Bull + fear = compra forta. Bear + fear = pànic |

**Sub-score 2 — Funding Rate (proxy on-chain):**

| Funding Rate | Score | Per què |
|:---:|:---:|---|
| > 0.03% per 8h | 0.5 | Massa apalancat long → risc de liquidacions |
| 0 – 0.03% | 0.8 | Normal |
| < 0% | 1.0 | Desapalancat → favorable per a llargs |

**Multiplicador final** = FG_score × FR_score. Exemple: FG=80 en bull (0.5) × FR=0.04% (0.5) = 0.25 → posició un 25% del normal.

```yaml
p2:
  onchain_enabled: true
  # Activa el càlcul del funding rate score.
  # Desactivar → multiplier basat únicament en Fear & Greed.
  # News sentiment v2: no implementat en v1 (DECISIONS.md ADR-11-B).
```

**Avantatges:** simple, sense model, actualitzat diari. **Limitació:** no detecta narratives macro (ETF approval, halvings) — previst per v2 via LLM sentiment.

---

### P3 — Estructura de Preu

**Fitxer:** `bots/gate/gates/p3_structure.py`

**Problema que resol:** No vols comprar "al mig del no-res". P3 busca nivells de preu on el mercat ha reaccionat històricament (suports, resistències) i verifica que el preu actual n'és a prop.

**Com funciona (3 mètodes deterministes, cap ML):**

**1. Pivots fractals:** Busca punts on el preu ha fet un swing high (resistència) o swing low (suport). Un pivot amb N=2 vol dir que el high/low central és el més alt/baix de les 2 candles anteriors i posteriors. Usa N=2 a 4H (sensible) i N=5 a diari (robust).

**2. Fibonacci retracement:** Dins l'últim swing significatiu (≥3%), calcula els nivells clàssics de 38.2%, 50%, 61.8% i 78.6%. Zones on estadísticament el preu sol reaccionar en un retracement.

**3. Volume Profile (HVN):** Divideix el rang de preus en 50 bins i compta el volum a cada un. Els bins amb volum >1.5× la mitjana són "High Volume Nodes" — zones on el mercat ha transaccionat molt, que actuen com a imants o barreres.

**Merge i scoring:** Els nivells de les 3 fonts es consoliden (si dos nivells estan a <0.5% de distància, es fusionen). Força: 1 font → 0.3; 2 fonts → 0.6; 3 fonts → 0.9. +0.1 per cada toc anterior sense trencar. Proximity adaptativa: un nivell fort (≥0.7) atrau a 2×ATR de distància, un de dèbil a 1×ATR.

**Filtre de volum d'acostament:** si les últimes 3 candles 4H tenen menys del 80% del volum mitjà de 20 períodes, el preu s'acosta sense convicció → P3 rebutja l'oportunitat.

```yaml
p3:
  min_level_strength: 0.4     # 0.4 = mínim per 1 font + 1 toc anterior
  fractal_n_4h: 2             # 2 períodes → pivots sensibles (2-barra fractal)
  fractal_n_1d: 5             # 5 períodes → pivots robustos (5-barra fractal)
  min_swing_pct: 0.03         # swing mínim 3% per calcular Fibonacci
  volume_profile_bins: 50     # 50 bins ≈ ~$1.500-$2.000 per bin (BTC a ~$80k)
  # Aumentar a 100 bins per mercat molt volàtil o preus molt alts.
  # Abaixar a 25 per accelerar el càlcul (menys resolució).
```

**Retorna:** `P3Result(has_actionable_level, best_level, stop_level, target_level, risk_reward, volume_ratio, level_type, level_strength)`

---

### P4 — Momentum Trigger

**Fitxer:** `bots/gate/gates/p4_momentum.py`

**Problema que resol:** P3 ha trobat un bon nivell, però el preu podria estar-se movent en contra. P4 verifica que el momentum actual és favorable per entrar. Mesura l'estat actual, no prediu. 4 senyals calculats sobre derivades EWM-smoothed (span=3):

| Senyal | Condició | Significat |
|--------|----------|------------|
| D1 positiu | `d1 > 0 AND d2 >= 0` | Momentum positiu accelerant (tendència en marxa) |
| D1 frenant | `d1 < 0 AND d2 > 0` | Momentum negatiu frenant (possible reversal imminent) |
| RSI-2 Connors | `RSI-2 < 10` | Sobrevenut extrem en 2 períodes (Connors 2008) |
| MACD cross | MACD(12,26,9) creuant senyal en alça | Canvi de momentum a 4H |

**Mínims adaptatius per règim:**
```
STRONG_BULL: 1 senyal (la tendència ja és clara)
WEAK_BULL:   2 senyals (necessita confirmació)
RANGING:     2 senyals (mercat sense tendència → exigir més)
```

```yaml
p4:
  ewm_span: 3
  # Span per suavitzar les derivades. Span=3 és agressiu (reacciona ràpid).
  # Augmentar a 5 per menys falsos senyals en mercats sorollosos.
  # Abaixar a 2 per reaccions més ràpides (millor per STRONG_BULL).

  rsi2_oversold: 10
  # Connors & Alvarez (2009) "Short-Term Trading Strategies": RSI-2 < 10
  # identifica condicions de sobrevenut extrem amb un edge estadístic demostrat.
  # Es pot augmentar a 15 per generar més senyals RSI.
```

**Retorna:** `P4Result(triggered, confidence, signals_active, signals_total=4, signals_detail, d2_value)`

---

### P5 — Risc i Gestió

**Fitxer:** `bots/gate/gates/p5_risk.py`

**Problema que resol:** Quant arriscar en cada trade, i quan sortir de posicions obertes. P5 té dues responsabilitats: sizing d'entrada i gestió de posicions.

**A) SIZING D'ENTRADA — "Quant comprar?"**

Usa una variant del Kelly Criterion (sizing matemàticament òptim):

```
1. risc_usdt  = capital × 1% × P2_multiplier × P4_confiança
                 ↑            ↑                ↑
          mai arriscar     salut del        quants senyals
          més de l'1%      mercat          de P4 s'han activat

2. dist_stop  = |preu_entrada - stop| / preu_entrada
                  ↑ distància al stop en % (definida per P3)

3. posició    = risc_usdt / dist_stop
                  ↑ stop lluny → posició petita; stop a prop → posició gran

4. size       = min(posició / capital, 95%)
                  ↑ mai més del 95% del capital en 1 posició
```

**Condicions de VETO (bloquegen l'entrada):**

| VETO | Llindar | Per què |
|------|---------|---------|
| Massa posicions | ≥ 2 obertes | Diversificació i risc concentrat |
| Drawdown setmanal | > 5% de pèrdua | Protecció contra sèries dolentes |
| R:R insuficient | < 1.5× (BULL) o < 2.0× (RANGING) | No val la pena el risc |
| Stop invàlid | distància ≤ 0 | Error de càlcul en P3 |

**B) GESTIÓ DE POSICIONS OBERTES — "Quan sortir?"**

Cada 4H, per a cada posició oberta. Prioritat de sortida:

**Trailing stop adaptatiu (activa amb ≥ 1×ATR de guany):**
```yaml
trailing_atr_multiplier:
  low_vol:    1.5   # ATR percentil < 30 → stop ajustat, deixa espai al mercat calm
  normal_vol: 2.0   # ATR percentil 30-70 → equilibri risc/benefici
  high_vol:   2.5   # ATR percentil > 70 → stop ample, evita stop-out en volatilitat
```

**Sortides condicionals (posicions obertes):**

| Condició | Acció | Umbral |
|----------|-------|--------|
| Règim invalida long (`BEAR/UNCERTAIN`) | Tanca immediatament | — |
| P2 multiplier = 0 (emergència) | Tanca immediatament | — |
| Stop loss tocat | Tanca | `price ≤ stop_level` |
| Circuit breaker | Re-avalua P3/P4 | `|candle| > 3×ATR` |
| Desacceleració | Tanca si persisteix | `d2 < 0` per N candles 4H |
| Estancament | Redueix 50% | Negatiu > `stagnation_days` amb portes obertes |

```yaml
decel_exit_candles:
  STRONG_BULL: 5   # 5 × 4H = 20h en desacceleració → sortida
  WEAK_BULL:   3   # 3 × 4H = 12h → sortida
  RANGING:     2   # 2 × 4H = 8h → sortida (menys toleràcia per ranging)
```

---

### Near-Miss Logger

**Fitxer:** `bots/gate/near_miss_logger.py`

S'activa quan P1+P2+P3 passen simultàniament (independentment de si P4 o P5 bloquegen el trade). Persiste un registre complet a la taula `gate_near_misses` (veure DATABASE.md).

**Per a l'anàlisi post-demo:** permet identificar si P4 és massa restrictiu (molts near-misses on P4 falla), si P5 veta trades bons (via R:R o drawdown), i en quin règim hi ha més oportunitats perdudes.

---

### Entrenament Gate System

```bash
# Pre-requisit: BD actualitzada
alembic upgrade head                              # crea gate_positions + gate_near_misses

# Entrenament complet (HMM K=2..6 BIC + XGBoost Optuna)
python scripts/train_gate_regime.py
# Opcions:
# --n-trials 100     → Optuna 100 trials (default: 50)
# --no-optuna        → paràmetres XGBoost per defecte
# --symbol BTC/USDT  → símbol de la BD (default)

# Verificar sortida
ls models/gate_hmm.pkl models/gate_xgb_regime.pkl
```

**Temps estimat (M4 Pro):**
- HMM (K=2..6, 10 inits cadascun): ~2-5 min
- XGBoost Optuna 50 trials, walk-forward 5 folds: ~20-40 min
- Total: ~30-45 min

**Activació a demo:**
```yaml
# config/demo.yaml
- config_path: config/models/gate.yaml
  enabled: true   # canviar de false a true un cop entrenats els models
```

---

## 9. Decisions de disseny globals

### 9.1 Mètrica d'optimització

| Categoria | Mètrica | Justificació |
|-----------|---------|--------------|
| Tots els ML | `precision_mean` | Redueix falsos positius (BUY incorrectes), minimitza pèrdues per overtrading |
| Tots els RL | `val_return_pct` | Objectiu directe: maximitzar retorn en dades de validació |

### 9.2 Llindar de predicció

Tots els models ML usen una cadena de filtres de confiança:

```
Raw probability (model.predict_proba())
  → prediction_threshold: 0.55  (filtre 1: elimina senyals febles)
  → min_confidence: 0.65         (filtre 2: filtre addicional conservador en el bot)
```

**Bug corregit (v2):** `_evaluate()` en GRU/PatchTST/TFT usava `threshold=0.35` durant
cross-validation, donant una estimació excessivament optimista de precision durant
l'entrenament. Corregit a 0.5 (threshold canònic binari, consistent amb sklearn).

### 9.3 Etiquetatge supervisat

```
BUY label = 1  si  max(close[t+1:t+24]) / close[t] - 1 > 0.01 (1%)
BUY label = 0  en qualsevol altre cas (HOLD implícit)
```

`forward_window=24` i `threshold_pct=0.01` van junts. No canviar un sense l'altre.

### 9.4 Hardware M4 Pro 24GB RAM — guia pràctica

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

> **Nota MPS:** PyTorch MPS (Apple Silicon) funciona per a GRU/PatchTST/TFT però
> Stable-Baselines3 (RL) usa CPU. SB3 no suporta MPS de forma estable a data 2026.

---

## 10. Comandes d'entrenament

### Ordre recomanat

```bash
# ── Pre-requisits (ja executats) ──────────────────────────────────────────────
python scripts/download_data.py
# Nota: el Gate System no usa train_ml.py ni train_rl.py.
# Veure secció §8 Gate System per la comanda específica.
python scripts/download_fear_greed.py
python scripts/update_futures.py

# ── FASE 1: Optimitzar models ML ─────────────────────────────────────────────
# Podem córrer en seqüència (o en paral·lel si tens RAM suficient)
python scripts/optimize_ml.py --model xgboost
python scripts/optimize_ml.py --model lightgbm
python scripts/optimize_ml.py --model catboost
python scripts/optimize_ml.py --model random_forest
python scripts/optimize_ml.py --model gru
python scripts/optimize_ml.py --model patchtst
python scripts/optimize_ml.py --model tft

# ── FASE 2: Entrenar models ML amb best_params ────────────────────────────────
python scripts/train_ml.py --model xgboost
python scripts/train_ml.py --model lightgbm
python scripts/train_ml.py --model catboost
python scripts/train_ml.py --model random_forest
python scripts/train_ml.py --model gru
python scripts/train_ml.py --model patchtst
python scripts/train_ml.py --model tft

# ── FASE 3: Optimitzar RL professional ───────────────────────────────────────
# NOTA: cal executar seqüencialment (no en paral·lel per RAM i CPU)
python scripts/optimize_rl.py --agent ppo_professional
python scripts/optimize_rl.py --agent sac_professional
python scripts/optimize_rl.py --agent td3_professional
python scripts/optimize_rl.py --agent td3_multiframe

# ── FASE 4: Entrenar RL professional amb best_params ─────────────────────────
python scripts/train_rl.py --agents ppo_professional
python scripts/train_rl.py --agents sac_professional
python scripts/train_rl.py --agents td3_professional
python scripts/train_rl.py --agents td3_multiframe

# ── FASE 5: Backtest complet ──────────────────────────────────────────────────
python scripts/backtest.py

# ── FASE 5.5: Entrenar Gate System (independent del pipeline ML/RL) ──────────
alembic upgrade head
python scripts/train_gate_regime.py  # ~30-45 min (HMM + XGBoost Optuna)

# ── FASE 6: Activar a demo els models que superin el criteri ──────────────────
# Edita config/demo.yaml: canvia enabled: false → enabled: true per cada bot
# Criteri mínim ML/RL: Sharpe > 1.0 + MaxDrawdown > -25% en test out-of-sample
# Gate System: activar quan models/*.pkl existeixin (no cal criteri Sharpe previ)
```

### Flags útils

```bash
# Smoke test RL (50k steps, comprova que l'entorn funciona sense entrenar del tot)
python scripts/train_rl.py --agents ppo_professional --smoke

# Veure resultats a MLflow
mlflow ui --port 5000
# Obre http://localhost:5000 al navegador
```

### Notes sobre re-entrenament

**No cal canviar paràmetres entre runs d'optimització i entrenament.** El pipeline funciona
de manera autònoma: `optimize_ml.py` actualitza la secció `best_params` al YAML, i
`train_ml.py` llegeix el YAML i aplica els `best_params` automàticament via
`apply_best_params()` a `core/config_utils.py`.

Cada crida a `optimize_ml.py` reescriu el YAML (i perd els comentaris originals, d'aquí
l'existència d'aquest fitxer). Els valors de `best_params` es sobrescriuen amb els nous
resultats d'Optuna.

Els models RL es guarden a `models/<nom>/` (directori) i els ML a `models/<nom>.pkl` o
`models/<nom>.pt`. Pots re-entrenar sense esborrar: els fitxers es sobreescriuen.

Un cop els ML estan entrenats, avalua els resultats amb `backtest.py` i edita
`config/models/ensemble.yaml` per afegir els sub-bots que hagin superat el criteri de
Sharpe > 1.0 i Drawdown > -25%.

El Gate System (`train_gate_regime.py`) és un pipeline independent que no toca els YAMLs
dels altres models. Els seus models es guarden a `models/gate_hmm.pkl` i
`models/gate_xgb_regime.pkl`. Pot re-entrenar-se en qualsevol moment sense afectar la resta.

---

**Última actualització:** Març 2026 · Versió 3.0 (Gate System afegit — §8; seccions 8→9, 9→10)
