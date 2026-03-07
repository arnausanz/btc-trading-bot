# Rationale d'Hiperparàmetres dels Models ML

**Tasca:** Classificació binària — BTC/USDT horari, predicció si el preu puja >0.5% en les properes 24h
**Dataset:** ~52.000 candles horàries (2019–2024, période d'entrenament)
**Context del bot Trend (optimitzat):** EMA fast=40h, EMA slow=106h → el mercat té tendències a escala de 1.7–4.4 dies; els models ML han de capturar dinàmiques similars.

---

## Random Forest (`rf_experiment_1.yaml`)

| Paràmetre | Valor anterior | Valor nou | Perquè |
|-----------|---------------|-----------|--------|
| `n_estimators` | 100 | **500** | La literatura recomana 500–2000 per a crypto (Khaidem et al., ScienceDirect 2022). Reducció de variança sense cost de temps greu (RF és paral·lelitzable). |
| `max_depth` | 10 | **8** | Arbres menys profunds generalitzen millor en sèries de temps sorolloses. Lopez de Prado (*Advances in Financial ML*, 2018) recomana contenir la profunditat. 8 és el compromís entre expressivitat i robustesa. |

**Notes fixes (no configurable):**
- `class_weight="balanced"` → gestiona el desequilibri sense scale_pos_weight
- `random_state=42`, `n_jobs=-1`

---

## XGBoost (`xgb_experiment_1.yaml`)

| Paràmetre | Valor anterior | Valor nou | Perquè |
|-----------|---------------|-----------|--------|
| `n_estimators` | 200 | **500** | Compensa la reducció del learning_rate; amb lr=0.02 calen ~500 arbres per convergir bé. Revisat per Chen & Guestrin (2016) i confirmat en múltiples estudis de crypto 2023–2024. |
| `max_depth` | 6 | **5** | Reduir d'1 nivell disminueix l'overfitting en crypto (senyals sorollosos). Papers de classificació de cripto xifren el sweet spot en 4–6. |
| `learning_rate` | 0.05 | **0.02** | Recepta "golden" de XGBoost: lr baix + molts arbres = millor generalització. Confirmada en competicions Kaggle finance i estudis Ethereum/BTC (ACM 2023). |
| `scale_pos_weight` | 2.0 | 2.0 | Mantenim: biaix lleu cap a la classe positiva (senyal de compra) és desitjable. |

---

## LightGBM (`lgbm_experiment_1.yaml`)

| Paràmetre | Valor anterior | Valor nou | Perquè |
|-----------|---------------|-----------|--------|
| `n_estimators` | 300 | **600** | LightGBM és molt ràpid (histograma-based); podem doblar els arbres sense impacte de temps significatiu. |
| `max_depth` | 6 | **7** | Augmentat per permetre `num_leaves=50` (restricció: `num_leaves ≤ 2^max_depth = 128`). |
| `learning_rate` | 0.03 | **0.02** | Lleugerament més conservador per acompanyar més estimadors. |
| `num_leaves` | 31 | **50** | LightGBM creix leaf-wise (no level-wise). 50 fulles és un salt modest sobre el default 31, mantenint la regularització però amb més expressivitat. Conservador respecte al màxim possible (128). |
| `scale_pos_weight` | 2.0 | 2.0 | Mantenim. |

---

## CatBoost (`catboost_experiment_1.yaml`)

| Paràmetre | Valor anterior | Valor nou | Perquè |
|-----------|---------------|-----------|--------|
| `iterations` | 300 | **700** | CatBoost usa *ordered boosting* (evita target leakage intern), cosa que el fa menys propens a overfitting amb moltes iteracions. 700 és segur. Literatura crypto recomana 500–1000. |
| `depth` | 6 | 6 | Mantenim: arbres simètrics a profunditat 6 és el sweet spot per a dades tabulars financeres (Prokhorenkova et al., 2018). Cada nivell dobla les fulles = 64 fulles totals. |
| `learning_rate` | 0.03 | 0.03 | Mantenim: CatBoost és robust en el rang 0.01–0.1 gràcies a la seva regularització intrínseca. |
| `scale_pos_weight` | 2.0 | 2.0 | Mantenim. |

---

## GRU Bidireccional (`gru_experiment_1.yaml`)

| Paràmetre | Valor anterior | Valor nou | Perquè |
|-----------|---------------|-----------|--------|
| `seq_len` | 50 | **72** | 3 dies de context horari. Papers de predicció de cripto (PMC 2025, MDPI 2023) troben que 48–168h és el rang òptim per a BTC horari. 72h captura cicles diaris complets + tendències de mig termini coincidents amb els paràmetres del TrendBot (EMA slow=106h). |
| `hidden_size` | 64 | **128** | BiGRU amb 128 hidden = 256 features totals (forward + backward). Alineat amb estudis 2024 que reporten 100–128 unitats ocultes òptimes. Cost computacional moderat. |
| `num_layers` | 1 | 1 | Mantenim: PyTorch + MPS en macOS té un deadlock conegut amb `num_layers > 1` i GRU bidireccional. |
| `dropout` | 0.3 | **0.25** | El GRU intern té dropout=0.0 quan `num_layers=1` (limitació PyTorch). El 0.25 s'aplica als layers FC. Lleugerament relaxat perquè la regularització principal ve de l'early stopping. |
| `epochs` | 15 | **50** | Més epochs + early stopping és la pràctica estàndard. L'early stopping tallarà als ~15–25 epochs si no millora. |
| `batch_size` | 128 | **64** | Batches petits = gradients més sorollosos = regularització implícita. Recomanat en literatura de sèries temporals (32–64). |
| `learning_rate` | 0.001 | **0.0005** | Lr menor + més epochs = convergència més suau. Confirmat en estudis BiGRU crypto (MDPI Fractal Fract. 2023). |
| `patience` | 4 | **10** | BTC és volàtil: amb patience=4 l'early stopping és massa agressiu i para en locals mínims. 10 epochs de marge permet al model sortir de plateaus temporals. |

**Nota arquitectura:**
- Optimizer: Adam + ReduceLROnPlateau (factor 0.5, patience 2)
- Gradient clipping: `max_norm=1.0`
- Threshold predicció: 0.35 (biaix cap a recall)

---

## PatchTST (`patchtst_experiment_1.yaml`)

| Paràmetre | Valor anterior | Valor nou | Perquè |
|-----------|---------------|-----------|--------|
| `seq_len` | 96 | **168** | 7 dies (1 setmana). Captura patrons setmanals de BTC (dilluns/dimecres/divendres tenen comportaments específics). PatchTST original (Nie et al., ICLR 2023) mostra millores fins a seq_len=336. 168 és el compromís amb el cost computacional. Formula: **(168 - 24) // 12 + 1 = 13 patches**. |
| `patch_len` | 16 | **24** | 1 dia per patch = unitat semàntica natural per a BTC horari. Cada patch encapsula un cicle diari complet. Més semàntic que 16h que no té referent temporal natural. |
| `stride` | 8 | **12** | 50% de solapament (stride = patch_len / 2). Recomanació de l'article original. Equilibri entre redundància i cobertura. |
| `d_model` | 64 | **128** | Representació interna x2 de riquesa. Nie et al. (2023) usa 128–512. 128 és el mínim recomanat per capturar suficients patrons. Cost MPS manejable. |
| `n_heads` | 4 | **8** | 128 / 8 = 16 dims per cap d'atenció (format vàlid). Papers de finançament recomanen 8–16 caps. Més caps = atenció multi-granular. |
| `n_layers` | 2 | **3** | Una capa encoder addicional per jerarquia de features (local → global). Literatura: 3–4 capes per tasques financeres. |
| `dropout` | 0.1 | 0.1 | Mantenim: AdamW + weight_decay=1e-4 ja regularitza. Transformers amb dropout molt alt perden capacitat expressiva. |
| `epochs` | 20 | **50** | Idem GRU: early stopping gestiona el tall real. Transformers acostumen a nécessiter més epochs que RNNs. |
| `batch_size` | 64 | **32** | Batches petits milloren generalització en Transformers (Popel & Bojar, 2018). Gradient noise com a regularitzador. |
| `learning_rate` | 0.001 | **0.0001** | Factor 10x reducció. AdamW per a Transformers necessita lr molt baix (1e-4 és el gold standard en la literatura). Amb lr=0.001 els Transformers divergeixen o oscil·len. |
| `patience` | 5 | **10** | Transformers triguen més a convergir que RNNs. Necessiten marge per baixar del plateau inicial. |

**Nota arquitectura:**
- Optimizer: AdamW (weight_decay=1e-4) — millor que Adam per a Transformers (Loshchilov & Hutter, 2019)
- Scheduler: CosineAnnealingLR — millor que ReduceLROnPlateau per a Transformers
- Pre-LN (norm_first=True): més estable que Post-LN
- Gradient clipping: `max_norm=1.0`
- Threshold predicció: 0.35

---

## Paràmetres de dades (comuns a tots els models)

| Paràmetre | Valor | Perquè |
|-----------|-------|--------|
| `forward_window` | 24 | Prediem a 24h vista, alineat amb el cicle diari de BTC i l'EMA del TrendBot (EMA fast=40h). |
| `threshold_pct` | 0.005 | 0.5% de moviment mínim per considerar senyal. Filtra soroll de petit rang. Per sota d'aquest llindar, el cost de transacció (fees ~0.1%) fa que el trade no valgui la pena. |
| `symbol` | BTC/USDT | Actiu principal del sistema. |
| `timeframes` | 1h | Granularitat principal. Els models capten el context a través del seq_len/forward_window. |

---

## Resum de canvis per model

| Model | Canvis principals |
|-------|------------------|
| **Random Forest** | n_estimators 100→500, max_depth 10→8 |
| **XGBoost** | n_estimators 200→500, max_depth 6→5, lr 0.05→0.02 |
| **LightGBM** | n_estimators 300→600, max_depth 6→7, lr 0.03→0.02, num_leaves 31→50 |
| **CatBoost** | iterations 300→700 |
| **GRU** | seq_len 50→72, hidden 64→128, epochs 15→50, batch 128→64, lr 1e-3→5e-4, patience 4→10 |
| **PatchTST** | seq_len 96→168, patch 16→24, stride 8→12, d_model 64→128, heads 4→8, layers 2→3, epochs 20→50, batch 64→32, lr 1e-3→1e-4, patience 5→10 |

---

## Fonts principals consultades

- Nie et al. (ICLR 2023) — *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST)
- Loshchilov & Hutter (ICLR 2019) — *Decoupled Weight Decay Regularization* (AdamW)
- Prokhorenkova et al. (NeurIPS 2018) — *CatBoost: unbiased boosting with categorical features*
- Lopez de Prado (2018) — *Advances in Financial Machine Learning*
- Khaidem et al. (ScienceDirect 2022) — *Random Forest for crypto futures forecasting*
- PMC (2025) — *Cryptocurrency price prediction using GRU and LSTM for BTC, LTC, ETH*
- MDPI Fractal Fract. (2023) — *Forecasting Cryptocurrency Prices Using LSTM, GRU and Bi-Directional LSTM*
- ACM DL (2023) — *XGBoost for Classifying Ethereum Short-term Return*
