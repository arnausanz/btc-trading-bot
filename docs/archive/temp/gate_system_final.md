# GATE SYSTEM — Especificació Tècnica Final

## 1. Context del Projecte

Plataforma de trading algorítmic per BTC/USDT. Infraestructura existent: TimescaleDB, ccxt, Docker, Poetry, Feature Store amb indicadors tècnics, models ML (XGBoost, LightGBM, CatBoost, GRU, PatchTST), agents RL (PPO, SAC), bots clàssics (DCA, Grid), backtesting amb walk-forward validation, MLflow per tracking d'experiments, Optuna per optimització d'hiperparàmetres.

El Gate System és un nou bot dins `bots/gate/` que segueix el patró BaseBot + ObservationSchema + `from_config()`. No substitueix res — competeix amb els bots existents.

### Principi Fonamental

> **El sistema no prediu preus. Avalua context.**
>
> Resposta que dona: "Ara, les condicions són favorables per a un swing trade? Amb quina confiança? A quins nivells?"
>
> Fonament empíric: les sèries temporals de crypto són estadísticament indistingibles de soroll brownià en context univariant — models Naive superen consistentment LSTM, GRU, XGBoost per predicció de preus (Lahmiri & Bekiros, 2025, arXiv:2502.09079). Però el ~10% de traders profitables exploten asimetria de risc condicional al context, no predicció de preus.

### Objectiu

Swing trading BTC/USDT. ~10 trades/mes. Holding period típic: 2-5 dies. Timeframes operatius: Diari (context) + 4H (decisions).

---

## 2. Les 5 Portes — Visió General

El sistema avalua seqüencialment 5 portes. Si una porta es tanca, les inferiors no s'avaluen.

```
Candle diària tanca:
  → P1 avalua règim (1 cop/dia)
  → P2 avalua salut qualitativa (1 cop/dia)

Candle 4H tanca (6 cops/dia):
  → P3 actualitza nivells d'estructura
  → SI P1 oberta I P2 > 0 I P3 oberta:
      → P4 avalua momentum
      → SI P4 trigger:
          → P5 calcula sizing / veta
          → SI no vetat: EXECUTAR TRADE
  → Per posicions obertes:
      → P5 re-avalua (trailing, sortides condicionals)
  → Circuit breaker: moviment > 3× ATR en 1 candle → re-avaluació immediata
```

| Porta | Pregunta | Freqüència | Output |
|---|---|---|---|
| **P1 — Règim** | En quina direcció operar? | 1/dia | Estat (6 possibles) + probabilitats |
| **P2 — Salut** | El context confirma el biaix? | 1/dia | position_multiplier [0.0–1.0] |
| **P3 — Estructura** | Hi ha un nivell accionable? | Cada 4H | Entry, stop, target, R:R |
| **P4 — Momentum** | Ara és el moment? | Cada 4H | trigger (bool) + confidence [0–1] |
| **P5 — Risc** | Puc permetre-m'ho? Com gestiono? | Event + cada 4H | Sizing + ordres, o VETO |

---

## 3. Porta 1 — Règim Macro

### Funció

Classifica l'estat actual del mercat per definir la direcció permesa i les restriccions de les portes inferiors.

### Estats

| Estat | Què significa | Operacions permeses | Restriccions |
|---|---|---|---|
| STRONG_BULL | Tendència alcista consolidada | Llargs, condicions estàndard | Cap |
| WEAK_BULL | Alcista però perdent força | Llargs estàndard + curts amb restriccions severes | P2 reduït, P5 R:R elevat per contra-tendència |
| RANGING | Mercat lateral, sense direcció clara | Ambdues direccions | P2 reduït, P4 requereix alta confluència |
| WEAK_BEAR | Baixista però perdent força | Curts estàndard + llargs amb restriccions severes | P2 reduït, P5 R:R elevat per contra-tendència |
| STRONG_BEAR | Tendència baixista consolidada | Curts, condicions estàndard | Cap |
| UNCERTAIN | Transició, règim inidentificable | Cap operació | Porta tancada |

### Model: HMM (offline) + XGBoost (online)

#### Fase 1 — Descoberta de règims (HMM, offline, mensual)

L'HMM descobreix els estats latents del mercat sense etiquetes predefinides. S'entrena sobre l'historial complet amb les següents observacions:

- Retorns diaris de BTC
- ATR 14 diari (volatilitat)
- Volum diari normalitzat

**Selecció del nombre d'estats:** Entrenar HMMs amb K = 2, 3, 4, 5, 6 estats. Seleccionar K usant BIC (Bayesian Information Criterion) — penalitza models amb massa estats, evitant sobreajustament. La literatura convergeix en 4 estats com a òptim per a mercats financers (Nguyen, 2018), però s'ha de validar empíricament per a BTC.

**Implementació:** Gaussian HMM (`hmmlearn` library en Python). Emissions gaussianes multivariades. Algoritme Baum-Welch per estimació de paràmetres. Algoritme Viterbi per decodificació d'estats.

**Re-entrenament:** Mensual, amb tot l'historial disponible.

#### Fase 2 — Predicció de règim (XGBoost, online, cada dia)

Un XGBoost classificador entrenat amb les etiquetes de l'HMM com a target prediu l'estat actual usant features observables en temps real.

**Features d'entrada (calculades al tancament diari):**

| Feature | Descripció | Racionalitat |
|---|---|---|
| ema_ratio_50_200 | close / EMA_200 normalitzat per EMA_50 | Posició relativa en la tendència |
| ema_slope_50 | Pendent de l'EMA 50 (últims 10 dies) | Direcció de la tendència a mig termini |
| ema_slope_200 | Pendent de l'EMA 200 (últims 20 dies) | Direcció de la tendència a llarg termini |
| adx_14 | Average Directional Index 14 períodes | Força de la tendència (no direcció) |
| atr_14_percentile | Percentil de l'ATR actual vs últims 252 dies | Règim de volatilitat relatiu |
| funding_rate_3d | Mitjana del funding rate últims 3 dies | Posicionament del mercat apalancat |
| funding_rate_7d | Mitjana del funding rate últims 7 dies | Tendència del posicionament |
| volume_sma_ratio | Volum / SMA(20) del volum | Activitat relativa |
| volume_trend | Pendent de regressió lineal del volum (20 dies) | Tendència d'activitat |
| exchange_netflow_7d | Netflow exchanges (outflow - inflow) 7 dies | Acumulació (+) o distribució (-) |
| fear_greed | Fear & Greed Index actual | Sentiment agregat |
| fear_greed_7d_change | Canvi del FG en 7 dies | Tendència del sentiment |
| rsi_14_daily | RSI 14 al diari | Momentum a llarg termini |
| returns_5d | Retorn acumulat 5 dies | Momentum recent |
| returns_20d | Retorn acumulat 20 dies | Momentum a mig termini |

**Output:** Probabilitats per a cadascun dels K estats. L'estat amb probabilitat més alta defineix el biaix. La probabilitat s'usa com a `strength` [0–1].

Si cap estat supera el llindar de confiança (0.60), l'estat és UNCERTAIN i la porta es tanca.

#### Configuració del XGBoost

**Objectiu:** `multi:softprob` (classificació multiclasse amb probabilitats).

**Mètrica d'avaluació:** `mlogloss` (log-loss multiclasse).

**Hiperparàmetres inicials (punt de partida per Optuna):**

```yaml
xgboost_regime:
  objective: "multi:softprob"
  eval_metric: "mlogloss"
  num_class: 4  # o el K seleccionat per BIC
  
  # Complexitat d'arbre — mantenir baixa per evitar overfitting
  max_depth: [3, 7]              # Rang Optuna. Dades financeres: 3-5 sol ser òptim
  min_child_weight: [5, 50]      # Alt per evitar fulles amb poques observacions
  gamma: [0.1, 5.0]             # Penalització per split addicional
  
  # Sampling — introduir aleatorietat per robustesa
  subsample: [0.6, 0.9]         # No usar tot el dataset per arbre
  colsample_bytree: [0.5, 0.9]  # No usar totes les features per arbre
  
  # Regularització
  reg_alpha: [0.01, 10.0]       # L1 regularització (sparsity)
  reg_lambda: [1.0, 10.0]       # L2 regularització (shrinkage)
  
  # Boosting
  learning_rate: [0.005, 0.1]   # Log-uniform. Baix + early stopping
  n_estimators: 1000             # Fix alt, controlat per early stopping
  early_stopping_rounds: 50      # Atura quan validation loss no millora
  
  # Classe desbalancejada
  scale_pos_weight: auto         # Ajustar si els estats no són equiprobables
```

**Claus de configuració per a dades financeres:**
- `max_depth` baix (3-5) — les relacions en dades financeres rarament requereixen arbres profunds, i la profunditat és el vector principal d'overfitting.
- `min_child_weight` alt — amb ~1000 observacions diàries (3 anys), vols mínim 10-30 observacions per fulla.
- `subsample` i `colsample_bytree` < 1.0 — introdueixen aleatorietat que regularitza naturalment.
- `learning_rate` baix amb `early_stopping` — dóna temps al model per convergir sense memoritzar.

#### Validació

**Walk-forward obligatòria.** Folds temporals (no aleatoris). Mínim 5 folds. Train: 2 anys mínims. Test: 3-6 mesos.

**Mètrica principal:** Accuracy out-of-sample consistent en >55% en tots els folds. No buscar accuracy màxima en un fold.

**Mètrica secundària:** Estabilitat de la matriu de confusió entre folds — l'error ha de ser repartit, no concentrat en un estat.

#### Optimització

Optuna amb TPE sampler + Hyperband pruner (el que ja tens implementat). Espai de cerca definit al YAML anterior. L'objectiu d'Optuna ha de ser la **mitjana de mlogloss across walk-forward folds**, no la mlogloss d'un sol fold.

---

## 4. Porta 2 — Salut Qualitativa

### Funció

Modula la mida de la posició segons el context ambiental. Output: `position_multiplier` [0.0–1.0] que escala la posició calculada per la Porta 5.

### Components

Cada component produeix un sub-score [0, 1]. El multiplier final és el **producte** de tots els sub-scores. Un sol component a 0 = veto total.

**Fear & Greed Index (contrarian dins tendència):**

| Valor FG | En bull (P1) | En bear (P1) |
|---|---|---|
| 0–25 (por extrema) | 1.0 (oportunitat) | 1.0 (coherent) |
| 25–40 | 0.9 | 0.8 |
| 40–60 (neutre) | 0.7 | 0.7 |
| 60–75 | 0.5 | 0.5 |
| 75–100 (cobdícia extrema) | 0.2 (perill) | 0.3 |

**On-chain flows:** Ratio outflow/inflow d'exchanges. En bull: outflow > inflow = acumulació = score alt. Mapejat: `score = min(1.0, ratio / 1.5)`.

**News sentiment:** Score agregat [-1, +1] de CryptoPanic o FinBERT. Mapejat a [0.2, 1.0] segons coherència amb biaix de P1.

**LLM analysis (opcional, Model C):** Crida diària a Claude API. Output estructurat: {bias, confidence, risks}. Mapejat a score [0, 1].

### Efecte sobre posicions obertes

Només afecta **noves entrades**. Excepció: si multiplier = 0.0, sortida d'emergència de totes les posicions.

---

## 5. Porta 3 — Estructura de Preu

### Funció

Identifica nivells on el preu podria reaccionar, basant-se en on ha reaccionat històricament. NO prediu moviment — mapeja el camp de batalla.

### Mètodes (deterministes, sense ML)

**Pivots fractals:** Swing highs/lows. N=5 al diari, N=2 al 4H.

**Fibonacci retracement:** Nivells 0.382, 0.5, 0.618, 0.786 des de l'últim swing significatiu (≥3-5% de moviment al diari).

**Volume Profile:** High Volume Nodes — bins amb volum > 1.5× mitjana. Calculat al 4H.

**Força del nivell [0–1]:** 1 font = 0.3, 2 fonts = 0.6, 3 fonts = 0.9. Ajust: +0.1 per cada toc anterior sense trencar. Cap: 1.0.

**Volum d'acostament (confirmació independent):**

El Volume Profile mesura volum *acumulat* en preus (estructura passada). El volum d'acostament mesura si el *moviment actual* cap al nivell té convicció. Són dues coses diferents.

```python
# Últimes 3 candles 4H (12h d'acostament al nivell)
vol_approach  = volume_4h.iloc[-3:].mean()
vol_ma        = volume_4h.rolling(20).mean().iloc[-1]
volume_ratio  = vol_approach / vol_ma
```

| volume_ratio | Interpretació | Efecte sobre has_actionable_level |
|---|---|---|
| ≥ 1.3 | Acostament amb convicció | Confirmat — condició complerta |
| 0.8 – 1.3 | Volum neutre | Confirmat — condició complerta |
| < 0.8 | Acostament flaix, possible fake | **NO confirmat** — `has_actionable_level = False` encara que força ≥ 0.4 |

*Fonament:* Quan el preu s'apropa a un nivell important amb un volum per sota del 80% de la seva mitjana, indica falta d'interès real del mercat. Els falsos breakouts i les trampes de liquiditat tendeixen a ocórrer precisament en acostaments de baix volum.

### Output

| Camp | Descripció |
|---|---|
| has_actionable_level | Bool: preu a < proximity_threshold + força ≥ 0.4 + `volume_ratio ≥ 0.8` |
| best_level | El nivell accionable amb millor R:R |
| stop_level | Proper nivell per sota (llargs) o per sobre (curts) |
| target_level | Proper nivell per sobre (llargs) o per sota (curts) |
| risk_reward | Distància a target / distància a stop |
| volume_ratio | Ràtio de volum d'acostament (loguejat per anàlisi posterior) |

**Proximity threshold adaptatiu:** Funció de la força del nivell. Nivell fort (≥0.7): 2.0× ATR. Nivell mitjà (0.4–0.7): 1.5× ATR. Nivell feble (<0.4): 1.0× ATR.

> **Nota sobre correlació:** Afegir el volum a P3 (i no a P4) és una decisió conscient. P4 mesura momentum de preu — afegir-hi un senyal de volum com a "punt addicional" no resoldria la correlació, ja que el volum alt i els senyals de preu tendeixen a coocórrer per construcció (un moviment gran implica volum gran). En canvi, el volum d'acostament a P3 actua com a *filtre independent* abans d'arribar a P4: impedeix que P4 s'activi en moviments sense convicció, reduint els falsos positius de P4 sense alterar la seva lògica interna.

---

## 6. Porta 4 — Momentum / Derivades

### Funció

Trigger d'entrada. Mesura l'estat actual del momentum (no prediu).

### Càlcul de derivades (timeframe 4H)

```python
smoothed = close_4h.ewm(span=3).mean()
d1 = smoothed.pct_change()                  # Velocitat
d2 = d1.diff()                              # Acceleració
d1_smooth = d1.ewm(span=3).mean()
d2_smooth = d2.ewm(span=3).mean()
```

Lag efectiu: ~12h (3 candles × 4H). Coherent amb swings de 3 dies.

### Condicions de trigger (llargs en bull)

1. **d1 > 0, d2 ≥ 0** — Momentum positiu accelerant
2. **d1 < 0, d2 > 0** — Momentum negatiu frenant (possible reversal)
3. **RSI-2 (4H) < 10** — Sobrevenut extrem (Connors)
4. **MACD creuant senyal (4H)** — MACD(12,26,9) captura cicles ~4 dies

### Mínim de senyals (adaptatiu al règim P1)

| Règim P1 | A favor de tendència | Contra tendència |
|---|---|---|
| STRONG trend | 1/4 senyals | No permès |
| WEAK trend | 2/4 senyals | 3/4 senyals |
| RANGING | 2/4 senyals | 2/4 senyals |

Confiança: nombre de senyals actius / total senyals.

---

## 7. Porta 5 — Risc i Gestió de Posicions

### 7.1 Sizing d'Entrada

**El % de risc NO és la mida de la posició. És el que perds si toca el stop.**

```
risc_en_euros = capital × max_risk_pct × multiplier_P2 × confidence_P4
distancia_stop_pct = |entry - stop| / entry
posicio = risc_en_euros / distancia_stop_pct
posicio_final = min(posicio, capital × max_exposure_pct)
```

Fonament: Fractional Kelly Criterion (1/4 a 1/2 del Kelly teòric) amb cap dur de 1-2%.

**Condicions de VETO:**

| Condició | Valor configurable |
|---|---|
| R:R mínim no assolit | STRONG: 1.5:1, WEAK/contra: 3:1, RANGING: 2:1 |
| Massa posicions obertes | Configurable (ex: 2-3) |
| Drawdown setmanal excedit | Configurable (ex: 5%) |
| Exposició total excessiva | Suma posicions > max_total_exposure_pct (ex: 150%) |

### 7.2 Gestió de Posicions Obertes (cada 4H)

**A) Trailing stop adaptatiu a volatilitat:**

Activació: preu ha recorregut ≥ 1× ATR a favor. Abans, stop original (P3) fix.

```
trailing_stop = highest_since_entry - (ATR_14_4H × multiplier)

# Multiplier adaptatiu:
# ATR percentil < 30 (calma):   1.5×
# ATR percentil 30-70 (normal): 2.0×
# ATR percentil > 70 (volàtil): 2.5×
```

**B) Sortida per desacceleració (condicional al règim):**

d2 negativa durant N candles 4H consecutives → sortir.

| Règim P1 | N candles | Temps equivalent |
|---|---|---|
| STRONG trend | 5 | 20h |
| WEAK trend | 3 | 12h |
| RANGING | 2 | 8h |

**C) Sortida per invalidació de règim:** P1 canvia a estat incompatible → sortida immediata.

**D) Sortida d'emergència:** P2 multiplier = 0 → sortida immediata de tot.

**E) Reducció per estancament (condicional, NO timer fix):**

- Trade en positiu: trailing stop gestiona tot. Pot córrer indefinidament.
- Trade en negatiu + portes obertes: mantenir. Les condicions d'entrada segueixen vigents.
- Trade en negatiu + alguna porta tancada: sortir. Les condicions ja no sustenten el trade.
- Trade en negatiu + portes obertes + >2× temps esperat del swing (~6 dies): reduir posició al 50%.

**F) Circuit breaker:** Moviment > 3× ATR en 1 candle 4H → re-avaluació immediata de totes les portes.

---

## 8. On ML Afegeix Valor (i On No)

| Component | ML? | Tecnologia |
|---|---|---|
| P1: Classificació de règim | **Sí** | HMM (labels) + XGBoost (predictor) |
| P2: Ponderació components | Marginal | Producte simple suficient; logistic regression opcional |
| P3: Estructura de preu | **No** | Fractals + Fibonacci + Volume Profile (deterministes) |
| P4: Trigger momentum | **No** | Derivades + RSI + MACD (càlculs directes) |
| P5: Sizing | **No** | Kelly + ATR (fórmules) |
| P5: Trailing multiplier | **No** | Lookup table per ATR percentil |
| P5: N candles desacceleració | **No** | Lookup table per règim P1 |
| P5: R:R mínim | **No** | Lookup table per règim P1 |
| P5: Min senyals P4 | **No** | Lookup table per règim P1 |
| Data drift monitor | **Sí** | Detecció distribucional (KS-test o similar) |

**Raó:** El règim (P1, que sí és ML) ja captura la informació necessària per adaptar tots els paràmetres de les altres portes via lookup tables. Afegir ML addicional a P3-P5 afegiria risc d'overfitting sense guany demostrable.

---

## 9. Entrenament i Optimització

### Pipeline d'entrenament

```
1. Descarregar/actualitzar dades (OHLCV diari + 4H, FG, on-chain, funding)
2. Entrenar HMM amb K = 2..6, seleccionar K per BIC
3. Generar etiquetes de règim amb l'HMM seleccionat (Viterbi)
4. Calcular features de la taula P1
5. Entrenar XGBoost amb walk-forward (5+ folds)
6. Optimitzar hiperparàmetres amb Optuna (objectiu: mlogloss mitjana across folds)
7. Guardar model + config optimitzada a config/training/gate_regime_xgb_optimized.yaml
8. Validar: accuracy out-of-sample > 55% consistent en tots els folds
```

### Cadència de re-entrenament

| Component | Cadència | Raó |
|---|---|---|
| HMM | Mensual | Els règims evolucionen lentament |
| XGBoost | Setmanal | Les features poden degradar-se |
| Data drift check | Diari | Detecció ràpida d'anomalies |

### Criteris d'èxit

| Mètrica | Valor objectiu |
|---|---|
| Accuracy out-of-sample (tots els folds) | > 55% |
| Mlogloss estabilitat entre folds | Desviació < 15% |
| Matriu de confusió | Error repartit, no concentrat en 1 estat |
| Walk-forward Sharpe del sistema complet | > 0.5 |
| Walk-forward max drawdown | < 15% |

### Near-Miss Logger

**Motivació:** El GateBot és conservador per disseny. Per entendre si els llindars estan ben calibrats, cal saber quantes vegades les condicions eren quasi-bones (i quin pas exactament va tancar la porta). Sense aquest log, la única informació disponible és el nombre de trades executats.

**Punt de captura:** S'enregistra *cada cop que P1, P2 i P3 passen simultàniament*, independentment de si el trade s'executa o no. Quan les tres portes de context estan obertes, s'ha entrat en la "zona d'interès" — qualsevol resultat a partir d'aquí és informatiu.

**Esquema de la taula `gate_near_misses` (TimescaleDB):**

```sql
CREATE TABLE gate_near_misses (
    timestamp           TIMESTAMPTZ NOT NULL,
    bot_id              TEXT        NOT NULL,
    -- P1
    p1_regime           TEXT,
    p1_confidence       FLOAT,
    -- P2
    p2_multiplier       FLOAT,
    -- P3
    p3_level_type       TEXT,        -- 'support' | 'resistance'
    p3_level_strength   FLOAT,
    p3_risk_reward      FLOAT,
    p3_volume_ratio     FLOAT,       -- volum d'acostament / vol_ma20
    -- P4 (senyals individuals)
    p4_d1_ok            BOOLEAN,     -- derivada 1a positiva
    p4_d2_ok            BOOLEAN,     -- derivada 2a confirma
    p4_rsi_ok           BOOLEAN,     -- RSI-2 en zona
    p4_macd_ok          BOOLEAN,     -- MACD creuant
    p4_score            FLOAT,       -- ex: 0.75 = 3/4
    p4_triggered        BOOLEAN,
    -- P5
    p5_veto_reason      TEXT,        -- NULL si no vetat
    p5_position_size    FLOAT,
    -- Resultat
    executed            BOOLEAN      -- si el trade s'ha executat
);
SELECT create_hypertable('gate_near_misses', 'timestamp');
```

**Queries d'anàlisi previstes:**

```sql
-- Quina porta de P4 tanca més sovint?
SELECT
    SUM(CASE WHEN NOT p4_d1_ok   THEN 1 ELSE 0 END) AS falla_d1,
    SUM(CASE WHEN NOT p4_d2_ok   THEN 1 ELSE 0 END) AS falla_d2,
    SUM(CASE WHEN NOT p4_rsi_ok  THEN 1 ELSE 0 END) AS falla_rsi,
    SUM(CASE WHEN NOT p4_macd_ok THEN 1 ELSE 0 END) AS falla_macd
FROM gate_near_misses WHERE NOT executed;

-- P5 veta molt? Per quin motiu?
SELECT p5_veto_reason, COUNT(*)
FROM gate_near_misses
WHERE p4_triggered AND NOT executed
GROUP BY 1 ORDER BY 2 DESC;

-- Quantes oportunitats perdem per timeframe?
SELECT date_trunc('week', timestamp) AS week,
       COUNT(*) AS candidats,
       SUM(executed::int) AS executats
FROM gate_near_misses GROUP BY 1 ORDER BY 1;
```

**Implementació:** `bots/gate/near_miss_logger.py` — classe `NearMissLogger` amb mètode `log(gate_states: GateSnapshot)`. El `GateBot.on_observation()` crida `logger.log(...)` immediatament després que P3 passi.

---

## 10. Temporalitats Finals

| Component | Timeframe | Freqüència |
|---|---|---|
| P1 — Règim (EMAs, ADX, features XGBoost) | Diari | 1/dia |
| P2 — Salut (FG, on-chain, sentiment) | Diari | 1/dia |
| P3 — Nivells majors (fractals, Fibonacci) | Diari | 1/dia |
| P3 — Nivells intermedis (Volume Profile) | 4H | Cada 4H |
| P4 — Momentum (derivades, RSI, MACD) | 4H | Cada 4H |
| P5 — Sizing | 4H | Event-driven |
| P5 — Gestió posicions | 4H | Cada 4H |
| Circuit breaker | 4H | Cada 4H |
| Re-entrenament HMM | Historial complet | Mensual |
| Re-entrenament XGBoost | Walk-forward | Setmanal |
| Data drift monitor | Features actuals vs train | Diari |

---

## 11. Punts Febles Reconeguts

| # | Punt feble | Estat v1 | Mitigació |
|---|---|---|---|
| 1 | **Black swans** | Assumit | Circuit breaker (3× ATR) ja inclòs. Per a paper trading és acceptable. |
| 2 | **Mercat lateral prolongat** | Assumit (disseny correcte) | P1 en RANGING = no operar és el comportament esperat, no un bug. |
| 3 | **Correlació de senyals a P4** | **Mitigat** | Volum d'acostament afegit a P3 (§5). Filtra moviments sense convicció *abans* d'arribar a P4, reduint falsos positius sense alterar la lògica interna de P4. |
| 4 | **Overfitting de llindars** | Assumit | Walk-forward validation + paper trading com a test out-of-sample real. Cap defensa addicional necessària en v1. |
| 5 | **Cost d'oportunitat** | **Mitigat** | Near-miss logger afegit a §9. Cada vegada que P1+P2+P3 passen i el trade no s'executa, es registra l'estat complet de totes les portes. Permet calibrar llindars amb dades reals en v2. |

---

## 12. Decisions Descartades i Per Què

Aquesta secció documenta conclusions que s'han explorat i rebutjat, per evitar re-explorar-les.

### Timer fix de sortida ("max_hold_candles")

**Explorat:** Sortir de posicions després de N candles independentment del context.

**Rebutjat perquè:** Si el trade està en negatiu però les portes segueixen obertes (les condicions d'entrada segueixen vigents), forçar sortida realitza una pèrdua innecessària. Si el preu després rebota, has perdut diners i oportunitat. Les sortides condicionals (invalidació de règim, desacceleració de momentum, porta tancada) cobreixen tots els escenaris sense timer arbitrari. L'únic cas especial: trade en negatiu + portes obertes + molt de temps → reducció al 50% (no sortida completa).

### Paràmetres fixos per a trailing stop, R:R mínim, candles desacceleració, etc.

**Explorat:** Valors únics per a tot el sistema (ex: trailing sempre a 2× ATR).

**Rebutjat perquè:** El comportament del mercat canvia segons el règim. En tendència forta, un trailing de 1.5× ATR seria massa ajustat (sortiries per soroll). En mercat calmat, 2.5× ATR seria massa ample (no protegiria profit). La solució correcta: lookup tables basades en el règim de P1 o en el percentil de volatilitat. No requereix ML perquè el règim (que sí és ML) ja captura la informació necessària.

### ML per a Porta 3 (estructura de preu)

**Explorat:** Usar ML per detectar suports/resistències.

**Rebutjat perquè:** Fractals, Fibonacci i Volume Profile són algorismes deterministes suficients. No tenen paràmetres entrenables, ergo no poden sobreajustar. Un model ML per S/R tindria el problema d'etiquetatge: com defines "un bon suport" per entrenar el model? Acabes tornant als mateixos mètodes deterministes per generar etiquetes, fent el ML redundant.

### ML per a Porta 4 (momentum)

**Explorat:** Substituir derivades + RSI + MACD per un classificador entrenat.

**Rebutjat perquè:** Les derivades, RSI i MACD són càlculs directes de l'estat del momentum, no prediccions. Substituir-los per un model entrenat convertiria un mesurament objectiu en una predicció subjectiva — exactament el que l'evidència diu que falla per a dades de preu.

### GMM sol per a classificació de règim (sense HMM)

**Explorat:** GMM (Gaussian Mixture Model) per soft clustering de règims.

**Rebutjat com a únic model perquè:** El GMM no té memòria temporal — pot canviar d'estat a cada candle, generant whipsaws. L'HMM penalitza canvis freqüents via la matriu de transició, produint estats més estables. Tanmateix, el GMM podria ser una alternativa a testar en la fase de validació.

### XGBoost sol per a classificació de règim (sense HMM)

**Explorat:** Entrenar XGBoost amb etiquetes supervisades definides manualment (ex: retorn futur > 0 = bullish).

**Rebutjat perquè:** Les etiquetes manuals són arbitràries (per què 5 dies de lookforward i no 10?) i introdueixen look-ahead bias subtil. L'HMM genera etiquetes objectives des de l'estructura estadística de les dades, no des de retorns futurs.

### Porta 2 afectant posicions obertes

**Explorat:** Si el multiplier baixa, reduir posicions obertes proporcionalment.

**Rebutjat perquè:** Generaria scale-outs freqüents per fluctuacions de sentiment, incrementant costos de transacció i complexitat. Les posicions obertes es gestionen per la Porta 5 (trailing stop, invalidació). L'única excepció es manté: multiplier = 0.0 → sortida d'emergència.

### Timeframe 1H per a derivades i momentum

**Explorat:** Calcular derivades al timeframe 1H per més granularitat.

**Rebutjat perquè:** Per swings de 3 dies, al 1H tens 72 candles amb massa soroll. Les derivades canviarien de signe múltiples vegades dins el mateix swing, generant whipsaws. Al 4H, 18 candles per swing — suficient per un arc de momentum. El MACD(12,26,9) al 4H captura cicles de ~4 dies (coherent). Al 1H, ~26 hores (massa curt).

### Usar models ML existents (XGBoost, GRU, etc.) dins les Portes 3 o 4

**Explorat:** Integrar els models ML del projecte com a components de P3 o P4.

**Rebutjat dins el Gate System perquè:** Aquests models prediuen preu/direcció, que l'evidència demostra que és equivalent a soroll en context univariant. Poden servir com a: (a) senyal addicional a P4 (una confirmació extra), però no com a component principal, o (b) sistema alternatiu complet per comparar via walk-forward amb el Gate System.

---

---

## 14. Pla d'Implementació

> **Decisions d'arquitectura incorporades** (veure §14.10 per al resum complet):
> - **Long-only en v1**: el GateBot compra BTC en règims alcistes i es queda en cash en bear/uncertain. Shorts en v2.
> - **P2 sense news sentiment**: Fear & Greed + on-chain cobreixen el cas. News en v2.
> - **14 features per P1**: `exchange_netflow_7d` exclosa (requereix API de pagament). S'entrenarà amb les 14 restants.
> - **Sense prefix als noms de feature**: cada timeframe té el seu df separat; el GateBot selecciona per nom i no hi ha ambigüitat.

### 14.1 Visió d'Arquitectura — Com encaixa amb el projecte existent

El GateBot segueix exactament el patró `BaseBot + ObservationSchema + from_config()` i s'integra al demo_runner com qualsevol altre bot. El demo_runner el tracta com una caixa negra: crida `observation_schema()` per saber quines dades necessita, crida `on_observation()` en cada candle del timeframe del bot, i rep un `Signal`.

**Dos canvis menors al core (backward-compatible, cap bot existent es veu afectat):**

1. `data/observation/builder.py` → `ObservationBuilder.build()` passa d'indexació per número de fila a indexació per timestamp. Per a cada timeframe secundari, troba la seva fila correcta via `searchsorted(timestamp)`. Per als bots actuals amb un sol timeframe, el comportament és idèntic.

2. `core/engine/demo_runner.py` → `_process_tick()` usa el timeframe primari del bot (primer element de `schema.timeframes`) per calcular el bucket de detecció de candle. Tots els bots actuals declaren `timeframes=["1h"]` → comportament idèntic. El GateBot declararia `timeframes=["4h", "1d"]` → `on_observation()` cridat cada 4H.

Sense aquests dos canvis, el GateBot hauria de gestionar les dades internament, trencant la consistència amb el FeatureBuilder compartit que garanteix que tots els bots veuen les mateixes dades de la BD.

---

### 14.2 Estructura de Fitxers (nous, cap modificació d'existents)

```
bots/gate/
├── __init__.py
├── gate_bot.py                    # GateBot principal (BaseBot)
└── gates/
    ├── __init__.py
    ├── p1_regime.py               # Porta 1: càrrega i inferència HMM+XGBoost
    ├── p2_health.py               # Porta 2: multiplicador qualitat (FG + on-chain)
    ├── p3_structure.py            # Porta 3: nivells de preu (determinista)
    ├── p4_momentum.py             # Porta 4: trigger momentum (determinista)
    └── p5_risk.py                 # Porta 5: Kelly sizing + gestió de posicions

bots/gate/regime_models/
├── __init__.py
├── hmm_trainer.py                 # HMM K=2..6, selecció per BIC, labels Viterbi
└── xgb_classifier.py             # XGBoost multiclasse + walk-forward + Optuna

config/models/gate.yaml            # Config unificada del GateBot
scripts/train_gate_regime.py       # Pipeline complet: dades → HMM → XGBoost → save
```

Fitxers del core que es modifiquen (canvis menors):
- `data/observation/builder.py` (indexació per timestamp)
- `core/engine/demo_runner.py` (bucket per timeframe de bot)
- `config/demo.yaml` (afegir entrada del GateBot, `enabled: false` inicialment)
- `core/db/demo_repository.py` + nova migració Alembic (taula `gate_positions`)

---

### 14.3 Canvi Core #1 — ObservationBuilder: indexació per timestamp

**Problema:** `build(schema, symbol, index)` usa el mateix número de fila per a tots els timeframes. Un `index = len(4h_df) - 1 ≈ 12000` aplicat a la daily df (≈3000 files) resulta en out-of-bounds.

**Solució (30 línies, backward-compatible):**

```python
# data/observation/builder.py → build()
def build(self, schema, symbol, index):
    # 1. Extreu timestamp del timeframe primari (com avui per a bots d'1 TF)
    primary_tf = schema.timeframes[0]
    primary_df = self._cache[f"{symbol}_{primary_tf}"]
    timestamp  = primary_df.index[index]

    for timeframe in schema.timeframes:
        df = self._cache[f"{symbol}_{timeframe}"]
        # 2. Per cada TF, troba l'índex correcte via timestamp
        tf_idx = df.index.searchsorted(timestamp, side="right") - 1
        if tf_idx < schema.lookback:
            raise ValueError(f"Lookback insuficient [{timeframe}]: tf_idx={tf_idx}")
        window = df.iloc[tf_idx - schema.lookback : tf_idx]
        observation[timeframe] = {
            "features":      window[schema.features].copy(),
            "current_price": float(df.iloc[tf_idx]["close"]),
            "timestamp":     df.index[tf_idx],
        }
```

Per a bots amb un sol timeframe, `primary_tf = timeframes[0]`, i `searchsorted(timestamp)` retorna exactament `index` — idèntic a l'actual.

---

### 14.4 Canvi Core #2 — demo_runner: bucket adaptable al timeframe

**Problema:** `is_new_candle` compara buckets de 1 hora per a tots els bots. El GateBot necessita detecció cada 4H.

**Solució (canvi localitzat a `_process_tick`, ~10 línies):**

```python
# core/engine/demo_runner.py — afegir mapa de timeframe → segons
_TF_SECONDS = {"1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200, "1d": 86400}

# Dins el loop de bots, substituir la detecció de candle:
primary_tf    = schema.timeframes[0]
tf_secs       = _TF_SECONDS.get(primary_tf, 3600)
candle_bucket = int(datetime.now(timezone.utc).timestamp()) // tf_secs
is_new_candle = (candle_bucket != self._last_candle_bucket.get(bot_id, -1))
if is_new_candle:
    self._last_candle_bucket[bot_id] = candle_bucket
    df    = builder.get_dataframe(symbol=self.symbol, timeframe=primary_tf)
    index = len(df) - 1
    ...
```

`_last_candle_hour` (existent) s'actualitza a `_last_candle_bucket` per reflectir que funciona per qualsevol granularitat. Per a tots els bots actuals `primary_tf="1h"` → `tf_secs=3600` → comportament idèntic.

---

### 14.5 YAML de configuració: `config/models/gate.yaml` (estructura clau)

```yaml
category: gate          # evita que discover_configs("ML") o ("RL") l'agafi
model_type: gate
module: bots.gate.gate_bot
class_name: GateBot
bot_id: gate_v1
symbol: BTC/USDT
timeframe: 4h           # timeframe primari de decisions (on_observation crida cada 4H)
aux_timeframe: 1d       # timeframe secundari per P1/P2 (re-evaluat quan tanca nova candle diària)

lookback: 300           # 300×4H = 50 dies  /  300×1D = ~1 any → cobreix EMA200

model_paths:
  hmm:  models/gate_hmm.pkl
  xgb:  models/gate_xgb_regime.pkl

# Features per cada timeframe (noms sense prefix — cada df és independent)
# L'ObservationSchema declara la UNIÓ (deduplicated); el GateBot accedeix
# obs["4h"]["features"][features_4h] i obs["1d"]["features"][features_1d] per separat.
features_4h:
  - close
  - high
  - low
  - volume
  - atr_14
  - rsi_14
  - macd
  - macd_signal
  - funding_rate

features_1d:
  - close
  - ema_50
  - ema_200
  - adx_14
  - rsi_14
  - atr_14
  - fear_greed_value
  - funding_rate
  - volume

# Dades externes (Fear & Greed + funding rate — sense netflow, no disponible sense API de pagament)
external:
  fear_greed: true
  funding_rate: true
  # blockchain: []   # exchange-netflow exclòs en v1 (requereix CryptoQuant/Glassnode)

# ── Porta 1 ─────────────────────────────────────────────────────────────────
p1:
  min_regime_confidence: 0.60   # per sota → UNCERTAIN, porta tancada

# ── Porta 2 ─────────────────────────────────────────────────────────────────
# v1: Fear & Greed + on-chain flows. News sentiment afegit en v2.
p2:
  onchain_enabled: true         # ratio outflow/inflow d'exchanges (de funding_rate proxy)

# ── Porta 3 ─────────────────────────────────────────────────────────────────
p3:
  min_level_strength: 0.4
  fractal_n_4h: 2
  fractal_n_1d: 5
  min_swing_pct: 0.03           # oscil·lació mínima per Fibonacci (3%)

# ── Porta 4 ─────────────────────────────────────────────────────────────────
p4:
  ewm_span: 3
  rsi2_oversold: 10

# ── Porta 5 ─────────────────────────────────────────────────────────────────
# v1 long-only: STRONG_BULL i WEAK_BULL → entrades llargues.
# RANGING → entrades amb restriccions. BEAR/UNCERTAIN → cap nova entrada.
p5:
  max_risk_pct: 0.01            # 1% capital en risc per trade
  max_exposure_pct: 0.95        # cap dur d'exposició total
  max_open_positions: 2
  weekly_drawdown_limit: 0.05   # 5% setmanal → veto

  # R:R mínim. En v1 (long-only) no hi ha entrades en BEAR.
  min_rr:
    STRONG_BULL: 1.5
    WEAK_BULL:   2.0
    RANGING:     2.0

  # Trailing stop: multiplicador ATR per percentil de volatilitat
  trailing_atr_multiplier:
    low_vol:    1.5             # percentil ATR < 30
    normal_vol: 2.0             # percentil ATR 30-70
    high_vol:   2.5             # percentil ATR > 70

  # Candles 4H consecutives amb d2 negatiu → sortida per desacceleració
  decel_exit_candles:
    STRONG_BULL: 5
    WEAK_BULL:   3
    RANGING:     2

  stagnation_days: 6            # dies màxims en negatiu + portes obertes → reduir 50%

# best_params s'omple automàticament per train_gate_regime.py (Optuna)
```

---

### 14.6 ObservationSchema del GateBot

No calen sufixos als noms de les features. L'`ObservationBuilder` carrega cada timeframe en un df separat; el GateBot accedeix `obs["4h"]` i `obs["1d"]` per separat, de manera que `close` en `obs["4h"]` és sempre el close 4H, i `close` en `obs["1d"]` és sempre el close diari. No hi ha cap ambigüitat.

L'`ObservationSchema` declara la **unió deduplificada** de totes les features necessàries. El GateBot internament selecciona les columnes rellevants per a cada porta.

```python
def observation_schema(self) -> ObservationSchema:
    features_4h = self.config["features_4h"]
    features_1d = self.config["features_1d"]
    # Unió deduplificada (preservant ordre): l'ObservationBuilder la passa a cada df
    all_features = list(dict.fromkeys(features_4h + features_1d))
    return ObservationSchema(
        features   = all_features,
        timeframes = ["4h", "1d"],
        lookback   = self.config["lookback"],   # 300 tant per 4H com per 1D
        extras     = {"external": self.config["external"]},
    )
```

A `on_observation()`, el GateBot selecciona les columnes de cada timeframe:
```python
feat_4h = obs["4h"]["features"][self.config["features_4h"]]  # 9 columnes 4H
feat_1d = obs["1d"]["features"][self.config["features_1d"]]  # 9 columnes 1D
```

---

### 14.7 Pipeline d'entrenament: `scripts/train_gate_regime.py`

```
Pas 1  Carregar OHLCV diari + Fear & Greed + funding rate (via FeatureBuilder 1d)
         [exchange_netflow exclòs en v1 — no hi ha font de dades sense API de pagament]

Pas 2  Calcular les 14 features de P1 des del df diari:
         ema_ratio_50_200   = close / ema_200 * ema_50 / close  (posició relativa en tendència)
         ema_slope_50       = pendent lineal EMA50 últims 10 dies
         ema_slope_200      = pendent lineal EMA200 últims 20 dies
         adx_14             = directament del df (ja el calcula FeatureBuilder)
         atr_14_percentile  = percentil(atr_14, window=252)  [règim de volatilitat relatiu]
         funding_rate_3d    = rolling(funding_rate, 3).mean()
         funding_rate_7d    = rolling(funding_rate, 7).mean()
         volume_sma_ratio   = volume / rolling(volume, 20).mean()
         volume_trend       = pendent regressió lineal del volum (20 dies)
         fear_greed         = fear_greed_value (directament del df)
         fear_greed_7d_change = fear_greed_value - fear_greed_value.shift(7)
         rsi_14_daily       = rsi_14 (directament del df)
         returns_5d         = close.pct_change(5)
         returns_20d        = close.pct_change(20)

Pas 3  HMM — entrenar K=2..6 (hmmlearn.GaussianHMM, emissions gaussianes multivariades)
         Observacions HMM (3 variables): [retorns_diaris, ATR_14_normalitzat, volum_sma_ratio]
         Selecció K: BIC = -2×log-verosimilitud + nparams×log(T)
           nparams(K) = K×(K-1)  [transicions]
                      + K×n_obs  [means]
                      + K×n_obs×(n_obs+1)/2  [covariances]
         Seleccionar K amb BIC mínim. Executar 10 inits aleatòries per K; quedar-se amb
         el millor (màxima log-verosimilitud) per estabilitat.
         Decodificació Viterbi → seqüència d'estats [0, 2, 1, 0, 3, ...]

Pas 4  Mapatge HMM states → RegimeState
         Per cada estat calcular: mean_return, std_return (vol), mean_adx, mean_atr_pct
         Ordenació per mean_return: estat amb major return → STRONG_BULL, menor → STRONG_BEAR
         Refinament: si dos estats adjacents en return tenen ADX similar, distingir per vol
         Si K=4: STRONG_BULL, WEAK_BULL, RANGING/WEAK_BEAR, STRONG_BEAR
         Si K=3: BULL, RANGING, BEAR (WEAK variants absents)
         Si K=5 o 6: mapeig complet als 5 estats documentats + UNCERTAIN residual
         Desar el mapeig {hmm_state_id → RegimeState} dins gate_hmm.pkl

Pas 5  Walk-forward XGBoost (5 folds, train ≥ 2 anys, test 3-6 mesos)
         Optuna TPE + Hyperband pruner
         Objectiu: minimitzar mlogloss MITJANA dels 5 folds (no el millor fold)
         Early stopping per fold (early_stopping_rounds=50)

Pas 6  Validació: accuracy OOS > 55% en TOTS els folds; desviació mlogloss < 15%
         Si no es compleix: logar avís, no fallida dura (el model pot ser acceptable)

Pas 7  Guardar models/gate_hmm.pkl + models/gate_xgb_regime.pkl
         Actualitzar secció best_params a config/models/gate.yaml
```

**Dependències noves necessàries**: `hmmlearn`. La resta (`xgboost`, `optuna`, `scikit-learn`, `scipy`) ja estan al projecte.

---

### 14.8 Gestió de posicions obertes i persistència d'estat

El GateBot manté `self._open_positions: list[dict]` en memòria. Cada posició és:
```python
{
  "entry_price":  float,    # preu d'entrada
  "stop_level":   float,    # stop loss actual (es mou amb trailing)
  "target_level": float,    # objectiu de profit (P3)
  "highest_price": float,   # màxim des d'entrada (per trailing)
  "size_usdt":    float,    # mida en USDT
  "regime":       str,      # règim en el moment d'entrada
  "opened_at":    datetime,
  "decel_counter": int,     # candles amb d2 negatiu consecutiu
}
```

Per a persistència entre reinicis, cal una nova taula `gate_positions` a la BD. La migració Alembic és minor (afegir una taula nova, sense tocar cap taula existent). `DemoRepository` s'estén amb `save_gate_position()`, `update_gate_position()`, `get_open_gate_positions(bot_id)`.

A `_load_bots()` (demo_runner), per al GateBot, restaurar `_open_positions` des de `repo.get_open_gate_positions(bot_id)` (si és un GateBot, detectat pel camp `model_type: gate`).

---

### 14.9 Seqüència d'implementació

```
1. Core canvis (builder.py + demo_runner.py)               → ~50 línies
2. Migració Alembic + demo_repository extensions            → ~40 línies
3. P3 — estructura de preu (fractals, Fibonacci, VP)        → determinista
4. P4 — momentum (EWM derivatives, RSI-2, MACD cross)      → determinista
5. P5 — risk/sizing (Kelly, trailing stop, exit conditions)  → fórmules
6. HMM trainer (hmm_trainer.py)                             → hmmlearn wrapper
7. XGBoost classifier (xgb_classifier.py)                   → segueix patró BaseTreeModel
8. P1 — regime evaluator (carrega models, inferència)       → wrapper
9. P2 — health multiplier (FG + on-chain, lookup tables)    → calcul directe
10. GateBot (gate_bot.py) — integra P1..P5                  → orquestra les portes
11. scripts/train_gate_regime.py                            → pipeline complet
12. config/models/gate.yaml                                 → config final
13. Entrenar: python scripts/train_gate_regime.py
14. Backtest: python scripts/run_comparison.py (GateBot vs Hold vs Trend)
15. Si criteris complerts (Sharpe>0.5, DD<15%): enabled: true a demo.yaml
```

---

### 14.10 Decisions Adoptades

Les quatre decisions arquitectòniques estan confirmades i incorporades a totes les seccions anteriors d'aquest pla (§14.1–§14.9).

**Decisió A — Long-only en v1** ✅
`PaperExchange` no suporta shorts (no té lògica de marge). El GateBot v1 operarà exclusivament en llarg:
- `STRONG_BULL` / `WEAK_BULL` → obertura de posicions llargues (mida escalada per `position_multiplier`).
- `BEAR` / `UNCERTAIN` / `STRONG_BEAR` / `WEAK_BEAR` → sense noves entrades. Si hi ha posició oberta, P5 gestiona la sortida.
- Shorts diferits a v2 si els resultats de la v1 ho justifiquen.

**Decisió B — P2 sense news sentiment en v1** ✅
El component de news (CryptoPanic / FinBERT / Claude API) queda fora de l'abast de v1. P2 calcularà el `position_multiplier` exclusivament amb:
- Fear & Greed Index (`fear_greed_value`, `fear_greed_class`)
- Mètriques on-chain de `blockchain.com` (hash-rate, adreces úniques)
- Funding rate
News sentiment es pot afegir com a feature addicional en v2 un cop el sistema base estigui validat.

**Decisió C — 14 features per P1 (sense exchange_netflow_7d)** ✅
`exchange_netflow_7d` requereix CryptoQuant o Glassnode (API de pagament). S'exclou de `features_1d`. El model de règim P1 s'entrena amb les **14 features** restants:

| # | Feature | Font |
|---|---------|------|
| 1–5 | `close`, `volume`, `atr_14`, `rsi_14`, `macd` | Tècniques 1D |
| 6–7 | `bb_upper`, `bb_lower` | Bollinger Bands 1D |
| 8–9 | `ema_20`, `ema_50` | EMAs 1D |
| 10 | `adx_14` | Força de tendència 1D |
| 11–12 | `fear_greed_value`, `funding_rate` | Externes |
| 13–14 | `hash_rate`, `n_unique_addresses` | On-chain |

**Decisió D — Sense prefix; union-deduplication per nom** ✅
`ObservationSchema.features` rep la unió de `features_4h + features_1d` sense duplicats (Python `dict.fromkeys`). Com que cada df es passa per separat via `obs["4h"]` i `obs["1d"]`, no hi ha ambigüitat: `GateBot.on_observation()` accedeix als DataFrames per clau de timeframe i selecciona les columnes per nom directament. No cal cap prefix (`_4h`, `_1d`) ni canviar la interfície de `BaseBot`.

---

## 13. Referències

- Lahmiri & Bekiros (2025). Quantifying Cryptocurrency Unpredictability. arXiv:2502.09079.
- Gupta et al. (2025). Multi-model ensemble-HMM voting framework. DSFE, 5(4).
- López de Prado (2018). Advances in Financial Machine Learning. Wiley.
- Kelly (1956). A new interpretation of information rate. Bell System Technical Journal.
- Connors & Alvarez (2009). Short-Term Trading Strategies That Work.
- Nguyen (2018). Hidden Markov Model for Stock Trading. IJFS, 6(2).
- Carver (2015). Systematic Trading. Harriman House.
- LuxAlgo (2025). ATR-based stop-losses reduce max drawdown 32% vs fixed stops.
- Two Sigma. Market Regime Detection Using Unsupervised Learning.
- Pohle et al. (2017). Selecting the Number of States in HMMs. arXiv:1701.08673.
- XGBoost Documentation. Notes on Parameter Tuning. xgboost.readthedocs.io.
