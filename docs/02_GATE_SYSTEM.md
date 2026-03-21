# Gate System — Documentació Completa

> Document definitiu del Gate System: disseny, implementació, entrenament i operació.
> Per al context del projecte: **[01_ARCHITECTURE.md](./01_ARCHITECTURE.md)**.
> Per a un trade pas a pas: **[examples/trade_walkthrough.md](./examples/trade_walkthrough.md)**.

---

## Principi fonamental

> **El sistema no prediu preus. Avalua context.**
>
> La pregunta que respon: *"Ara, les condicions són favorables per a un swing trade? A quins nivells? Amb quina confiança?"*
>
> Fonament empíric: les sèries temporals de crypto són estadísticament indistingibles de soroll brownià en context univariant — models Naive superen consistentment LSTM, GRU, XGBoost per **predicció de preus** (Lahmiri & Bekiros, 2025). Però el ~10% de traders profitables exploten **asimetria de risc condicional al context**, no predicció de preus.

---

## La idea en 30 segons

Imagina un trader professional que, abans d'obrir una posició, es fa 5 preguntes per ordre:

1. **"Com està el mercat?"** (P1 — Règim) → Si estem en bear market, no compro.
2. **"El mercat està sa?"** (P2 — Salut) → Si hi ha eufòria o pànic extrem, redueixo la mida.
3. **"Hi ha un bon preu?"** (P3 — Estructura) → Si no hi ha un nivell de suport clar a prop, no compro.
4. **"Hi ha momentum ara?"** (P4 — Momentum) → Si el preu no es mou en la meva direcció, espero.
5. **"Quant arriscar?"** (P5 — Risc) → Calculo la mida exacta i verifico els límits.

Cada pregunta és una "porta" (gate). Si qualsevol porta diu NO, no hi ha trade. Punt. Això fa que el bot sigui molt selectiu (~10-15 trades/mes) però cada trade ha passat 5 filtres rigorosos.

---

## Flux visual complet

```
                          CADA 4 HORES (candle 4H nova)
                                    │
                    ┌───────────────┴───────────────┐
                    │   Hi ha posicions obertes?     │
                    └───────┬───────────────┬───────┘
                      SÍ    │               │   NO
                            ▼               │
               P5 gestiona posicions:       │
               • Actualitza trailing stop   │
               • Comprova sortides          │
               • Si cal, retorna SELL       │
                            │               │
                            ▼               ▼
              ┌──────────────────────────────────────┐
              │  P1 — RÈGIM MACRO (avalua 1×/dia)    │
              │  "En quin tipus de mercat estem?"     │
              │                                      │
              │  HMM offline → etiquetes de règim    │
              │  XGBoost online → prediu règim avui  │
              │                                      │
              │  ✓ STRONG_BULL / WEAK_BULL / RANGING │──→ Continua
              │  ✗ STRONG_BEAR / WEAK_BEAR / UNCERT. │──→ HOLD
              └───────────────┬──────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  P2 — SALUT DEL MERCAT (1×/dia)      │
              │  "El mercat està sa o sobreescalfat?" │
              │                                      │
              │  Fear & Greed × Funding Rate         │
              │  → multiplier [0.0 – 1.0]            │
              │                                      │
              │  ✓ multiplier > 0 → Continua         │──→ Continua
              │  ✗ multiplier = 0 → HOLD (emergència)│──→ HOLD
              └───────────────┬──────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  P3 — ESTRUCTURA DE PREU (cada 4H)   │
              │  "Hi ha un bon nivell on comprar?"    │
              │                                      │
              │  Pivots fractals + Fibonacci          │
              │  + Volume Profile (HVN)              │
              │  → nivells de suport/resistència      │
              │  Filtre: volum d'acostament ≥ 0.8    │
              │                                      │
              │  ✓ has_actionable_level → Continua   │──→ Continua
              │  ✗ No hi ha nivell → HOLD             │──→ HOLD
              └───────────────┬──────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  NEAR-MISS LOG    │ ← P1+P2+P3 han passat.
                    │  Registra sempre  │   Sigui quin sigui el resultat
                    │  per anàlisi      │   de P4 i P5, es registra aquí
                    └─────────┬─────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  P4 — MOMENTUM TRIGGER (cada 4H)     │
              │  "El preu es mou a favor?"            │
              │                                      │
              │  4 senyals: derivades d1/d2, RSI-2,  │
              │  MACD cross → mínim adaptat al règim │
              │  STRONG_BULL: 1 senyal               │
              │  WEAK_BULL / RANGING: 2 senyals      │
              │                                      │
              │  ✓ Prou senyals → Continua           │──→ Continua
              │  ✗ Massa pocs → HOLD (near-miss log) │──→ HOLD
              └───────────────┬──────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  P5 — RISC I SIZING (cada 4H)        │
              │  "Puc entrar? Amb quanta pasta?"      │
              │                                      │
              │  Kelly fraccionari → mida exacta     │
              │  VETOs: massa posicions, drawdown,   │
              │         R:R insuficient              │
              │                                      │
              │  ✓ No hi ha VETO → BUY              │──→ BUY
              │  ✗ VETO → HOLD (near-miss log)      │──→ HOLD
              └──────────────────────────────────────┘
```

---

## P1 — Règim Macro

**Fitxers:** `bots/gate/regime_models/hmm_trainer.py` · `bots/gate/regime_models/xgb_classifier.py` · `bots/gate/gates/p1_regime.py`

**Problema:** No vols comprar en un bear market. P1 classifica el mercat en 6 estats per evitar operar a contracorrent.

### Model en cascada: HMM → XGBoost

**Pas 1 — HMM (offline, mensual)**

Un Hidden Markov Model descobreix "règims" ocults a les dades diàries sense etiquetes predefinides. El procés:

- Prova K = 2, 3, 4, 5, 6 estats. Selecciona el K amb BIC (Bayesian Information Criterion) mínim — penalitza models massa complexos.
- Per cada K, fa 10 inicialitzacions aleatòries (l'HMM depèn del punt de partida) i es queda la millor per log-likelihood.
- Observacions: `[daily_return, normalized_atr14, volume_sma_ratio]`
- Decodificació Viterbi → seqüència d'estats per cada dia

**Pas 2 — XGBoost (online, diari)**

Entrena un classificador supervisat sobre les etiquetes de l'HMM, usant 14 features observables en temps real. Validació walk-forward 5 folds. Optuna TPE + Hyperband per mlogloss.

**Per què 2 models?** L'HMM necessita tota la sèrie temporal (no pot predir online). XGBoost pot predir en temps real, però necessita etiquetes per entrenar → les agafa de l'HMM. La cascada combina el millor dels dos mons.

### Les 14 features de P1

| Feature | Descripció | Font |
|---------|-----------|------|
| `ema_ratio_50_200` | close / EMA_200, normalitzat per EMA_50 | Posició relativa en tendència |
| `ema_slope_50` | Pendent EMA_50 (últims 10 dies) | Direcció tendència mig termini |
| `ema_slope_200` | Pendent EMA_200 (últims 20 dies) | Direcció tendència llarg termini |
| `adx_14` | Average Directional Index | Força de la tendència (no la direcció) |
| `atr_14_percentile` | Percentil ATR actual vs últims 252 dies | Règim de volatilitat relatiu |
| `funding_rate_3d` | Mitjana funding rate últims 3 dies | Posicionament mercat apalancat |
| `funding_rate_7d` | Mitjana funding rate últims 7 dies | Tendència del posicionament |
| `volume_sma_ratio` | Volum / SMA(20) del volum | Activitat relativa |
| `volume_trend` | Pendent regressió lineal volum (20 dies) | Tendència d'activitat |
| `fear_greed` | Fear & Greed Index actual | Sentiment agregat |
| `fear_greed_7d_change` | Canvi del FG en 7 dies | Tendència del sentiment |
| `rsi_14_daily` | RSI-14 al diari | Momentum a llarg termini |
| `returns_5d` | Retorn acumulat 5 dies | Momentum recent |
| `returns_20d` | Retorn acumulat 20 dies | Momentum mig termini |

> `exchange_netflow_7d` **no inclòs**: requereix CryptoQuant/Glassnode (APIs de pagament). Tots els 14 features actuals provenen de fonts gratuïtes (Binance + alternative.me). Decisió: `08_DECISIONS.md §11-C`.

### 6 règims possibles

| Règim | Descripció | Permet entrades? | Invalida posicions? |
|-------|-----------|:---:|:---:|
| STRONG_BULL | Tendència alcista consolidada | Sí | — |
| WEAK_BULL | Alcista però perdent força | Sí | — |
| RANGING | Mercat lateral, sense direcció | Sí (R:R més alt) | — |
| UNCERTAIN | Confiança < 0.60 en qualsevol règim | No | Sí |
| WEAK_BEAR | Baixista però perdent força | No | Sí |
| STRONG_BEAR | Tendència baixista consolidada | No | Sí |

```yaml
p1:
  min_regime_confidence: 0.60
  # Si cap règim assoleix 0.60 de probabilitat → UNCERTAIN → porta tancada.
  # Protegeix contra mercats sense tendència clara on l'HMM oscil·la.
```

---

## P2 — Salut Qualitativa del Mercat

**Fitxer:** `bots/gate/gates/p2_health.py`

**Problema:** Fins i tot en bull market, hi ha moments sobreescalfats o massa apalancats. P2 detecta aquests moments i redueix la mida de la posició (o la veta totalment).

### Com funciona

Calcula un `position_multiplier ∈ [0.0, 1.0]` com a **producte** de dues puntuacions independents. Si qualsevol és 0, el multiplicador final és 0 (veto total + sortida d'emergència de posicions obertes).

**Sub-score 1 — Fear & Greed Index (contrarian)**

| Rang F&G | Significat | Score en Bull | Score en Bear |
|:---:|---|:---:|:---:|
| > 75 | Cobdícia extrema | 0.5 | 0.9 |
| 60–75 | Cobdícia | 0.6 | 0.8 |
| 40–60 | Neutral | 0.8 | 0.7 |
| 25–40 | Por | 0.9 | 0.6 |
| < 25 | Por extrema | 1.0 | 0.5 |

*Racional:* En bull market, la cobdícia extrema indica que l'acumulació és sobreestesa (score baix). La por extrema = oportunitat (score alt). Lògica contrarian.

**Sub-score 2 — Funding Rate**

| Funding Rate (8h) | Score | Per què |
|:---:|:---:|---|
| > 0.03% | 0.5 | Mercat massa apalancat long → risc de liquidacions en cascada |
| 0 – 0.03% | 0.8 | Normal |
| < 0% | 1.0 | Desapalancat → favorable per a llargs |

**Exemple:** F&G = 82 en bull (0.5) × FR = 0.04% (0.5) = **0.25** → posició un 25% del que seria normalment.

```yaml
p2:
  onchain_enabled: true   # activa el funding rate score
  # News sentiment (LLM): no implementat v1. Previst per v2. (DECISIONS §11-B)
```

---

## P3 — Estructura de Preu

**Fitxer:** `bots/gate/gates/p3_structure.py`

**Problema:** No vols comprar "al mig del no-res". P3 busca zones on el mercat ha reaccionat històricament i verifica que el preu actual n'és a prop.

### 3 mètodes deterministes (cap ML)

**1. Pivots fractals**

Detecta swing highs (resistència) i swing lows (suport). Un pivot amb N=2 vol dir que el high/low és el màxim/mínim de les 2 candles adjacents de cada banda.

```
    N=2, pivot high a l'índex i:
    high[i] > max(high[i-2], high[i-1], high[i+1], high[i+2])
```

Usa N=2 al 4H (sensible, pivots curts) i N=5 al 1D (robust, pivots importants).

**2. Fibonacci retracement**

Dins l'últim swing significatiu (≥3% de moviment al 1D), calcula els nivells 38.2%, 50%, 61.8% i 78.6%. Zones on estadísticament el preu sol reaccionar.

**3. Volume Profile (HVN)**

Divideix el rang de preus en 50 bins i acumula el volum a cada un. Els bins amb volum > 1.5× la mitjana són "High Volume Nodes": zones on el mercat ha transaccionat molt. Actuen com a imants o barreres en futures visites.

### Scoring i merge

Nivells de les 3 fonts es consoliden: si dos estan a <0.5% de distància, es fusionen. La força resultant:

| Fonts coincidents | Força base | Exemple |
|:---:|:---:|---|
| 1 | 0.3 | Pivot fractal |
| 2 | 0.6 | Fractal + Fibonacci al mateix preu |
| 3 | 0.9 | Fractal + Fibonacci + HVN |

Cada toc anterior sense trencar suma +0.1 (cap: 1.0).

### Proximity threshold adaptatiu

| Força del nivell | Zona de proximitat |
|:---:|:---:|
| ≥ 0.7 (fort) | 2.0 × ATR |
| 0.4 – 0.7 (moderat) | 1.5 × ATR |
| < 0.4 (filtrat) | no passa el filtre de força mínima |

### Filtre de volum d'acostament

```python
vol_approach = df_4h["volume"].iloc[-3:].mean()   # últimes 3 candles 4H = 12h
vol_ma       = df_4h["volume"].rolling(20).mean().iloc[-1]
volume_ratio = vol_approach / vol_ma
```

Si `volume_ratio < 0.8` → el mercat s'apropa al nivell sense convicció → `has_actionable_level = False`, independentment de la força del nivell.

*Per què a P3 i no a P4?* El volum a P3 actua com a filtre independent **abans** que P4 s'activi, impedint que el momentum s'avaluï en moviments sense interès real. Si s'afegís a P4 com un senyal addicional, el volum alt i els senyals de preu tendirien a coocórrer per construcció (un moviment gran implica volum gran), creant correlació interna.

```yaml
p3:
  min_level_strength: 0.4       # 0.4 = 1 font + 1 toc anterior
  fractal_n_4h: 2               # pivots sensibles
  fractal_n_1d: 5               # pivots robustos
  min_swing_pct: 0.03           # swing mínim 3% per calcular Fibonacci
  volume_profile_bins: 50       # ~$1.500–$2.000 per bin (BTC ~$80k)
```

---

## P4 — Momentum Trigger

**Fitxer:** `bots/gate/gates/p4_momentum.py`

**Problema:** P3 ha trobat un bon nivell, però el preu podria estar movent-se en contra. P4 verifica que el momentum actual és favorable.

### 4 senyals (binaris)

Les derivades es calculen sobre el preu suavitzat amb EWM (span=3):
```python
smoothed = close_4h.ewm(span=3, adjust=False).mean()
d1 = smoothed.pct_change()    # velocitat
d2 = d1.diff()                # acceleració
```

| # | Senyal | Condició | En paraules |
|:-:|--------|----------|-------------|
| 1 | **D1 accelerant** | `d1 > 0 AND d2 ≥ 0` | El preu puja i guanya velocitat |
| 2 | **D1 frenant** | `d1 < 0 AND d2 > 0` | El preu baixa però frena (possible gir) |
| 3 | **RSI-2 sobrevenut** | `RSI-2 < 10` | Sobrevenut extrem en 2 períodes (Connors, 2009) |
| 4 | **MACD cross** | MACD(12,26,9) creua senyal cap amunt | Canvi de momentum 4H |

### Mínims adaptatius al règim

| Règim P1 | Senyals mínims | Raonament |
|----------|:-:|---|
| STRONG_BULL | 1 | La tendència ja és clara — n'hi ha prou amb 1 confirmació |
| WEAK_BULL | 2 | Cal més evidència |
| RANGING | 2 | Mercat indecís → exigir confirmació |
| Qualsevol altre | 9 (impossible) | Protecció de seguretat — P1 ho hauria bloquejat |

```yaml
p4:
  ewm_span: 3        # suavitzat derivades (3=agressiu, 5=conservador)
  rsi2_oversold: 10  # llindar RSI-2 (Connors: <10 = sobrevenut extrem)
```

---

## P5 — Risc i Gestió de Posicions

**Fitxer:** `bots/gate/gates/p5_risk.py`

P5 té dues responsabilitats molt diferenciades:

### A) Sizing d'entrada

Usa una variant del **Kelly Criterion** per calcular la mida matemàticament òptima:

```
1.  risc_usdt  = capital × 1% × P2_multiplier × P4_confiança
                   ↑           ↑                 ↑
             risc base    ajust de     ajust per quants
             mai >1%      salut del    senyals P4 actius
                          mercat

2.  dist_stop  = |entrada - stop| / entrada       (en % — definit per P3)

3.  posició    = risc_usdt / dist_stop
                   ↑ stop lluny → posició petita; stop a prop → posició gran

4.  size_final = min(posició / capital, 95%)       (mai >95% del capital)
```

### Condicions de VETO (bloquegen l'entrada)

| Condició | Llindar | Raonament |
|----------|---------|-----------|
| Massa posicions | ≥ 2 obertes | Risc concentrat |
| Drawdown setmanal | > 5% de pèrdua | Protecció contra sèries dolentes |
| R:R insuficient | < 1.5× (BULL), < 2.0× (RANGING) | El trade no justifica el risc |
| Stop invàlid | distància ≤ 0 | Error de càlcul a P3 |

### B) Gestió de posicions obertes (cada 4H)

S'avalua per a cada posició oberta. Prioritat de sortida (de més a menys urgent):

| Prioritat | Condició | Acció | Notes |
|:---------:|----------|-------|-------|
| 1 | P2 = 0 (emergència) | SELL 100% | Mercat en estat extrem |
| 2 | Règim invalida llargs | SELL 100% | Passat a BEAR o UNCERTAIN |
| 3 | Circuit breaker | SELL 100% | Moviment > 3×ATR en 1 candle (crash) |
| 4 | Stop loss tocat | SELL 100% | Preu ≤ stop_level |
| 5 | Trailing stop activat | Actualitza stop | Preu ha pujat ≥ 1×ATR → stop puja |
| 6 | Desacceleració persistent | SELL 100% | d2 < 0 durant N candles consecutives |
| 7 | Estancament | SELL 50% | En pèrdues + P3 tancada + >6 dies obert |

**Trailing stop adaptatiu a la volatilitat:**

```
Activació: quan preu ≥ entrada + 1×ATR

Stop = preu_màxim_vist - (multiplicador × ATR)

Multiplicador depenent del percentil ATR actual:
  < 30 (mercat calm)    → 1.5×  (stop ajustat, protegeix guanys)
  30 – 70 (normal)      → 2.0×  (equilibri)
  > 70 (volàtil)        → 2.5×  (stop ample, evita falsos stop-outs)

El stop mai baixa (only goes up).
```

**Deceleration exit per règim:**

| Règim | Candles 4H consecutives amb d2 < 0 | Temps |
|-------|:-:|---|
| STRONG_BULL | 5 | 20 hores de desacceleració |
| WEAK_BULL | 3 | 12 hores |
| RANGING | 2 | 8 hores (menor tolerància) |

---

## Near-Miss Logger

**Fitxer:** `bots/gate/near_miss_logger.py`

Cada vegada que P1+P2+P3 passen simultàniament (independentment del resultat de P4 i P5), es registra un snapshot complet a la taula `gate_near_misses`.

**Per a l'anàlisi post-demo:**

```sql
-- Quantes oportunitats s'han perdut per porta?
SELECT
    CASE
        WHEN NOT p4_triggered THEN 'P4 blocat'
        WHEN p5_veto_reason IS NOT NULL THEN 'P5: ' || p5_veto_reason
        ELSE 'Executat'
    END AS resultat,
    COUNT(*) as n,
    AVG(p2_multiplier) as p2_avg,
    AVG(p3_risk_reward) as rr_avg
FROM gate_near_misses
GROUP BY 1
ORDER BY n DESC;

-- En quin règim hi ha més oportunitats perdudes?
SELECT p1_regime, COUNT(*) as oportunitats, SUM(executed::int) as executats
FROM gate_near_misses
GROUP BY p1_regime
ORDER BY oportunitats DESC;
```

---

## Pipeline d'entrenament

### Prerequisits

```bash
# 1. Dades actualitzades a la BD
python scripts/download_data.py

# 2. Taules Gate a la BD (una sola vegada)
alembic upgrade head   # crea gate_positions + gate_near_misses
```

### Entrenament

```bash
# Entrenament complet (HMM K=2..6 BIC + XGBoost Optuna walk-forward)
python scripts/train_gate_regime.py

# Opcions útils:
#   --n-trials 100       Optuna: 100 trials (default: 50)
#   --no-optuna          Usar paràmetres XGBoost per defecte (ràpid per testing)
#   --symbol BTC/USDT    Símbol a la BD (default)
```

**Output esperat:**
```
[INFO] HMM training: K=2..6, 10 inits/K
[INFO] BIC mínima obtinguda amb K=4: BIC=1234.5
[INFO] XGBoost walk-forward: 5 folds
[INFO] Fold 1/5: accuracy=0.61, mlogloss=0.87
...
[INFO] Mean accuracy OOS: 0.63 ± 0.04
[INFO] Mean mlogloss: 0.82 ± 0.08
[INFO] Validació passada (accuracy > 0.55, std < 0.15)
[INFO] Guardant models/gate_hmm.pkl i models/gate_xgb_regime.pkl
```

**Temps estimat (M4 Pro):**
- HMM (K=2..6, 10 inits): ~2–5 min
- XGBoost + Optuna 50 trials: ~20–40 min
- **Total: ~30–45 min**

### Verificació

```bash
ls -lh models/gate_hmm.pkl models/gate_xgb_regime.pkl

# Verificar que les dates d'entrenament i test coincideixen
python -c "
import pickle
with open('models/gate_xgb_regime.pkl', 'rb') as f:
    m = pickle.load(f)
print('Règims:', m['label_encoder'].classes_)
print('Features:', len(m['feature_names']))
"
```

### Activació a demo

```yaml
# config/demo.yaml
bots:
  - config_path: config/models/gate.yaml
    enabled: true    # canviar de false a true
```

---

## Decisions de disseny i alternatives descartades

| Decisió | Alternativa descartada | Per què es va descartar |
|---------|----------------------|------------------------|
| Long-only en v1 | Curts des del principi | Risc asimètric en bull market estructural; complexitat addicional d'stop; valida primer la infraestructura. V2 ho afegirà |
| Sense news sentiment en v1 | LLM sentiment a P2 | Requereix API externa de pagament + LLM + cron diari. Afegeix complexitat innecessària per validar la hipòtesi central |
| 14 features (sense exchange_netflow) | Afegir netflow on-chain | CryptoQuant/Glassnode costen $50-$200/mes. Tots els 14 features actuals són gratuïts |
| P3 determinista (sense ML) | ML per classificar nivells | ML per estructura de preu requereix etiquetatge subjectiu, afegeix complexitat sense evidència d'avantatge vs. mètodes deterministes ben parametritzats |
| P4 derivades (4 senyals) | ML per predir momentum | Mateixa raó: el momentum és mesurable, no cal predir-lo; les derivades captura directament l'estat sense model intermig |
| GMM per règims | Usar únicament GMM | GMM no captura dependències temporals entre estats successius; HMM sí. Per mercats financers la transició entre règims és temporal, no independent |
| Volum d'acostament a P3 | Volum a P4 com a senyal addicional | El volum alt i senyals de preu coocorren (correlació); P3 és el lloc correcte per filtrar abans que P4 s'activi |

---

## Configuració completa (`config/models/gate.yaml`)

```yaml
bot_id:     gate_v1
module:     bots.gate.gate_bot
class_name: GateBot
category:   gate

timeframe:      4h     # timeframe primari (decisions)
aux_timeframe:  1d     # timeframe diari (context P1/P2)
lookback:       300    # candles 4H de context

features_4h:
  - open
  - high
  - low
  - close
  - volume
  - ema_9
  - ema_21
  - ema_50
  - rsi_14
  - macd
  - macd_signal
  - atr_14
  - volume_sma_20
  - bb_upper
  - bb_lower

features_1d:
  - open
  - high
  - low
  - close
  - volume
  - ema_50
  - ema_200
  - rsi_14
  - adx_14
  - atr_14
  - volume_sma_20

external:
  fear_greed:    true
  funding_rate:  true
  funding_rate_symbol: BTC/USDT:USDT

model_paths:
  hmm: models/gate_hmm.pkl
  xgb: models/gate_xgb_regime.pkl

p1:
  min_regime_confidence: 0.60

p2:
  onchain_enabled: true

p3:
  min_level_strength:    0.4
  fractal_n_4h:          2
  fractal_n_1d:          5
  min_swing_pct:         0.03
  volume_profile_bins:   50

p4:
  ewm_span:       3
  rsi2_oversold:  10

p5:
  max_risk_pct:            0.01   # 1% del capital per trade
  max_exposure_pct:        0.95   # mai >95% en posicions
  max_open_positions:      2
  weekly_drawdown_limit:   0.05   # 5% → circuit breaker setmanal
  stagnation_days:         6.0    # dies en pèrdues → reduir 50%

  trailing_atr_multiplier:
    low_vol:    1.5
    normal_vol: 2.0
    high_vol:   2.5

  decel_exit_candles:
    STRONG_BULL: 5
    WEAK_BULL:   3
    RANGING:     2

  min_rr:
    STRONG_BULL: 1.5
    WEAK_BULL:   2.0
    RANGING:     2.0
```

---

## Tests

```bash
# 54 tests unitaris cobreixen tot el Gate System
python -m pytest tests/unit/test_gate_system.py -v

# Resultat esperat: 54 passed
```

| Mòdul | Tests | Cobertura |
|-------|:-----:|---|
| P1 Regime | 7 | Features, inferència, llindars, regimes longables |
| P2 Health | 7 | F&G, funding rate, producte sub-scores, missing data |
| P3 Structure | 8 | Pivots, fibonacci, volume profile, merge, approach volume, R:R |
| P4 Momentum | 6 | Derivades, RSI-2, MACD, senyals per règim |
| P5 Risk | 11 | Kelly sizing, VETOs, trailing stop, tots els exits |
| HMM Trainer | 4 | Observacions, BIC, mapping K=2/K=5 |
| XGB Classifier | 2 | Walk-forward splits, no-overlap |
| NearMiss Logger | 2 | Exception safety, defaults |
| GateBot | 2 | `_hold()` signal, daily candle detection |
| Config | 3 | YAML loading, features spec, regime names |
