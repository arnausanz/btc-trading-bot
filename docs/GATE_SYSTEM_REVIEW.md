# Gate System — Revisió Completa

**Data:** 2026-03-21
**Abast:** Revisió exhaustiva de la implementació del Gate System (5 portes), models de règim, integració amb DemoRunner, coherència config/DB, eficiència i tests.

**Resultat global:** La implementació és sòlida i coherent amb l'especificació. S'han detectat **3 bugs funcionals**, **2 problemes de performance** i **5 millores recomanades**. Els 54 tests unitaris passen al 100%.

---

## 1. BUGS FUNCIONALS (Prioritat ALTA)

### 1.1 `_hold()` amb `bot_id` hardcoded
- **Fitxer:** `bots/gate/gate_bot.py`, línia ~402
- **Problema:** `_hold()` és un `@staticmethod` que usa `bot_id = "gate_v1"` directament en comptes de `self.bot_id`. Si mai canvies el `bot_id` a config (p.ex. `gate_v2`), els senyals HOLD tindran un identificador incorrecte.
- **Fix:** Convertir a mètode d'instància o passar `self.bot_id` com a argument.

### 1.2 `p3_open` sempre avalua a `True`
- **Fitxer:** `bots/gate/gate_bot.py`, línia ~319
- **Problema:** `p3_open = any(True for _ in self._open_positions)` retorna True si hi ha qualsevol posició oberta, independentment de l'avaluació P3 real. L'objectiu era re-avaluar P3 per detectar estancament (stagnation), però el codi actual simplement comprova si hi ha posicions.
- **Fix:** Re-avaluar P3 amb el preu actual per determinar si l'estructura de suport encara és vàlida.

### 1.3 Circuit breaker compara moviment total vs. moviment en 1 candle
- **Fitxer:** `bots/gate/gates/p5_risk.py`, línia ~208
- **Problema:** `abs(current_price - entry_price) > 3.0 * atr_14` compara el moviment acumulat des de l'entrada. L'especificació (gate_system_final.md) indica que hauria de ser el moviment en **1 sola candle** (3× ATR en un sol període), no el total.
- **Fix:** Comparar `abs(current_price - previous_close)` en lloc de `abs(current_price - entry_price)`.

---

## 2. PROBLEMES DE PERFORMANCE (Prioritat MITJANA)

### 2.1 `_count_touches()` usa `iterrows()`
- **Fitxer:** `bots/gate/gates/p3_structure.py`, línia ~315
- **Problema:** `iterrows()` és O(n×m) on n=candles i m=nivells. Amb 300 candles 4H i múltiples nivells, és innecessàriament lent.
- **Fix vectoritzat:**
```python
for level in levels:
    touches = ((df['low'] <= level.price + tol) & (df['high'] >= level.price - tol)).sum()
    level.touches = int(touches)
```

### 2.2 `_volume_profile()` amb bucle Python
- **Fitxer:** `bots/gate/gates/p3_structure.py`, línia ~253
- **Problema:** Usa un bucle Python per assignar volum a bins de preu. NumPy ho fa directament.
- **Fix:**
```python
bin_edges = np.linspace(low, high, n_bins + 1)
bin_indices = np.digitize(df['close'].values, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
volume_profile = np.bincount(bin_indices, weights=df['volume'].values, minlength=n_bins)
```

---

## 3. DISCREPÀNCIES AMB L'ESPECIFICACIÓ (Prioritat BAIXA)

### 3.1 `_FIB_LEVELS` inclou 0.236
- **Fitxer:** `bots/gate/gates/p3_structure.py`
- **Problema:** La constant `_FIB_LEVELS` inclou el nivell 0.236, que no apareix a l'especificació (gate_system_final.md §5 defineix 0.382, 0.5, 0.618, 0.786).
- **Impacte:** Baix — afegir un nivell extra no trenca res, però genera nivells addicionals que podrien diluir la qualitat dels merge.
- **Recomanació:** Eliminar 0.236 per alinear-se amb l'spec, o documentar-ho com a decisió de disseny intencional.

### 3.2 `_fit_final()` sense early_stopping
- **Fitxer:** `bots/gate/regime_models/xgb_classifier.py`, línies ~232-244
- **Problema:** El model final entrena amb tot el dataset sense `early_stopping_rounds`, mentre que la validació walk-forward sí que l'usa (50 rounds). Això pot causar sobreajust subtil.
- **Recomanació:** Usar un petit split de validació (5-10%) per mantenir early_stopping en el model final, o fixar `n_estimators` al valor mitjà observat durant walk-forward.

---

## 4. MILLORES RECOMANADES

### 4.1 P1: Cache de features
`_compute_features()` recalcula totes les EMAs i derivades cada vegada que s'invoca. Com que P1 només s'avalua un cop al dia, l'impacte és mínim, però un cache amb hash del timestamp de la darrera candle evitaria recomputacions innecessàries si es crida múltiples vegades dins el mateix cicle.

### 4.2 DemoRepository: Sessions compartides
Cada operació a `core/db/demo_repository.py` crea i tanca la seva pròpia sessió de DB. Per a operacions ràpides consecutives (p.ex. actualitzar múltiples posicions), un context manager amb sessió compartida reduiria l'overhead.

### 4.3 GateBot: Múltiples exits per cicle
`on_observation()` retorna només el primer senyal d'exit quan múltiples posicions necessiten tancament simultani. Si dues posicions triguen un stop al mateix candle, només se'n tanca una. Considerar retornar una llista de senyals o processar tots els exits en un sol cicle.

### 4.4 Tipus de retorn de `on_observation()`
El mètode retorna `Signal | None`, però `BaseBot` defineix `on_observation() → Signal`. Seria més net retornar sempre un Signal (amb `action=HOLD` quan no hi ha acció) per mantenir consistència amb la interfície.

### 4.5 Walk-forward: Folds configurables
El nombre de folds a `xgb_classifier.py` està fixat a 5. Fer-ho configurable via `gate.yaml` permetria experimentar amb la granularitat de validació.

---

## 5. COHERÈNCIA VERIFICADA ✓

Tots aquests aspectes s'han verificat i són correctes:

- **Config ↔ Codi:** `gate.yaml` carrega correctament, tots els paràmetres arriben als gates, `bot_id/module/class_name` coincideixen.
- **DB ↔ Codi:** `GatePositionDB` i `GateNearMissDB` tenen tots els camps que usen `GateBot` i `NearMissLogger`. L'alembic migration és coherent.
- **DemoRunner ↔ GateBot:** El bucket 4H funciona, `_TF_SECONDS` inclou `4h: 14400`, el `from_config()` carrega correctament, l'`ObservationSchema` multi-timeframe (4h+1d) s'integra bé amb `ObservationBuilder`.
- **P1-P5 pipeline seqüencial:** L'ordre d'avaluació és correcte (P1→P2→P3→P4→P5), amb short-circuit adequat a cada porta.
- **Near-miss logging:** Es dispara quan P1+P2+P3 passen, captura totes les dades relevants, mai bloqueja el flux de trading.
- **Regime model pipeline:** HMM→Viterbi→XGBoost walk-forward amb Optuna, validació amb llindars (accuracy >55%, mlogloss std <15%).
- **Long-only v1:** No hi ha cap lògica de SHORT amagada — tot correcte.

---

## 6. TESTS

**54 tests unitaris** creats a `tests/unit/test_gate_system.py`, tots passant:

| Mòdul | Tests | Cobertura |
|--------|-------|-----------|
| P1 Regime | 7 | Features, inferència, llindars confiança, regimes longables |
| P2 Health | 7 | Taula FG, funding rate, producte sub-scores, missing data |
| P3 Structure | 8 | Pivots, fibonacci, volume profile, merge, approach volume, R:R |
| P4 Momentum | 6 | Derivades, RSI-2, MACD, senyals per règim, unknown regime |
| P5 Risk | 11 | Kelly sizing, VETOs, trailing stop, exits, ATR percentile |
| HMM Trainer | 4 | Observacions, BIC, mapping K=2/K=5 |
| XGB Classifier | 2 | Walk-forward splits, no-overlap |
| NearMiss Logger | 2 | Exception safety, defaults |
| GateBot | 2 | _hold() signal, daily candle detection |
| Config | 3 | YAML loading, features spec, regime names |

---

## 7. RESUM EXECUTIU

El Gate System està ben dissenyat i la implementació segueix l'especificació amb fidelitat. Els 3 bugs funcionals detectats no són crítics (el bot funciona), però haurien de corregir-se abans de posar-ho en producció:

1. **Urgent:** Fix del circuit breaker (1.3) — pot no disparar-se quan hauria o disparar-se quan no toca.
2. **Important:** Fix de `p3_open` (1.2) — la verificació d'estancament no funciona correctament.
3. **Menor:** Fix de `_hold()` (1.1) — només és problemàtic si canvies el `bot_id`.

Les optimitzacions de performance (2.1, 2.2) no són bloquejants amb 300 candles, però es notarien amb datasets més grans o backtests extensos.
