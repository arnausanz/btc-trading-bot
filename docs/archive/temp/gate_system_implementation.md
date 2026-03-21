# Gate System — Resum d'Implementació

> Document generat durant la implementació. S'actualitza pas a pas.
> Referència tècnica completa: `docs/gate_system_final.md`

---

## Resum executiu

El Gate System és un bot de swing trading (BTC/USDT, ~10 trades/mes) que avalua seqüencialment 5 portes. Si qualsevol porta es tanca, el bot no opera. V1 és long-only; P1 usa HMM (descoberta de règims) + XGBoost (classificació en temps real); P2–P5 són deterministes.

S'integra com un bot més al `DemoRunner` (mateix patró `BaseBot + ObservationSchema`). Cap bot existent es veu afectat.

---

## 1. Canvis al core (backward-compatible)

### `data/observation/builder.py`

**Problema original:** `build(schema, symbol, index)` aplicava el mateix `index` a tots els timeframes. Un `index ≈ 12.000` del df 4H aplicat al df diari (≈ 3.000 files) causava out-of-bounds.

**Solució implementada:** Per cada timeframe secundari, es troba l'índex correcte via `searchsorted(timestamp)` del timeframe primari. Per a bots amb un sol timeframe, `searchsorted` retorna exactament el mateix `index` que abans — comportament idèntic.

Canvi clau a `build()`:
```python
primary_tf  = schema.timeframes[0]
primary_df  = self._cache[f"{symbol}_{primary_tf}"]
timestamp   = primary_df.index[index]
# Per cada TF: tf_idx = df.index.searchsorted(timestamp, side="right") - 1
```

### `core/engine/demo_runner.py`

**Problema original:** `_last_candle_hour` usava buckets d'1 hora per tots els bots. El GateBot (timeframe primari 4H) necessita detecció cada 4H.

**Solució implementada:** `_last_candle_bucket` + mapa `_TF_SECONDS` per calcular el bucket correcte per a cada bot. Per a tots els bots actuals (`timeframe=1h`), `tf_secs=3600` → comportament idèntic.

---

## 2. Base de dades

### `core/db/models.py` — models afegits

- **`GatePositionDB`** (`gate_positions`): persisteix les posicions obertes del GateBot entre reinicis. Camps: entry_price, stop_level, target_level, highest_price, size_usdt, regime, opened_at, decel_counter, bot_id.
- **`GateNearMissDB`** (`gate_near_misses`): registra cada avaluació on P1+P2+P3 passen, independentment de si s'executa un trade. Permet analitzar quines portes tanquen més i per quin motiu.

### Migració Alembic

Fitxer: `alembic/versions/b3c4d5e6f7a8_add_gate_tables.py` — afegeix les dues taules noves. No modifica cap taula existent.

### `core/db/demo_repository.py` — mètodes afegits

- `save_gate_position()`, `update_gate_position()`, `delete_gate_position()`, `get_open_gate_positions(bot_id)` → per a `GatePositionDB`
- `save_gate_near_miss()` → per a `GateNearMissDB`

---

## 3. Estructura de fitxers nous

```
bots/gate/
├── __init__.py
├── gate_bot.py                    # GateBot(BaseBot) — orquestra P1..P5
├── near_miss_logger.py            # Escriu a gate_near_misses via DemoRepository
└── gates/
    ├── __init__.py
    ├── p1_regime.py               # Càrrega HMM + XGBoost, inferència de règim
    ├── p2_health.py               # Fear & Greed + on-chain → position_multiplier [0,1]
    ├── p3_structure.py            # Fractals + Fibonacci + Volume Profile + approach vol
    ├── p4_momentum.py             # Derivades EWM + RSI-2 + MACD cross → trigger + score
    └── p5_risk.py                 # Kelly sizing + trailing stop + gestió posicions

bots/gate/regime_models/
├── __init__.py
├── hmm_trainer.py                 # HMM K=2..6, BIC selection, Viterbi labels, state mapping
└── xgb_classifier.py             # XGBoost multiclasse, walk-forward CV, Optuna

config/models/gate.yaml            # Configuració unificada del GateBot
scripts/train_gate_regime.py       # Pipeline complet: dades → HMM → XGBoost → save
```

---

## 4. Detall de cada fitxer

### `bots/gate/gates/p3_structure.py`

Porta 3: identifica nivells de preu on el bot podria actuar.

Retorna `P3Result(has_actionable_level, best_level, stop_level, target_level, risk_reward, volume_ratio)`.

Tres mètodes de detecció de nivells (deterministes, sense ML):
- **Pivots fractals**: swing highs/lows amb N=2 al 4H i N=5 al diari
- **Fibonacci retracement**: nivells 38.2%, 50%, 61.8%, 78.6% de l'últim swing significatiu (≥3%)
- **Volume Profile**: High Volume Nodes (bins amb volum > 1.5× la mitjana) al 4H

Força del nivell [0–1]: 1 font → 0.3, 2 fonts → 0.6, 3 fonts → 0.9. +0.1 per cada toc anterior sense trencar. Proximity threshold adaptatiu per ATR (1.0–2.0× ATR depenent de la força).

Filtre de volum d'acostament: si `volume_ratio = mean(últimes 3 candles) / vol_ma20 < 0.8` → `has_actionable_level = False` independent de la força del nivell.

### `bots/gate/gates/p4_momentum.py`

Porta 4: trigger d'entrada. Mesura l'estat actual del momentum, no prediu.

Retorna `P4Result(triggered, confidence, signals_active, signals_detail)`.

4 senyals calculats sobre el 4H EWM-smoothed (span=3):
1. `d1 > 0 AND d2 >= 0` — momentum positiu accelerant
2. `d1 < 0 AND d2 > 0` — momentum negatiu frenant (reversal)
3. `RSI-2 < 10` — sobrevenut extrem (Connors)
4. MACD(12,26,9) creuant senyal en alça

Mínim de senyals adaptatiu al règim P1:
- `STRONG_BULL/BEAR`: 1/4 en favor de tendència
- `WEAK_BULL/BEAR`: 2/4 en favor
- `RANGING`: 2/4

### `bots/gate/gates/p5_risk.py`

Porta 5: gestió de risc, sizing i posicions obertes.

Retorna `P5Result(vetoed, veto_reason, position_size_fraction, order_action)` per a noves entrades. Per a posicions obertes: `P5TrailingResult(should_exit, exit_reason, new_stop)`.

**Sizing** (Kelly fraccionari):
```
risc_eur = capital × max_risk_pct × multiplier_P2 × confidence_P4
dist_stop = |entry - stop| / entry
posicio_usdt = risc_eur / dist_stop
size_fraction = min(posicio_usdt / usdt_balance, max_exposure_pct)
```

**Condicions de VETO**: R:R < mínim per règim, massa posicions obertes, drawdown setmanal excedit, exposició total excessiva.

**Trailing stop adaptatiu**: activat quan el preu ha guanyat ≥ 1× ATR. Multiplicador: 1.5× (vol baix), 2.0× (normal), 2.5× (vol alt).

**Sortides condicionals**: desacceleració (d2 negatiu N candles), invalidació de règim, emergència (P2 = 0), estancament prolongat (50% reduït).

**Circuit breaker**: moviment > 3× ATR en 1 candle → re-avaluació immediata.

### `bots/gate/regime_models/hmm_trainer.py`

Entrena un `GaussianHMM` (hmmlearn) sobre 3 observacions diàries: `[daily_return, normalized_atr14, volume_sma_ratio]`.

Selecciona K (nombre d'estats) de 2 a 6 per BIC mínim. Per cada K: 10 inicialitzacions aleatòries, es queda amb la de màxima log-verosimilitud.

Decodifica amb Viterbi → seqüència d'estats `[0, 2, 1, 0, 3, ...]`.

Mapeig automàtic HMM state → `RegimeState` (`STRONG_BULL`, `WEAK_BULL`, `RANGING`, `WEAK_BEAR`, `STRONG_BEAR`, `UNCERTAIN`) ordenat per mean_return i discriminat per ADX i volatilitat.

Guarda `{model, state_mapping, k}` en `models/gate_hmm.pkl`.

### `bots/gate/regime_models/xgb_classifier.py`

Entrena un XGBoost multiclasse (`multi:softprob`) sobre les 14 features de P1 amb les etiquetes de l'HMM com a target.

Walk-forward validation: ≥5 folds, train ≥2 anys, test 3-6 mesos. Optuna TPE + Hyperband minimitza la `mlogloss` mitjana across folds.

Guarda el model final (re-entrenat sobre tot el dataset) a `models/gate_xgb_regime.pkl`.

### `bots/gate/gates/p1_regime.py`

Carrega `gate_hmm.pkl` + `gate_xgb_regime.pkl` en inicialitzar.

`evaluate(df_1d) → P1Result(regime, confidence, probabilities)`:
- Calcula les 14 features de P1 sobre l'última fila del df diari
- XGBoost prediu probabilitats per a cada estat HMM
- L'estat amb probabilitat màxima és el règim actiu
- Si max_prob < `min_regime_confidence (0.60)` → `UNCERTAIN`, porta tancada

### `bots/gate/gates/p2_health.py`

Calcula `position_multiplier [0.0–1.0]` com a producte de sub-scores:
- **Fear & Greed** (contrarian en tendència): taula de lookup {FG_range: score} diferenciada per bull vs bear
- **On-chain flow proxy** (funding rate com a proxy): funding negatiu → distribució → score baix; funding positiu en bull → acumulació → score alt

V1 sense news sentiment. El multiplier és el producte dels sub-scores. Si qualsevol és 0 → veto total.

### `bots/gate/near_miss_logger.py`

Classe `NearMissLogger` amb mètode `log(snapshot: GateSnapshot)` que persiste un registre a `gate_near_misses` via `DemoRepository.save_gate_near_miss()`.

Es crida des de `GateBot.on_observation()` cada cop que P1+P2+P3 passen, registrant l'estat de totes les portes (inclús si P4 o P5 no deixen executar el trade).

### `bots/gate/gate_bot.py`

`GateBot(BaseBot)` — orquestra les 5 portes.

`observation_schema()`: declara `timeframes=["4h","1d"]`, `lookback=300`, unió deduplificada de features 4H i 1D.

`on_start()`: carrega P1 (models HMM + XGBoost), restaura posicions obertes des de `DemoRepository`.

`on_observation(obs)`:
1. Extrau `feat_4h = obs["4h"]` i `feat_1d = obs["1d"]`
2. Si nova candle diària: re-evalua P1 + P2 (1 cop/dia)
3. Sempre (cada 4H): avalua P3 → si passa: avalua P4 → si passa: avalua P5
4. Si P1+P2+P3 passen: crida `near_miss_logger.log()`
5. Si trade executat: persiste posició via `DemoRepository.save_gate_position()`
6. Per posicions obertes: crida P5 trailing/exit logic
7. Circuit breaker: si `|close_change| > 3×ATR` → re-avaluació immediata

Retorna `Signal(action=BUY/SELL/HOLD, size=fraction, confidence=p4_score, reason=...)`.

### `scripts/train_gate_regime.py`

Pipeline complet d'entrenament en 7 passos:
1. Carregar OHLCV 1D + FG + funding rate via `FeatureBuilder`
2. Calcular les 14 features de P1 (EMA slopes, ADX, ATR percentile, etc.)
3. Entrenar HMM K=2..6, seleccionar per BIC, decodificar Viterbi → labels
4. Mapejar HMM states → `RegimeState` per mean_return + ADX
5. Walk-forward XGBoost (5 folds) + Optuna
6. Validar (accuracy OOS >55%, mlogloss estabilitat <15%)
7. Guardar `models/gate_hmm.pkl` + `models/gate_xgb_regime.pkl`

### `config/models/gate.yaml`

Config unificada: timeframes (4h + 1d), lookback (300), paths dels models, llistes de features per TF, paràmetres de cada porta (P1 min_confidence, P3 fractal_n, P4 ewm_span, P5 max_risk_pct, trailing multipliers, etc.).

---

## 5. Com entrenar i activar

```bash
# 1. Crear les taules a la BD
alembic upgrade head

# 2. Entrenar els models de règim (P1)
cd /path/to/btc-trading-bot
python scripts/train_gate_regime.py
# Amb Optuna 100 trials: python scripts/train_gate_regime.py --n-trials 100
# Sense Optuna (paràmetres per defecte): python scripts/train_gate_regime.py --no-optuna

# 3. Verificar que els models s'han guardat
ls models/gate_hmm.pkl models/gate_xgb_regime.pkl

# 4. Activar al demo
# config/demo.yaml: gate_bot → enabled: true

# 5. Iniciar demo
python scripts/run_demo.py
```

---

## 6. Dependències noves

- `hmmlearn ^0.3.3` (afegit a `pyproject.toml`) — GaussianHMM per a descoberta no supervisada de règims
- La resta (`xgboost`, `optuna`, `scikit-learn`, `scipy`) ja existien al projecte

---

## 7. Fitxers modificats al core

| Fitxer | Canvi | Impacte |
|---|---|---|
| `data/observation/builder.py` | `searchsorted` per indexar TFs secundaris | Zero — bots 1-TF idèntics |
| `core/engine/demo_runner.py` | `_TF_SECONDS` + bucket per bot | Zero — tots els actuals usen 1H |
| `core/db/models.py` | Afegits `GatePositionDB` + `GateNearMissDB` | Addictiu, no trenca res |
| `core/db/demo_repository.py` | Afegits 5 nous mètodes Gate | Addictiu, no trenca res |
| `pyproject.toml` | `hmmlearn ^0.3.3` | Nova dependència |
| `config/demo.yaml` | Entrada gate_bot (enabled: false) | Inactive fins activació manual |

---

*Document actualitzat en finalitzar la implementació. Data: 2026-03-20.*
