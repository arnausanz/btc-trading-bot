# Revisió Pre-Producció — Informe Final

> **Data:** Març 2026
> **Estat:** Revisió completada — veure secció "Passos finals" per al que queda fer abans de deployar

---

## Resum executiu

El projecte és **tècnicament sòlid** i ben estructurat. Té una arquitectura neta, documentació àmplia, i disseny que escala. S'han trobat **2 bugs crítics** (un dels quals hauria causat crash en producció), **4 inconsistències de codi menor**, i **5 errors de documentació**. Tot s'ha corregit en aquesta revisió.

El projecte **no està llest per a Oracle Cloud avui** — no perquè hi hagi deute tècnic greu, sinó perquè les tasques del ROADMAP (A, B, D) encara no s'han completat: els models RL no estan entrenats amb hiperparàmetres òptims i el backtest EnsembleBot no s'ha executat.

---

## 1. Bugs trobats i corregits

### BUG CRÍTIC 1 — `bots/rl/rl_bot.py`: obs_shape incorrecte per agents professionals

**Gravetat:** 🔴 Crític — hauria causat `ValueError: Unexpected observation shape` en producció

**Descripció:** Els agents professionals (`ppo_professional`, `sac_professional`, `td3_professional`, `td3_multiframe`) s'entrenen amb `obs_shape = n_features × lookback + 4`, on els 4 dims addicionals representen l'estat de posició: `[pnl_pct, position_fraction, steps_norm, drawdown_pct]`. El codi d'inferència a `rl_bot.py` construïa un vector de `n_features × lookback` sense afegir aquests 4 dims, causant un mismatch d'observació en el primer pas d'inferència.

**Per què no s'havia detectat:** Cap model professional `.zip` existia a `models/` (tots pendents d'entrenament), de manera que el bug dormia.

**Fix aplicat:**
```python
# rl_bot.py — afegit _PROFESSIONAL_TYPES i bloc d'augmentació
_PROFESSIONAL_TYPES: frozenset[str] = frozenset({
    "ppo_professional", "sac_professional",
    "td3_professional", "td3_multiframe",
})

# A on_observation(), després de construir obs_flat:
if agent_type in self._PROFESSIONAL_TYPES:
    portfolio = observation.get("portfolio", {})
    usdt  = float(portfolio.get("USDT", 0.0))
    btc   = float(portfolio.get("BTC",  0.0))
    price = float(features["close"].iloc[-1])
    total = usdt + btc * price
    pos_frac  = (btc * price) / max(total, 1e-8)
    pnl       = (total - self._entry_capital) / max(self._entry_capital, 1.0)
    steps_norm = min(self._steps_since_trade / 100.0, 1.0)
    peak      = max(self._peak_value, total)
    dd        = (total - peak) / max(peak, 1e-8)
    pos_state = np.array([pnl, pos_frac, steps_norm, dd], dtype=np.float32)
    obs_flat  = np.concatenate([obs_flat, pos_state])
```

---

### BUG MENOR 2 — `exchanges/paper.py`: USDT pot quedar lleugerament negatiu

**Gravetat:** 🟡 Menor — no causa crash però crea estat de portfolio inconsistent

**Descripció:** Quan `signal.size == 1.0` (tot el capital), la condició de validació `capital_to_use > self._portfolio["USDT"]` passa (100 == 100), però llavors `USDT -= capital_to_use + fees` deixa el saldo en `-fees` (uns -0.1 USDT).

**Fix aplicat:**
```python
# Abans (incorrecte):
if capital_to_use > self._portfolio["USDT"]:
    return  # insuficient capital

# Després (correcte):
if capital_to_use * (1 + self.fee_rate) > self._portfolio["USDT"]:
    return  # insuficient capital (inclou comissions)
```

---

## 2. Inconsistències de codi corregides

### 2.1 — Paths de config incorrectes en bots clàssics

Quatre bots tenien valors default de config_path apuntant a `config/bots/` (carpeta que no existeix) en lloc de `config/models/`:

| Fitxer | Valor incorrecte | Valor correcte |
|--------|-----------------|----------------|
| `bots/classical/dca_bot.py` | `config/bots/dca.yaml` | `config/models/dca.yaml` |
| `bots/classical/hold_bot.py` | `config/bots/hold.yaml` | `config/models/hold.yaml` |
| `bots/classical/grid_bot.py` | `config/bots/grid.yaml` | `config/models/grid.yaml` |
| `bots/classical/trend_bot.py` | `config/bots/trend.yaml` | `config/models/trend.yaml` |

### 2.2 — `bots/rl/rewards/advanced.py`: duplicació de la constant ATR_REFERENCE

`advanced.py` definia `_ATR_REFERENCE = 0.02` en lloc d'importar-la de `bots/rl/constants.py`, que és la font de veritat. Si la constant canviava a `constants.py`, `advanced.py` quedava desincronitzat en silenci.

**Fix:** Canviat a `from bots.rl.constants import ATR_REFERENCE; _ATR_REFERENCE = ATR_REFERENCE`

### 2.3 — `bots/rl/trainer.py`: obs_shape reportada erròniament al log de MLflow

El log i MLflow reportaven `obs_shape = lookback × n_features` per a tots els agents, sense afegir els +4 dims dels agents professionals. Hauria creat confusió en depurar models.

**Fix:** Afegit `pos_state_dims = 4 if agent_type in _PROFESSIONAL_ENV_TYPES else 0` al càlcul i al log.

### 2.4 — `bots/ml/ml_bot.py`: docstring incorrecte sobre com afegir models

El docstring deia "Afegir un nou model ML = una línia a `_MODEL_REGISTRY`", que és incorrecte — el registre es construeix per auto-discovery des de `config/models/*.yaml`.

**Fix:** Corregit per reflectir el mecanisme real.

---

## 3. Errors de documentació corregits

### 3.1 — `docs/CONFIGURATION.md`: slippage_rate incorrecte

Mostrava `0.0001` però `config/exchanges/paper.yaml` real té `0.0005`. Ara coincideix.

### 3.2 — `docs/CONFIGURATION.md`: regla obs_shape incompleta

La secció sobre `obs_shape` per a RL no incloïa els variants TD3 (`td3_professional`, `td3_multiframe`) ni explicava que els 4 dims addicionals són `[pnl_pct, position_fraction, steps_norm, drawdown_pct]`. Corregit i ampliat.

### 3.3 — `docs/EXTENDING.md`: Secció EnsembleBot obsoleta

La secció 6 sobre EnsembleBot deia "ROADMAP — pendent" i tenia pseudocodi placeholder. L'EnsembleBot ja estava completament implementat. Substituïda per la documentació real: polítiques disponibles, YAML d'exemple, criteris d'entrada per a sub-bots.

### 3.4 — `docs/PROJECT.md`: referència a carpeta obsoleta

El mapa de directoris referencava `[OLD]config/` que no existeix. Eliminat.

### 3.5 — `docs/ROADMAP.md`: estat C2 marcat incorrectament com a pendent

El ROADMAP indicava que la persistència d'estat (restauració de portfolio + posicions al reinici) era pendent. En realitat `demo_runner.py` ja implementava `restore_state()` via `get_last_state()` de `demo_repository.py`. Marcat com ✅.

---

## 4. Actualitzacions de documentació noves

### 4.1 — `docs/ROADMAP.md` reeditat a v4.0

- Taula d'estat del sistema actualitzada (C2 State restore → ✅)
- Camí crític actualitzat
- **Nova secció I:** Nous models recomanats amb prioritats:
  - EnsembleBot `weighted` (Alta): Sharpe sliding-window
  - N-BEATS / N-HiTS (Mitjana): architectures de forecasting pures
  - TabNet (Mitjana): tabular attention-based
  - BreakoutBot (Mitjana): pivot points + ATR, complement natural de TrendBot
  - DreamerV3 (Baixa): world model RL, projecte de recerca independent
- **Nova secció J:** Capa LLM per a sentiment de notícies (⭐ Recomanada)
  - Arquitectura: CryptoPanic/RSS → LLM → feature `news_sentiment [-1,1]` → BD → FeatureBuilder
  - Per què feature i no gatekeeper (evitar vetoes incorrectes)
  - 6 passos d'implementació
  - Taula de riscos i mitigació
  - Cost: ~$20/any
  - Prioritat: implementar un cop la demo estigui en marxa

### 4.2 — `config/demo.yaml`: agents professionals afegits

Afegides 4 entrades que faltaven (`ppo_professional`, `sac_professional`, `td3_professional`, `td3_multiframe`) com a `enabled: false` amb comentaris sobre com entrenar-les.

---

## 5. Anàlisi de coherència: models vs objectiu (swing BTC trading)

### Adequació general: ✅ Excel·lent

Tots els bots i models estan dissenyats per a swing trading en timeframes 1H–12H. No hi ha bots de scalping ni estratègies que necessiten latència molt baixa.

### Anàlisi per model

| Model | Adequació | Observació |
|-------|-----------|------------|
| TrendBot | ✅ | RSI + EMA — ideal per a tendències 1H |
| DCABot | ✅ | Acumulació gradual — robusta com a baseline |
| GridBot | ⚠️ | Útil en rang però els paràmetres cal ajustar per BTC alta vol |
| MeanReversionBot | ✅ | Bollinger Bands — complementa TrendBot |
| MomentumBot | ✅ | ROC + RSI — captura continuació de tendència |
| HoldBot | ✅ | Benchmarck correcte (buy and hold BTC) |
| EnsembleBot | ✅ | Meta-bot sòlid; majority_vote evita overfit a un sol model |
| RF / XGB / LGBM / CB | ✅ | Ensemble de trees, ben implementats amb CV temporal |
| GRU | ✅ | Seq2seq, adequat per patrons temporals |
| PatchTST | ✅ | Transformer amb patching — millor que GRU en dades financeres llargues |
| PPO / SAC baseline | ✅ | 1H, bon equilibri entre exploració i explotació |
| PPO / SAC on-chain | ✅ | Fear & Greed + funding rate aporta context macro valuós |
| PPO / SAC professional | ✅ | ATR stop, pos state — molt adequat per gestió de risc real |
| TD3 professional | ✅ | Acció contínua + regime_adaptive reward — el més sofisticat |
| TD3 multiframe | ✅ | 12H + 4H — excel·lent per multi-escala en swing trading |

### Sobre l'annualització de mètriques

La funció `metrics.py` usa `sqrt(365 × 24)` per dades 1H (no 252 com borsàtica), correcte per mercats cripto 24/7.

### Sobre la separació temporal

Walk-forward correcte: TRAIN_UNTIL = 2024-12-31, TEST_FROM = 2025-01-01. Cap bot pot veure dades de test durant l'entrenament — no hi ha lookahead bias estructural.

---

## 6. Verificació de la demo: no hi ha "trampes"

### Comissions: ✅

`PaperExchange` aplica `fee_rate = 0.001` (0.1%) a cada operació, tant en compra com en venda. Equivalent a la tarifa taker de Binance.

### Slippage: ✅

`slippage_rate = 0.0005` (0.05%) simulat afegint impacte de mercat al preu d'execució. Conservador però realista per BTC (mercat molt líquid).

### Sincronització de candles: ✅

El `DemoRunner` crida `on_observation()` **una vegada per candle tancada** (no en temps real). Les decisions es prenen amb preus de tancament confirmats — no hi ha lookahead intra-candle.

### Persistència d'estat: ✅

Portfolio USDT + BTC i `_in_position` es restauren des de la BD en cada reinici. Un reinici del servidor no reinicia falsament el capital.

### Separació train/test: ✅

Els models veuen dades fins al 2024-12-31 durant l'entrenament. La demo opera en temps real (post 2025-01-01). No hi ha contaminació.

---

## 7. Valoració de qualitat de codi

### Punts forts

- **Arquitectura neta:** Interfaces (`BaseBot`, `BaseRLAgent`, `BaseMLModel`) ben definides. Herència apropiada, no abusiva.
- **Auto-discovery:** Sense llistes hardcodejades de models — afegir un model nou = crear YAML + classe.
- **Reward registry:** Patró `@register("name")` elegant. Fàcil afegir reward functions sense tocar entorns.
- **Scaffold compartit:** `BaseTreeModel` elimina duplicació entre RF/XGB/LGBM/CB. `TimeSeriesDataset` compartit entre GRU/PatchTST.
- **Gestió d'errors:** `ValueError` amb missatges explicatius quan el dataset és massa petit o l'obs_shape no coincideix.
- **Logging consistent:** `logging.getLogger(__name__)` a tots els mòduls.
- **Configuració unificada:** Un sol YAML per model (entrenament + optimització + desplegament).

### Punts millorables (no blocants)

- Els bots clàssics (`trend_bot.py`, etc.) llegeixen el YAML directament en `__init__` en lloc de rebre el config com a diccionari. Funcional però menys testable. (No canviat per no introduir regressions.)
- `demo_runner.py` té `_RL_TYPES = {"ppo", "sac", "td3"}` hardcodejat en lloc d'usar el camp `category` del YAML. Funciona perquè `ppo_professional` usa `model_type: ppo` al YAML, però és lleugerament implícit.

---

## 8. Llista completa de fitxers modificats

| Fitxer | Tipus de canvi |
|--------|---------------|
| `bots/rl/rl_bot.py` | 🔴 Bug crític — obs_shape professional |
| `exchanges/paper.py` | 🟡 Bug menor — USDT negatiu amb size=1.0 |
| `bots/classical/dca_bot.py` | 🔵 Qualitat — path config incorrecte |
| `bots/classical/hold_bot.py` | 🔵 Qualitat — path config incorrecte |
| `bots/classical/grid_bot.py` | 🔵 Qualitat — path config incorrecte |
| `bots/classical/trend_bot.py` | 🔵 Qualitat — path config incorrecte |
| `bots/rl/rewards/advanced.py` | 🔵 Qualitat — constant duplicada |
| `bots/rl/trainer.py` | 🔵 Qualitat — obs_shape MLflow log incorrecte |
| `bots/ml/ml_bot.py` | 🔵 Qualitat — docstring incorrecte |
| `docs/CONFIGURATION.md` | 📄 Doc — slippage_rate + obs_shape RL |
| `docs/EXTENDING.md` | 📄 Doc — secció EnsembleBot actualitzada |
| `docs/PROJECT.md` | 📄 Doc — carpeta obsoleta eliminada |
| `docs/ROADMAP.md` | 📄 Doc — v4.0, secció I+J nous models+LLM |
| `config/demo.yaml` | ⚙️ Config — 4 agents RL professionals afegits |

---

## 9. Passos finals abans de migrar a Oracle Cloud

Ordenats per prioritat i dependència:

### Blocanters (cal fer-los en ordre)

**[1] Acabar optimització Optuna** (ja en curs)
```bash
python scripts/optimize_bots.py
python scripts/optimize_models.py
```

**[2] Entrenar tots els models amb hiperparàmetres òptims**
```bash
python scripts/train_models.py          # RF, XGB, LGBM, CB, GRU, PatchTST
python scripts/train_rl.py              # PPO, SAC, PPO_onchain, SAC_onchain
python scripts/train_rl.py --agents ppo_professional sac_professional td3_professional td3_multiframe
```
Recorda ajustar `total_timesteps` als YAMLs (1M per agents 1H, 500k per agents 12H) abans d'entrenar.

**[3] Backtest EnsembleBot**
```bash
python scripts/run_comparison.py --bots hold trend mean_reversion momentum \
    ml_xgb ml_lgbm ml_rf rl_ppo rl_sac ensemble
```
Criteri d'entrada a la demo: **Sharpe > 1.0 i Drawdown màx. < -25%** en out-of-sample (TEST_FROM en endavant).

Consulta els resultats per decidir quins bots activar a `config/models/ensemble.yaml` (secció `sub_bots`) i quins activar a `config/demo.yaml`.

**[4] Migrar a Oracle Cloud Free Tier**

Infraestructura mínima al servidor (veure ROADMAP §D):
- Cron jobs: `update_data.py` (1H), `update_fear_greed.py` (diari), `update_futures.py` (4H), `update_blockchain.py` (diari)
- Servei systemd amb restart automàtic: `python scripts/run_demo.py`
- Backup nocturn: `pg_dump btc_trading`

**[5] Iniciar demo** (3-6 mesos de paper trading)
```bash
python scripts/run_demo.py
```

### No blocants (fer mentre la demo corre)

- **Capa LLM sentiment** (recomanat, secció J del ROADMAP): ~1 jornada de treball, cost ~$20/any
- **Telegram millorat**: resum diari, alertes drawdown, comandes `/status`, `/portfolio`, `/ranking`
- **Dashboard Streamlit**: millor fer-lo un cop hi hagi mesos de dades reals
- **Nous models** (N-BEATS, TabNet, BreakoutBot): prioritat mitjana, no bloquen la demo

---

## 10. Valoració final de readiness per a producció

| Criteri | Estat | Notes |
|---------|-------|-------|
| Arquitectura | ✅ Sòlida | Clean, extensible, ben documentada |
| Bugs crítics | ✅ Corregits | 2 bugs trobats i resolts en aquesta revisió |
| Demo sense trampes | ✅ Verificat | Comissions, slippage, candle-sync, persistència correctes |
| Documentació | ✅ Actualitzada | Totes les inconsistències corregides |
| Models entrenats | ⏳ Pendent | Optuna en curs → entrenament òptim pendent |
| Backtest EnsembleBot | ⏳ Pendent | Prerequisit per a la demo |
| Servidor Oracle | ⏳ Pendent | Depèn de completar [1-3] |

**Conclusió:** El codi és production-ready. Ara cal completar el cicle d'entrenament → validació → deploy.

---

*Revisió realitzada: Març 2026*
