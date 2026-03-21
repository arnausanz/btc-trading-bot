# Roadmap — Camí a la Demo 24/7

> **Regla d'or:** cap bot va a demo fins que en backtest out-of-sample superi HoldBot en Sharpe i Calmar.

---

## Estat del sistema

| Component | Estat | Notes |
|-----------|-------|-------|
| Classical bots (6) | ✅ | Trend, DCA, Grid, Hold, MeanReversion, Momentum |
| ML models (6+1) | ⚠️ Parcial | RF, XGB, LGBM, CB, GRU, PatchTST ✅ · TFT ⏳ Pendent |
| RL agents — baseline | ✅ | PPO, SAC (1H, 500k steps) |
| RL agents — on-chain | ✅ | PPO + SAC amb Fear&Greed, funding rate, hash-rate |
| RL agents — professional | ✅ | PPO + SAC (12H, ATR stop, position state, reward professional) |
| RL agents — TD3 | ✅ | td3_professional (12H) + td3_multiframe (12H+4H) |
| Dades externes | ✅ | Fear&Greed, funding rates, open interest, blockchain |
| Optimize workflow | ✅ | Optuna + best_params in-place al YAML base |
| BacktestEngine + MLflow | ✅ | Walk-forward, mètriques Sharpe/Calmar |
| DemoRunner | ✅ | Persistència + Telegram |
| **EnsembleBot v1** | ✅ | majority_vote — `bots/classical/ensemble_bot.py` |
| **State restore** | ✅ | Portfolio + _in_position restaurats des de BD al reinici |
| **Gate System v1** | ✅ | 5 portes seqüencials, HMM+XGBoost, swing trading 4H+1D |
| Documentació | ✅ | PROJECT, MODELS, EXTENDING, CONFIG, DATABASE, DECISIONS + Gate |

---

## Camí crític cap a la demo

```
[✅]   EnsembleBot v1 implementat (majority vote)
[✅]   State persistence (portfolio + posició restaurada des de BD)
[✅]   Gate System v1 implementat (5 portes, HMM+XGBoost, swing 4H+1D)
[ara]  Optimització Optuna de tots els models (en curs)
          ↓
[A]    Entrenar tots els models amb hiperparàmetres òptims
          + entrenar Gate System: alembic upgrade head + train_gate_regime.py
          ↓
[B]    Backtest EnsembleBot — ha de superar HoldBot
          ↓
[D]    Migrar a servidor 24/7
          ↓
[E]    🚀 Iniciar demo — 3-6 mesos de paper trading
```

Les tasques G–K es fan **en paral·lel** mentre la demo corre. No bloquegen l'inici.

---

## Tasques pendents

---

### A — Entrenar tots els models (pendent entrenament òptim)

Abans d'entrenar cal acabar l'optimització Optuna.

> ⚠️ **TFT (Temporal Fusion Transformer):** El model TFT NO ha estat optimitzat ni entrenat per la seva càrrega computacional extrema (estimació: 8-16h d'optimització Optuna + 8-16h d'entrenament, amb ~8 GB de RAM). Pendent de completar quan el temps disponible ho permeti. Tots els altres models ML estan optimitzats i entrenats. Veure `docs/03_ML_RL_MODELS.md §4.3` per als detalls.

**Recomanació de steps per a entrenament RL:**

| Agent | Steps recomanats | Justificació |
|-------|-----------------|-------------|
| PPO / SAC baseline (1H) | **1M–1.5M** | ~35k candles training → 500k = 14 passes; 1M = 28 passes, punt òptim |
| PPO / SAC on-chain (1H) | **1M–1.5M** | Igual que baseline; les features addicionals necessiten més training |
| PPO / SAC professional (12H) | **500k** | ~1.2k candles → ja 420 passes amb 500k; augmentar seria overfitting |
| TD3 professional / multiframe (12H) | **500k** | Igual que professional |

```bash
# Entrenar tot d'una (un cop l'Optuna hagi acabat)
python scripts/train_models.py          # tots els ML
python scripts/train_rl.py              # tots els RL (modifica total_timesteps als YAMLs primer)

# Gate System (pipeline propi, no usa train_models.py)
alembic upgrade head                    # crea gate_positions + gate_near_misses a la BD
python scripts/train_gate_regime.py     # HMM K=2..6 BIC + XGBoost Optuna → models/gate_*.pkl
```

---

### B — Backtest EnsembleBot (validació pre-demo)

EnsembleBot v1 ja implementat (`bots/classical/ensemble_bot.py`, política `majority_vote`).

Abans d'activar-lo a la demo, cal confirmar que supera HoldBot en backtest out-of-sample:

```bash
python scripts/run_comparison.py --bots hold trend mean_reversion momentum \
    ml_xgb ml_lgbm ml_rf rl_ppo rl_sac ensemble
```

Criteri mínim d'entrada a la demo: **Sharpe > 1.0 i Drawdown màx. < -25%** en backtest out-of-sample.

**Quins bots entren a l'ensemble:** edita `config/models/ensemble.yaml` (secció `sub_bots`). Descomenta els que hagin superat el criteri. Reinicia el DemoRunner per aplicar el canvi.

**Polítiques futures (mentre la demo corre):**

| Política | Quan implementar |
|---------|-----------------|
| `weighted` | Pes proporcional al Sharpe dels darrers N dies de cada sub-bot |
| `stacking` | Model ML de 2a capa entrenat sobre les prediccions dels sub-bots |

---

### D — Migrar a servidor 24/7

Per a la demo necessites un servidor que estigui sempre actiu. Opcions:

| Opció | Cost | RAM/CPU | Ideal per a |
|-------|------|---------|------------|
| **Oracle Cloud Free Tier** | **Gratuït per sempre** | 4 vCPU ARM · 24 GB RAM · 200 GB SSD | ✅ **Recomanat** — la millor relació preu/prestacions |
| Hetzner CX32 | ~10€/mes | 4 vCPU · 8 GB RAM · 80 GB SSD | Si prefereixes pagar per fiabilitat europea |
| Raspberry Pi 5 | ~100€ únic | 4 vCPU ARM · 8 GB RAM | Si tens internet fiable a casa |

**Oracle Cloud** és la recomanació clara: ARM Ampere A1 (mateixa arquitectura que Apple Silicon — tots els paquets Python/ML funcionen perfectament), 24 GB de RAM (sobrat per executar PostgreSQL + DemoRunner + inference), i és **gratuït per sempre**.

**Infraestructura mínima al servidor:**
```bash
# Cron jobs
0 * * * *   python scripts/update_data.py        # candles cada hora
0 8 * * *   python scripts/update_fear_greed.py  # F&G diari
0 */4 * * * python scripts/update_futures.py     # funding rate
0 6 * * *   python scripts/update_blockchain.py  # on-chain diari

# Servei systemd (restart automàtic)
/etc/systemd/system/btc-demo.service → python scripts/run_demo.py

# Backup
0 3 * * *   pg_dump btc_trading > /backups/btc_$(date +%Y%m%d).sql
```

---

### E — Inici de la demo

Un cop [A-D] completats:

```bash
python scripts/run_demo.py
```

**Objectiu:** mínim **3-6 mesos** de paper trading en temps real, tots els bots que hagin superat el criteri de backtest actius simultàniament, registre complet a la BD.

---

## En paral·lel mentre la demo corre

### G — Telegram millorat

- Resum diari: PnL per bot, comparativa vs BTC spot
- Alertes de drawdown configurables per bot (ex: alerta si drawdown > 10%)
- Rànquing setmanal
- Comandes interactives: `/status`, `/portfolio`, `/ranking`

### H — Dashboard Streamlit complet

Quan hi hagi mesos de dades reals:
- Portfolios en temps real per bot (gràfic PnL acumulat)
- Trade log filtrable per bot i data
- Comparativa vs BTC spot
- Ranking per Sharpe/Calmar dels darrers 30/90/180 dies

---

## Nous models i millores futures

### I — Nous models de ML/RL

| Model | Tipus | Prioritat | Notes |
|-------|-------|-----------|-------|
| EnsembleBot weighted | Meta | 🟢 **Alta** | Pes proporcional al Sharpe dels últims N dies — fàcil un cop la demo té dades |
| Gate System v2 | Gate | 🟢 **Alta** | Afegir shorts + news sentiment en P2 + exchange_netflow; un cop v1 valida el concepte |
| N-BEATS / N-HiTS | Deep Learning | 🟡 Mitjana | Arquitectures purament de forecasting, eficients sense RNN; sovint superen GRU |
| TabNet | Tabular | 🟡 Mitjana | Competitiu amb XGBoost, attention-based interpretatiu — bon reemplaçament o complement |
| BreakoutBot | Clàssic | 🟡 Mitjana | Pivot points + ATR per confirmar ruptures; complement natural a TrendBot |
| DreamerV3 | RL model-based | 🔴 Baixa | World model (RSSM) — projecte de recerca independent, massa complex per ara |

---

### J — Capa LLM per a sentiment de notícies ⭐ (Recomanat)

#### La proposta

Afegir una font de dades diària on un LLM analitza headlines de BTC i retorna un senyal de sentiment `[-1.0, 1.0]`. L'objectiu és capturar informació macro que els indicadors tècnics no detecten: aprovació d'ETFs, regulació, col·lapses d'exchanges, moviments institucionals.

#### Té sentit per a swing trading?

**Sí, i especialment per a swing trading.** Per a scalping o day trading, les notícies impacten en mil·lisegons i un LLM seria massa lent. Per a operacions de dies/setmanes (timeframe 1H–12H), el sentiment diari és predictiu: diverses recerques 2023–2025 mostren que senyals LLM basats en notícies milloren la predicció de BTC en horitzons de 24–72h (López-Lira & Tang 2023, Shen et al. 2024).

#### Arquitectura recomanada: LLM com a feature, no com a gatekeeper

```
CryptoPanic/RSS → LLM (GPT-4o-mini / Claude Haiku) → news_sentiment [-1, 1]
                                    ↓
                   BD: taula `news_sentiment` (timestamp, score, raw_text)
                                    ↓
              FeatureBuilder → columna `news_sentiment` en tots els models
                                    ↓
              ML models + RL professional la usen com a feature extra
```

**Per què feature i no gatekeeper?** Un gatekeeper LLM que "veta" els senyals dels altres bots eliminaria moltes oportunitats bones. Com a feature, els models aprenen ells mateixos quan el sentiment importa i quan no.

**Fonts de dades:** CryptoPanic API (gratuïta, headlines des de 2014), CoinDesk RSS, CoinTelegraph RSS.

**Freqüència:** una vegada/dia (suficient per swing trading). Cost: ~$0.01–$0.05 USD/dia.

#### Passos d'implementació

```
1. data/sources/news_sentiment.py     # Fetcher: headlines → LLM → score [-1,1]
2. core/db/models.py + alembic        # Nova taula: news_sentiment
3. data/processing/external.py        # Loader: retorna DataFrame UTC-indexed
4. FeatureBuilder                     # Integra news_sentiment com a feature diària
5. YAMLs dels models                  # features.external.news_sentiment: true
6. scripts/update_news_sentiment.py   # Cron diari (ex: 07:00h)
```

**Risks i mitigació:**

| Risc | Mitigació |
|------|-----------|
| LLM al·lucina / error API | Fallback a 0.0 (neutral) si l'API falla |
| Biaix de lookahead | Analitzar notícies del dia anterior, no del dia actual |
| Historial limitat | CryptoPanic té dades des de 2014 — suficient per entrenar |
| Cost | Cap preocupació: <$20/any |

**Prioritat recomanada:** 🟢 Alta. Implementar un cop la demo estigui en marxa i les primeres comparatives mostrin on fallen els models. El sentiment LLM pot capturar els "black swan" positius i negatius que cap indicador tècnic no detecta.

---

### K — Neteja de BD per a la demo

La BD de dades de mercat (OHLCV, fear_greed, etc.) s'ha de conservar íntegra.
El que cal netejar és l'estat de demo anterior (si n'hi ha):

```sql
-- Executar si vols iniciar la demo des de zero
TRUNCATE TABLE demo_ticks;   -- estat del portfolio anterior
TRUNCATE TABLE demo_trades;  -- historial de trades anteriors
-- Les taules de candles i dades externes NO s'han de tocar
```

*(Verifica els noms exactes de taules amb `\dt` a psql)*

---

## Resposta a la pregunta clau

> *"El desenvolupament de Telegram i Dashboard el puc fer mentre la demo ja vagi corrent?"*

**Sí, completament.** El DemoRunner és independent del Telegram i del Dashboard. Pots iniciar la demo amb el Telegram bàsic que ja tens i millorar-lo mentre les dades s'acumulen. El Dashboard és fins i tot millor fer-lo un cop tens mesos de dades reals.

**Ordre de prioritats real:**
1. Acabar optimització Optuna + entrenar models (ML/RL + Gate System)
2. Backtest EnsembleBot
3. Servidor Oracle + demo
4. LLM sentiment layer (mentre corre la demo) → candidat per Gate System v2 P2
5. Dashboard, Telegram millorat, nous models ML/RL, Gate System v2

---

*Última actualització: Març 2026 · Versió 5.0 (Gate System v1 marcat com a implementat)*
