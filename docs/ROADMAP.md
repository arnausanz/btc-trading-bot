# Roadmap — Camí a la Demo 24/7

> **Regla d'or:** cap bot va a demo fins que en backtest out-of-sample superi HoldBot en Sharpe i Calmar.
>
> **Versió 7.0 — Març 2026** (reescrit post-auditoria — eliminades seccions completades D i K)

---

## Estat del sistema

Tot el sistema base està implementat i operatiu al servidor Oracle Cloud. L'únic pendent per iniciar la demo és completar el camí crític A→B→E i resoldre el deute tècnic DT1.

---

## Deute Tècnic — Resoldre ABANS de la demo

| # | Problema | Fitxer | Prioritat |
|---|---------|--------|----------|
| **DT1** | `CandleDB` sense `UniqueConstraint` — duplicats silenciosos possibles | `core/db/models.py:12` | 🔴 CRÍTIC |
| DT2 | N+1 queries a `OHLCVFetcher._save()` — molt lent per descàrregues grans | `data/sources/ohlcv.py:93` | 🟠 Alta |
| DT3 | Sense validació `obs_shape` al carregar model RL — crash silent en runtime | `bots/rl/rl_bot.py:60` | 🟠 Alta |
| DT4 | `win_rate()` usa `iterrows()` — O(n) Python loop, lent amb molts ticks | `core/backtesting/metrics.py:126` | 🟡 Mitjana |
| DT5 | ATR fallback a `ATR_REFERENCE` sense log warning | `bots/rl/environment_professional.py:158` | 🟡 Mitjana |

**Fix DT1 (concret):**
```bash
alembic revision -m "add_unique_constraint_candles"
# Dins la migració upgrade():
# 1. Eliminar duplicats: DELETE FROM candles WHERE id NOT IN (SELECT MIN(id) FROM candles GROUP BY exchange, symbol, timeframe, timestamp)
# 2. CREATE UNIQUE INDEX uq_candles ON candles (exchange, symbol, timeframe, timestamp)
```

---

## Camí crític cap a la demo

```
[DT1] Fix CandleDB UniqueConstraint + migració Alembic
          ↓
[ara]  Comparativa walk-forward + selecció de sub-bots per EnsembleBot
          ↓
[A]    Entrenar tots els models amb hiperparàmetres òptims
          (pendent del resultat de la comparativa — TFT possiblement descartat)
          + entrenar Gate System: alembic upgrade head + train_gate_regime.py
          ↓
[B]    Backtest EnsembleBot — ha de superar HoldBot
          (Sharpe > 1.0, Drawdown màx. < -25%, Calmar > 0.5 en test out-of-sample)
          ↓
[E]    🚀 Iniciar demo — 3-6 mesos de paper trading
          (commit + python scripts/run_demo.py al servidor)
```

---

### A — Entrenar tots els models (pendent validació comparativa)

L'optimització Optuna ja s'ha completat. Abans d'entrenar cal revisar els resultats de la comparativa walk-forward per decidir quins models val la pena reentrenar.

> ⚠️ **TFT:** No optimitzat ni entrenat per càrrega computacional extrema (~8-16h). Es valorarà descartar-lo si GRU/PatchTST ofereixen prou rendiment.

**Steps recomanats per a entrenament RL:**

| Agent | Steps recomanats | Justificació |
|-------|-----------------|-------------|
| PPO / SAC baseline (1H) | **1M–1.5M** | ~35k candles training → 1M = 28 passes |
| PPO / SAC on-chain (1H) | **1M–1.5M** | Features addicionals necessiten més training |
| PPO / SAC professional (12H) | **500k** | ~1.2k candles → 420 passes; augmentar seria overfitting |
| TD3 professional / multiframe (12H) | **500k** | Igual que professional |

```bash
python scripts/train_models.py          # tots els ML
python scripts/train_rl.py              # tots els RL
alembic upgrade head                    # crea gate_positions + gate_near_misses
python scripts/train_gate_regime.py     # HMM + XGBoost → models/gate_*.pkl
```

---

### B — Backtest EnsembleBot (validació pre-demo)

```bash
python scripts/run_comparison.py --bots hold trend mean_reversion momentum \
    ml_xgb ml_lgbm ml_rf rl_ppo rl_sac ensemble
```

Edita `config/models/ensemble.yaml` (`sub_bots`) per activar els que hagin superat el criteri.

**Polítiques futures (mentre la demo corre):**

| Política | Quan implementar |
|---------|-----------------|
| `weighted` | Pes proporcional al Sharpe dels darrers N dies de cada sub-bot |
| `stacking` | Model ML de 2a capa entrenat sobre les prediccions dels sub-bots |

---

### E — Inici de la demo

```bash
git add . && git commit -m "feat: ensemble bot + optimal models"
git push
# Al servidor:
git pull && python scripts/run_demo.py
```

**Objectiu:** mínim **3-6 mesos** de paper trading en temps real.

---

## En paral·lel mentre la demo corre

### G — Telegram millorat

- Resum diari: PnL per bot, comparativa vs BTC spot
- Alertes de drawdown configurables per bot (ex: alerta si drawdown > 10%)
- Rànquing setmanal
- Comandes interactives: `/status`, `/portfolio`, `/ranking`

---

### H — Dashboard Streamlit complet

Quan hi hagi mesos de dades reals:
- Portfolios en temps real per bot (gràfic PnL acumulat)
- Trade log filtrable per bot i data
- Comparativa vs BTC spot
- Ranking per Sharpe/Calmar dels darrers 30/90/180 dies

---

### I — Nous models de ML/RL

| Model | Tipus | Prioritat | Notes |
|-------|-------|-----------|-------|
| EnsembleBot weighted | Meta | 🟢 **Alta** | Pes proporcional al Sharpe dels últims N dies |
| Gate System v2 | Gate | 🟢 **Alta** | Shorts + news sentiment en P2 + exchange_netflow |
| N-BEATS / N-HiTS | Deep Learning | 🟡 Mitjana | Forecasting pur, eficient, sovint supera GRU |
| TabNet | Tabular | 🟡 Mitjana | Attention-based, bon complement a XGBoost |
| BreakoutBot | Clàssic | 🟡 Mitjana | Pivot points + ATR per confirmar ruptures |
| DreamerV3 | RL model-based | 🔴 Baixa | World model — massa complex per ara |

---

### J — Capa LLM per a sentiment de notícies ⭐ (Recomanat)

#### Arquitectura

```
CryptoPanic/RSS → LLM (GPT-4o-mini / Claude Haiku) → news_sentiment [-1, 1]
                                    ↓
                   BD: taula `news_sentiment` (timestamp, score, raw_text)
                                    ↓
              FeatureBuilder → columna `news_sentiment` en tots els models
```

#### Passos d'implementació

```
1. data/sources/news_sentiment.py     # Fetcher: headlines → LLM → score [-1,1]
2. core/db/models.py + alembic        # Nova taula: news_sentiment
3. data/processing/external.py        # Loader: retorna DataFrame UTC-indexed
4. FeatureBuilder                     # Integra news_sentiment com a feature diària
5. YAMLs dels models                  # features.external.news_sentiment: true
6. scripts/update_news_sentiment.py   # Cron diari (07:00 UTC)
```

**Costs i riscos:**

| Risc | Mitigació |
|------|-----------|
| LLM al·lucina / error API | Fallback a 0.0 (neutral) si l'API falla |
| Biaix de lookahead | Analitzar notícies del dia anterior, no del dia actual |
| Cost | <$20/any |

---

## Millores de robustesa (sense data límit)

| Item | Acció | Fitxer afectat |
|------|-------|---------------|
| **Model staleness** | Rolling Sharpe 30d per bot; alarma Telegram si cau < 0.5 | `core/engine/demo_runner.py` |
| **Refactoritzar `OHLCVFetcher._save()`** | Patró batch (com `FearGreedFetcher`) | `data/sources/ohlcv.py:85` |
| **Validar `obs_shape`** | Assert a `RLBot._load_agent()` | `bots/rl/rl_bot.py:60` |
| **Vectoritzar `win_rate()`** | Reemplaçar `iterrows()` | `core/backtesting/metrics.py:113` |
| **Warning ATR fallback** | LOG WARNING quan s'activa `ATR_REFERENCE` | `bots/rl/environment_professional.py:158` |

---

> **Nota sobre l'ordre de prioritats real:**
> 1. Resoldre DT1 (CandleDB)
> 2. Comparativa walk-forward → selecció → entrenament final
> 3. Backtest EnsembleBot
> 4. Commit + iniciar demo al servidor
> 5. LLM sentiment layer (mentre corre la demo)
> 6. Dashboard, Telegram millorat, nous models, Gate v2

---

*Última actualització: Març 2026 · Versió 7.0 (post-auditoria — D i K eliminats, deute tècnic afegit)*
