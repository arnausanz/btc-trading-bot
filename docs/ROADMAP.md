# Roadmap — Tasques Pendents i Visió de Futur

> **Regla d'or:** cap bot va a demo fins que en backtest out-of-sample superi HoldBot en Sharpe i Calmar.
> Per a l'arquitectura del sistema: veure **[PROJECT.md](./PROJECT.md)**.

---

## Estat actual dels components

| Component | Estat | Notes |
|-----------|-------|-------|
| PaperExchange | ✅ Operatiu | Fees 0.1% + slippage 0.01% |
| OHLCVFetcher (Binance/ccxt) | ✅ Operatiu | 1h, 4h, 12h, 1d des de 2019 |
| ObservationBuilder | ✅ Operatiu | Cache en memòria per tick |
| TrendBot | ✅ Operatiu | EMA crossover + RSI |
| DCABot | ✅ Operatiu | Compra periòdica fixa |
| GridBot | ✅ Operatiu | Bollinger Bands |
| HoldBot | ✅ Operatiu | Benchmark buy & hold |
| MeanReversionBot | ✅ Operatiu | Z-score + RSI extrem + filtre volum |
| MomentumBot | ✅ Operatiu | ROC + volum confirmat + MACD |
| MLBot (RF, XGB, LGBM, CB, GRU, PatchTST) | ✅ Operatiu | 6 backends |
| RLBot (PPO, SAC) | ✅ Operatiu | Discret + continu |
| RLBot on-chain (PPO, SAC) | ⚠️ Pendent entrenament | Configs creades, requereix BD externa + `train_rl.py --agents ppo_onchain sac_onchain` |
| RLBot Professional (PPO, SAC) | ⚠️ Pendent entrenament complet | Política professional implementada (12H, Discrete-5/Continuous, ATR stop, position state, reward professional). Smoke OK. Training complet + Optuna pendent. |
| BacktestEngine + MLflow | ✅ Operatiu | Registre automàtic |
| BotComparator | ✅ Operatiu | Ranking per Sharpe |
| DemoRunner (multi-bot) | ✅ Operatiu | Persistència + Telegram |
| TelegramNotifier | ✅ Operatiu | Trades, status horari, drawdown |
| Tests (smoke + unit) | ✅ 123 tests | Cobertura bàsica sense BD |
| Optimize workflow | ✅ Operatiu | Auto-carrega `_optimized.yaml` |
| Config YAML unificat | ✅ Operatiu | Un YAML per model (classic/ML/RL), search space integrat |
| Walk-forward split | ✅ Operatiu | TRAIN_UNTIL / TEST_FROM al settings |
| Documentació | ✅ Completa | PROJECT, MODELS, DB, EXTENDING, CONFIG (v2.0 YAML unificat) |
| Mètriques backtesting | ✅ Correctes | Sharpe/Calmar/WinRate correctament implementats |
| Dashboard (Streamlit) | ⚠️ Bàsic | Només preus de la BD |
| Tests d'integració | ⚠️ Esquelet | 5 tests, necessiten BD real |
| Dades externes (Fear&Greed + on-chain) | ✅ Operatiu | fear_greed, funding_rates, open_interest, blockchain_metrics — integrat a FeatureBuilder/DatasetBuilder/ObservationBuilder |
| EnsembleBot | ❌ Pendent | Meta-capa de combinació |
| Feature Store | ❌ Pendent | Placeholder reservat |
| Risk Manager | ❌ Pendent | Placeholder reservat |

---

## Tasques pendents

---

### ✅ B — Dades externes (completament implementat)

Fonts implementades i operatives. Scripts de descàrrega inicial + update incremental per a cron.

| Font | Taula BD | Cobertura | Script cron |
|------|----------|-----------|-------------|
| Fear & Greed Index | `fear_greed` | Des de feb. 2018, diari | `update_fear_greed.py` (diari) |
| Funding rates | `funding_rates` | Des de set. 2019, cada 8h | `update_futures.py` (horari) |
| Open Interest (Vision S3) | `open_interest` (5m) | Des de des. 2021, cada 5min | `update_binance_vision.py` (diari) |
| Open Interest (REST) | `open_interest` (1h) | Darrers 30 dies, cada 1h | `update_futures.py` (horari) |
| Hash rate | `blockchain_metrics` | Des de ~2009, diari | `update_blockchain.py` (diari) |
| Adreces actives | `blockchain_metrics` | Des de ~2009, diari | `update_blockchain.py` (diari) |
| Transaction fees | `blockchain_metrics` | Des de ~2009, diari | `update_blockchain.py` (diari) |

**Comprovació de completesa:** `python scripts/check_data_completeness.py`

✅ **Integració al pipeline:** `FeatureBuilder`, `DatasetBuilder` i `ObservationBuilder` ja suporten totes les fonts externes via config YAML. No cal tocar codi — s'activen amb `features.external` al YAML del model. Config de referència: `config/models/ppo_onchain.yaml`.

---

#### B_pendent. Anàlisi de notícies i sentiment (NLP)
La font més complexa però potencialment la més poderosa per anticipar moviments.

**Opcions per obtenir el senyal:**
- **Claude API:** enviar titulars/articles i demanar un score de sentiment BTC (-1.0 a 1.0) + classificació de rellevància. Flexible i d'alta qualitat, però té cost per token.
- **BERT/FinBERT fine-tuned:** model local entrenat en notícies financeres. Gratuït en inferència, però cal mantenir-lo.
- **CryptoPanic API:** agrega notícies de cripto amb votes (bullish/bearish) de la comunitat. Té pla gratuït.

**Fonts de notícies candidates:** CryptoPanic, CoinDesk RSS, Reuters cripto, tweets de comptes influents (X API).

**Historial per entrenament:** CryptoPanic té historial via API. Per a sentiment retroactiu amb Claude/BERT, caldria processar un arxiu de titulars antics.

**Estructura proposada:**
```
data/sources/
├── fear_greed.py        # B1
├── onchain.py           # B2
└── news_sentiment.py    # B3 (fetch + score via Claude API o FinBERT)
```

**Taula BD genèrica per a tot extern:**
```sql
CREATE TABLE external_signals (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),        -- 'fear_greed', 'sopr', 'news_sentiment', etc.
    timestamp TIMESTAMPTZ,
    value FLOAT,               -- el valor numèric principal
    metadata JSON              -- dades extra (classificació, text, etc.)
);
```

---

### 🟡 C — Nous bots

Cada bot nou segueix el cicle complet: recerca → implementació → configuració → optimització Optuna → backtest → (si supera HoldBot) → demo.

#### ✅ C1. Bots clàssics (implementats)

MeanReversionBot i MomentumBot implementats seguint estat de l'art per BTC.
**Pas següent:** executar backtest, optimitzar amb Optuna i, si superen HoldBot, activar al demo.

```bash
# Optimitzar:
python scripts/optimize_bots.py --bots mean_reversion momentum

# Comparar amb la resta de bots:
python scripts/run_comparison.py --bots hold mean_reversion momentum trend grid
```

| Bot | Lògica | Estat |
|-----|--------|-------|
| MeanReversionBot | Z-score del preu + RSI extrem + filtre de volum | ✅ Implementat — pendent backtest + Optuna |
| MomentumBot | Rate of Change (ROC) + volum confirmat + MACD | ✅ Implementat — pendent backtest + Optuna |
| BreakoutBot | Suports/resistències (pivot points) + ATR per confirmar | ❌ Pendent |

---

#### C2. Nous models ML

Cada model nou necessita: implementació `BaseMLModel`, entrada al `_MODEL_REGISTRY`, config d'entrenament + Optuna, backtest complet.

| Model | Tipus | Notes |
|-------|-------|-------|
| Temporal Fusion Transformer (TFT) | Deep Learning | Estat de l'art per séries temporals financeres amb covarites externes; ideal per incorporar dades de B1/B2/B3 |
| N-BEATS / N-HiTS | Deep Learning | Arquitectures pures sense RNN; molt eficients |
| TabNet | Tabular | Competitiu amb XGBoost, interpretable via atenció |

---

#### ✅ C_onchain. PPO + SAC amb features on-chain (implementat, pendent entrenament)

Variants on-chain dels agents RL baseline. Mateixa política i reward que els baseline, però amb 5 features addicionals: Fear & Greed, funding rate, Open Interest (1h) i hash rate.

| Agent | Config | obs_shape | Estat |
|-------|--------|-----------|-------|
| PPO on-chain | `config/models/ppo_onchain.yaml` | 14×96=1344 | ✅ Config validada — pendent entrenament |
| SAC on-chain | `config/models/sac_onchain.yaml` | 14×96=1344 | ✅ Config creada — pendent entrenament |

```bash
# Entrenar (30-90 min per agent):
python scripts/train_rl.py --agents ppo_onchain sac_onchain

# Optimitzar primer (opcional, ~3h):
python scripts/optimize_models.py --agents ppo_onchain sac_onchain

# Comparar amb baseline:
python scripts/run_comparison.py --bots hold ppo sac ppo_onchain sac_onchain
```

**Pre-requisit:** dades externes a la BD (`python scripts/download_fear_greed.py`, `download_blockchain.py`, `update_futures.py`).

---

#### ✅ C3. Investigació RL professional + nova política (implementat, pendent entrenament complet)

**Investigació realitzada:** `docs/decisions/trading_policy_reference.md` — recerca profunda sobre gestió de risc professional, regime detection, position sizing, stop-loss i reward shaping per a BTC swing trading sense apalancament. Document de disseny final: `docs/nova_politica/DISSENY.md` (v2.0).

**Política implementada:** `ppo_professional` + `sac_professional` (timeframe 12H, cicles de 5–7 dies).

| Element del pla | Implementació |
|-----------------|--------------|
| Reward shaping (drawdown + overtrading) | ✅ `professional_risk_adjusted` — quadràtic progressiu + penalització per trades prematurs |
| Position sizing [0.0–1.0] | ✅ SAC continuous Box([0,1]) + deadband ±0.05; PPO Discrete-5 (HOLD/BUY_FULL/BUY_PARTIAL/SELL_PARTIAL/SELL_FULL) |
| Regime detection | ✅ ADX-14 (força tendència) + vol_ratio = ATR-5/ATR-14 (expansió/compressió) en observació |
| Portfolio state en observació | ✅ 4 dims afegides: pnl_pct, position_fraction, steps_in_position, drawdown_pct |
| Stop-loss | ✅ Stop ATR dur forçat per l'entorn (agent no pot moure'l); `stop_atr_multiplier=2.0` |
| Multi-timeframe obs | 🔄 Substituït per 12H × 90 lookback (45 dies) — menys soroll, millor per a swing |

**Fitxers nous:** `bots/rl/environment_professional.py`, `bots/rl/rewards/professional.py`, `config/models/ppo_professional.yaml`, `config/models/sac_professional.yaml`.

**Estat:** Smoke OK (-8.7% SAC, -39.7% PPO amb 50k steps aleatoris). Training complet (500k steps) + Optuna pendent.

```bash
# Optimitzar (en curs):
python scripts/optimize_models.py --agents ppo_professional sac_professional

# Entrenar amb millors hiperparàmetres:
python scripts/train_rl.py --agents ppo_professional sac_professional

# Comparar:
python scripts/run_comparison.py --bots hold ppo sac ppo_professional sac_professional
```

**Extensions futures de C3 (pendent):**

| Agent | Notes |
|-------|-------|
| TD3 (Twin Delayed DDPG) | Millor que SAC en entorns molt sorollosos; estable |
| Curriculum learning (PPO/SAC) | Entrenar en fases: tendències clares → rangs → alta volatilitat |
| DreamerV3 | Model-based RL; aprèn un model intern del mercat i simula internament — molt eficient en mostres |

---

### 🟡 D — EnsembleBot

La capa de meta-decisió: combinar els senyals de tots els bots per produir un sol senyal més robust.

**Més allà del propi bot, caldrà:**
- **Historial de prediccions:** guardar els senyals de cada bot a la BD per poder calcular pesos dinàmics
- **Mètriques en finestra lliscant:** calcular el Sharpe dels últims N dies per bot per ponderar el vote
- **Backtest de l'ensemble:** sistema per backtesta combinacions de bots (no un bot sol)
- **Selecció de components:** assegurar diversitat baixa correlació entre els bots seleccionats

**Polítiques a implementar (per ordre de complexitat):**

| Política | Descripció | Quan usar |
|---------|-----------|-----------|
| `majority_vote` | Si >50% de bots diuen BUY → BUY | Punt de partida |
| `weighted` | Pes proporcional al Sharpe dels últims N dies | Quan alguns bots son clarament millors |
| `unanimous` | Només actua si TOTS coincideixen | Molt conservador, poques operacions |
| `regime_routing` | Selecciona el millor bot per règim de mercat | Quan tenim detecció de règim |
| `stacking` | Model ML de 2a capa entrena sobre les prediccions | Quan tenim prou historial de predictions |

---

### 🟢 E — Millores del sistema

#### E1. Telegram millorat per al demo
Quan hi hagi mesos de dades de demo reals, millorar les notificacions:

- **Resum diari:** PnL de cada bot, millor/pitjor del dia, comparativa vs. BTC spot
- **Alertes de drawdown:** configurables per bot (ex: alerta si drawdown > 10%)
- **Rànquing setmanal:** quin bot va millor la setmana
- **Comandes interactives:** `/status`, `/portfolio`, `/trades`, `/ranking` via Telegram bot
- **Gràfics:** enviar una imatge amb el portfolio de cada bot (matplotlib inline)

---

#### E2. Correccions del DemoRunner

- **Sincronització de candles:** el bot ha d'actuar quan tanca una candle (1 cop/hora), no cada 60s sobre la mateixa candle oberta
- **Persistir estat intern:** `_in_position` i `_tick_count` han de persistir a la BD i restaurar-se en reiniciar

---

#### E3. Dashboard complet
Quan hi hagi mesos de dades de demo reals:

- Portfolios en temps real per bot (gràfic PnL acumulat)
- Drawdowns visuals i alertes configurables
- Trade log amb filtre per bot i data
- Comparativa vs. BTC spot (benchmark)
- Matriu de correlació de retorns entre bots
- Ranking per Sharpe + Calmar dels últims 30/90/180 dies

---

### 🔵 F — Desplegament i demo llarg

#### F1. Migrar a servidor
Explorar opcions (queda lluny, però cal tenir-ho en ment):

| Opció | Cost | Notes |
|-------|------|-------|
| Oracle Cloud Free Tier | Gratuït | 4 vCPU, 24GB RAM; suficient per ara |
| Hetzner Cloud CX22 | ~4€/mes | Millor latència a Europa |
| Raspberry Pi 4/5 | Hardware únic | Opció local, sense cost mensual |

**Infraestructura mínima per al desplegament:**
- Cron per `update_data.py` + `update_fear_greed.py` cada hora
- Systemd (o Docker) per restart automàtic del DemoRunner
- Backup automàtic de la BD (pg_dump diari)
- Watchdog extern amb alerta Telegram per caigudes

---

#### F2. Demo 24/7 durant mesos
L'objectiu final d'aquesta fase del projecte:

- Mínimament **3-6 mesos** de paper trading en temps real
- Tots els bots que hagin superat el criteri de backtest (Sharpe > 1.5, Drawdown < -20%, supera HoldBot)
- L'EnsembleBot actiu des del primer dia del demo llarg
- Registre complet a la BD de tots els ticks i trades
- Anàlisi mensual: ajustar pesos de l'ensemble si cal

---

## Seqüència recomanada

```
[B ✅] Fear & Greed + on-chain (dades, scripts, FeatureBuilder/DatasetBuilder/ObservationBuilder)
              ↓
[C1 ✅] MeanReversionBot + MomentumBot ← implementats, pendent backtest + Optuna
              ↓
[C_onchain ✅] PPO/SAC on-chain ← configs creades i validades, PENDENT ENTRENAMENT
              ↓
[C3 ✅] RL professional → ppo_professional + sac_professional (12H, ATR stop, position state, reward professional) — pendent entrenament complet + Optuna
              ↓
[B_pendent/NLP] NLP sentiment ← quan els models ML ja consumeixin les noves features
              ↓
[C2] TFT / N-BEATS ← quan les dades externes estiguin llestes
              ↓
[D] EnsembleBot (majority vote → weighted → stacking)
              ↓
[E1] Telegram millorat
              ↓
[F1] Migrar a servidor
              ↓
[F2] Demo 24/7 — 3-6 mesos
```

---

*Última actualització: Març 2026 · Versió 1.5*
