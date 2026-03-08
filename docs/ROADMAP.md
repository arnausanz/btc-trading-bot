# Roadmap — Tasques Pendents i Visió de Futur

> **Regla d'or:** cap bot va a demo fins que en backtest out-of-sample superi HoldBot en Sharpe i Calmar.
> Per a l'arquitectura del sistema: veure **[PROJECT.md](./PROJECT.md)**.

---

## Estat actual dels components

| Component | Estat | Notes |
|-----------|-------|-------|
| PaperExchange | ✅ Operatiu | Fees 0.1% + slippage 0.01% |
| OHLCVFetcher (Binance/ccxt) | ✅ Operatiu | 1h, 4h, 1d des de 2019 |
| ObservationBuilder | ✅ Operatiu | Cache en memòria per tick |
| TrendBot | ✅ Operatiu | EMA crossover + RSI |
| DCABot | ✅ Operatiu | Compra periòdica fixa |
| GridBot | ✅ Operatiu | Bollinger Bands |
| HoldBot | ✅ Operatiu | Benchmark buy & hold |
| MLBot (RF, XGB, LGBM, CB, GRU, PatchTST) | ✅ Operatiu | 6 backends |
| RLBot (PPO, SAC) | ✅ Operatiu | Discret + continu |
| BacktestEngine + MLflow | ✅ Operatiu | Registre automàtic |
| BotComparator | ✅ Operatiu | Ranking per Sharpe |
| DemoRunner (multi-bot) | ✅ Operatiu | Persistència + Telegram |
| TelegramNotifier | ✅ Operatiu | Trades, status horari, drawdown |
| Tests (smoke + unit) | ✅ 123 tests | Cobertura bàsica sense BD |
| Optimize workflow | ✅ Operatiu | Auto-carrega `_optimized.yaml` |
| Walk-forward split | ✅ Operatiu | TRAIN_UNTIL / TEST_FROM al settings |
| Documentació | ✅ Completa | PROJECT, MODELS, DB, EXTENDING, CONFIG |
| Mètriques backtesting | ✅ Correctes | Sharpe/Calmar/WinRate correctament implementats |
| Dashboard (Streamlit) | ⚠️ Bàsic | Només preus de la BD |
| Tests d'integració | ⚠️ Esquelet | 5 tests, necessiten BD real |
| Dades externes (Fear&Greed + on-chain) | ✅ Operatiu | fear_greed, funding_rates, open_interest, blockchain_metrics |
| EnsembleBot | ❌ Pendent | Meta-capa de combinació |
| Feature Store | ❌ Pendent | Placeholder reservat |
| Risk Manager | ❌ Pendent | Placeholder reservat |

---

## Tasques pendents

---

### ✅ B — Dades externes (implementat)

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

**Pendent d'aquesta fase:** integrar les noves features al `DatasetBuilder` i `ObservationBuilder` per que els models les puguin usar en entrenament.

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

#### C1. Bots clàssics pendents

| Bot | Lògica | Estat |
|-----|--------|-------|
| MeanReversionBot | RSI extrems (<20/>80) + Z-score del preu vs. mitjana | ❌ Pendent |
| MomentumBot | Rate of Change (ROC) + confirmació de volum | ❌ Pendent |
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

#### C3. Nous agents RL — investigació profunda

**El problema actual:** PPO i SAC s'han implementat ràpidament amb rewards simples (retorn directe). Però un trader professional no pren decisions basant-se únicament en el retorn instantani.

**Objectiu d'aquesta fase:** fer una investigació profunda sobre com treballa un trader professional — quins factors considera, com gestiona el risc, quan entra i surt — i traduir aquesta mentalitat en una política RL ben definida.

**Preguntes a respondre durant la recerca:**
- Com gestiona el risc un trader professional? (stop-loss, position sizing, correlació d'actius)
- Quina és la diferència entre momentum trading, swing trading i scalping en termes de senyals d'entrada?
- Com es defineix "una bona entrada"? (confirmació múltiple vs. senyal ràpid)
- Com s'aplica el concepte de "regime detection" (tendència vs. rang vs. alta volatilitat)?
- Quins indicadors usen els traders professionals vs. els que usen els bots? On hi ha divergència?

**Elements clau per a la política RL:**

| Element | Descripció |
|---------|-----------|
| Reward shaping | Penalitzar drawdown (calmar-based), overtrading (cost per operació), inactivitat excessiva |
| Position sizing | L'agent controla el sizing [0.0–1.0], no sempre tot el capital |
| Regime detection | L'entorn informa l'agent del règim de mercat actual |
| Multi-timeframe obs | L'observation inclou 1h + 4h + 1d simultàniament |
| Portfolio state | USDT balance, BTC balance, PnL latent com a part de l'observation |
| Stop-loss implícit | Reward molt negatiu per drawdowns superiors a X% |

**Nous agents a investigar i implementar:**

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
[B ✅] Fear & Greed + on-chain (dades a la BD, scripts operatius)
              ↓
[B_pendent] Integrar features al DatasetBuilder + re-entrenar models
              ↓
[C1] MeanReversionBot + MomentumBot ← clàssics, ràpids
              ↓
[C3] Investigació RL professional → nova política → TD3 / Curriculum
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

*Última actualització: Març 2026 · Versió 1.2*
