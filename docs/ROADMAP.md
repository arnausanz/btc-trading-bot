# Roadmap i Anàlisi del Projecte
**Versió:** 1.0 — Març 2026

---

## PART 1 — Mapa del Projecte Actual

### Què és i per a què serveix

Plataforma de trading algorísmic en **paper trading** (diners virtuals) per a BTC/USDT. L'objectiu és provar múltiples estratègies en paral·lel durant mesos, comparar-les rigorosament, i quan n'hi hagi prou evidència de quines funcionen, portar-les a producció real.

---

### Diagrama de capes

```
┌────────────────────────────────────────────────────────┐
│  SCRIPTS (punts d'entrada manual)                      │
│  run_demo.py · run_comparison.py · train_models.py...  │
└──────────────────────┬─────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────┐           ┌──────────────────┐
│  DEMO RUNNER  │           │  BACKTEST ENGINE │
│  (temps real) │           │  (simulació)     │
└───────┬───────┘           └────────┬─────────┘
        │                            │
        └──────────┬─────────────────┘
                   ▼
        ┌──────────────────────┐
        │  BOTS (estratègies)  │
        │  ├── Classical       │
        │  │   ├── TrendBot    │
        │  │   ├── DCABot      │
        │  │   ├── GridBot     │
        │  │   └── HoldBot     │
        │  ├── ML (MLBot)      │
        │  │   ├── RandomForest│
        │  │   ├── XGBoost     │
        │  │   ├── LightGBM    │
        │  │   ├── CatBoost    │
        │  │   ├── GRU         │
        │  │   └── PatchTST    │
        │  └── RL (RLBot)      │
        │      ├── PPO         │
        │      └── SAC         │
        └──────────┬───────────┘
                   │ observation_schema()
                   ▼
        ┌──────────────────────┐
        │  DATA LAYER          │
        │  ObservationBuilder  │
        │  → compute_features  │
        │  → DB (candles)      │
        └──────────────────────┘
```

---

### El flux pas a pas — com funciona un tick

```
1. [Font de dades]
   Binance API (ccxt) → candeles OHLCV → PostgreSQL (taula candles)

2. [Features]
   candles DB → compute_features() → DataFrame amb:
   EMA, RSI, MACD, Bollinger Bands, ATR
   (es guarden en memòria, no a la DB)

3. [Tick del Demo Runner — cada 60 s]
   Binance API → preu actual
       ↓
   Per a cada bot:
     ObservationBuilder.build() → finestra de N candles + features
         ↓
     bot.on_observation(obs) → Signal (BUY / SELL / HOLD)
         ↓
     PaperExchange.send_order(signal) → Order (amb fees + slippage simulat)
         ↓
     DemoRepository.save_tick() + save_trade() → PostgreSQL
         ↓
     TelegramNotifier (si hi ha trade, o cada hora per status)

4. [Backtesting — offline]
   Mateix flux però sobre TOTES les candles històriques
   → BacktestMetrics (Sharpe, Drawdown, Win Rate, Calmar)
   → MLflow (registre automàtic de paràmetres i mètriques)
```

---

### Estat actual de cada component

| Component | Estat | Notes |
|---|---|---|
| PaperExchange | ✅ Operatiu | Fees + slippage simulats |
| OHLCVFetcher | ✅ Operatiu | Binance via ccxt |
| ObservationBuilder | ✅ Operatiu | Cache en memòria |
| TrendBot | ✅ Operatiu | EMA crossover + RSI |
| DCABot | ✅ Operatiu | Compra periòdica |
| GridBot | ✅ Operatiu | Bollinger Bands |
| HoldBot | ✅ Operatiu | Benchmark passiu |
| MLBot | ✅ Operatiu | RF, XGB, LGB, CB, GRU, PatchTST |
| RLBot | ✅ Operatiu | PPO, SAC |
| BacktestEngine | ✅ Operatiu | + MLflow |
| BotComparator | ✅ Operatiu | Ranking per Sharpe |
| DemoRunner | ✅ Operatiu | Multi-bot simultani |
| DemoRepository | ✅ Operatiu | Persistència ticks + trades |
| TelegramNotifier | ✅ Operatiu | Status, trades, drawdown, resum diari |
| Dashboard | ⚠️ Bàsic | Només mostra preus de la BD |
| Feature Store | ❌ Buit | Placeholder reservat |
| Risk Manager | ❌ Buit | Placeholder reservat |
| Tests integració | ❌ Buits | Només __init__.py |
| Dades externes | ❌ No existeix | Fear&Greed, sentiment, on-chain |

---

## PART 2 — Anàlisi d'Escalabilitat

### Què funciona bé avui i escalarà sense problemes

- **Afegir un bot clàssic nou** → OK. Herència de BaseBot, YAML de config, registre al demo.yaml i _load_bots().
- **Afegir un model ML nou** → OK. `_MODEL_REGISTRY` al MLBot, mètodes train/predict/save/load.
- **Afegir un agent RL nou** → OK. `_AGENT_REGISTRY` al RLBot.
- **Canviar el símbol (BTC→ETH)** → OK. Tot configurable via YAML.
- **Afegir un timeframe nou** → OK. ObservationBuilder el detecta automàticament.
- **Afegir un indicador tècnic nou** → OK. `TechnicalIndicators` + `compute_features()`.

---

### Limitacions estructurals actuals — on s'ha de treballar

#### 2.1 Dades externes (Fear&Greed, Sentiment, On-chain)

**Problema actual:** L'`ObservationSchema` té un camp `extras: dict` però l'`ObservationBuilder` no el processa. No hi ha cap mecanisme per afegir fonts de dades que no siguin OHLCV.

**Impacte:** Si vols afegir Fear&Greed Index, Twitter sentiment o on-chain data (SOPR, exchange flows), no hi ha lloc on posar-ho. El bot les rebria però el builder no les construiria.

**Solució futura:** Crear `DataSourceRegistry` — un sistema d'extensions on cada font de dades externa s'integra com un mòdul independent que l'ObservationBuilder sap invocar. La DB necessita taules noves per a cada font externa (o una taula genèrica `external_signals`).

---

#### 2.2 Feature Store — càlculs duplicats

**Problema actual:** Cada bot calcula les seves pròpies features. TrendBot recalcula les EMAs a cada tick. GridBot recalcula les Bollinger Bands a cada tick. Si tens 8 bots actius tots sobre les mateixes candles, calcules les mateixes coses 8 vegades.

**Impacte:** Ineficiència de CPU, però suportable amb pocs bots. El problema real és que no hi ha un "lloc central de veritat" per a les features — si un bot calcula l'EMA-50 d'una manera i un altre d'una altra, els resultats podrien diferir subtilment.

**Solució futura:** El Feature Store (ja reservat) ha de pre-calcular TOTES les features una sola vegada i deixar-les a memòria/DB. Els bots consulten, no calculen.

---

#### 2.3 Entrenament RL — Environments rígids

**Problema actual:** `BtcTradingEnvDiscrete` i `BtcTradingEnvContinuous` estan dissenyats per a OHLCV + indicadors tècnics numèrics. Si vols afegir dades qualitatives (sentiment score, Fear&Greed) a l'observation space, has de modificar l'entorn directament.

**Impacte:** Dificultat per experimentar amb entorns que incorporin dades externes. L'`observation_space` és hardcoded a `(lookback * n_features,)`.

**Solució futura:** Separar el "que entra a l'entorn" de "com es construeix". Un `ObservationComposer` flexible que permeti combinar OHLCV + externos + portfolio state en un sol vector normalitzat.

---

#### 2.4 Ensembles / Meta-bots

**Problema actual:** No hi ha cap mecanisme per combinar senyals de múltiples bots. Cada bot opera de forma completament independent. Si el TrendBot, el MLBot i el RLBot coincideixen en un BUY, no hi ha res que detecti aquesta convergència.

**Impacte:** Pèrdua d'una de les estratègies més potents: el voting/stacking entre models.

**Solució futura:** `EnsembleBot` — un bot que és "contenidor" d'altres bots. Rep els senyals de cada sub-bot i aplica una política de combinació (majority vote, weighted average, unanimous only, etc.).

---

#### 2.5 Mètriques de backtesting incorrectes

**Problema actual crític:**
- **Sharpe Ratio**: usa `sqrt(365)` però els ticks son horaris → hauria de ser `sqrt(365*24)`. El Sharpe actual és ~5x sobreestimat.
- **Calmar Ratio**: usa `len(df)` com a dies → amb dades horàries, el retorn anualitzat és completament incorrecte.
- **Win Rate**: no mesura round-trips (buy→sell) sinó canvis positius entre qualsevol tick amb ordre filled.

**Impacte:** Tots els resultats de backtest que tens ara són **no comparables** amb benchmarks estàndard del sector. No pots fiar-te del Sharpe actual per prendre decisions.

---

#### 2.6 Sincronització Demo Runner — candles tancades

**Problema actual:** El Demo Runner s'executa cada 60 segons i el bot actua sobre el preu actual, però les seves features estan basades en candles horàries tancades. Resultats:
- El bot pot generar fins a 60 senyals per hora sobre la mateixa candle no tancada.
- La candle actual (en construcció) no es pot afegir a la cache, per tant el bot sempre veu les dades de l'hora anterior.

**Impacte:** En estratègies d'1h com el TrendBot, és acceptable (actua 1-2 cops per hora com a màxim si hi ha senyal). En estratègies més sensibles podria generar overtrading.

---

## PART 3 — Definició d'Estratègies: Estat de l'Art

### Filosofia: tenir 5-10 bots amb arquitectures molt diverses

L'objectiu no és que tots siguin millors — és que **no estiguin correlats**. Un bot que guanya quan cau el mercat complementa un bot de tendència. L'ensemble dels 5-10 millors és més robust que qualsevol individual.

---

### 3.1 Classical (Baseline + Benchmark)

| Bot | Ja existeix? | Notes |
|---|---|---|
| HoldBot (Buy&Hold) | ✅ | Benchmark mínim que tot ha de superar |
| DCABot | ✅ | Benchmark acumulació |
| TrendBot (EMA crossover) | ✅ | Clàssic, cal optimitzar paràmetres |
| GridBot (Bollinger) | ✅ | Cal afinar el grid real (múltiples nivells) |
| **MeanReversionBot** | ❌ Pendent | RSI extrems + Z-score. Alta freqüència relativa |
| **MomentumBot** | ❌ Pendent | ROC + volum. Diferent filosofia del Trend |
| **BreakoutBot** | ❌ Pendent | Suports/resistències, ATR per confirmar |

---

### 3.2 ML Supervisat — Millores des de l'estat de l'art

**Models actuals:** Random Forest, XGBoost, LightGBM, CatBoost, GRU, PatchTST.

**Problemes a resoldre primer:**
1. Label engineering: ara el target és "puja >1% en 24h". Cal provar múltiples horitzons (4h, 12h, 48h, 168h) i múltiples thresholds.
2. Feature leakage: cal assegurar que no hi ha cap feature calculada amb dades futures.
3. Walk-forward validation: en lloc de train/test split simple, usar una finestra lliscant que simuli el desplegament real.

**Millores per incorporar:**

| Millora | Prioritat | Descripció |
|---|---|---|
| Walk-forward CV | 🔴 Alta | TimeSeriesSplit o expanding window, no random split |
| Features de microestructura | 🟡 Mitjana | Volume imbalance, spread simulat, tick direction |
| Features multi-timeframe | 🟡 Mitjana | Combinar 1h + 4h + 1d en el mateix feature vector |
| Sentiment extern | 🟢 Baixa | Fear&Greed Index com a feature numèrica |
| On-chain features | 🟢 Baixa | Exchange inflows, SOPR, MVRV |

**Models nous a considerar:**
- **Temporal Fusion Transformer (TFT)**: Estat de l'art per a séries temporals financeres. Millor que LSTM/GRU per incorporar features de múltiples horizons i covarites externes.
- **N-BEATS / N-HiTS**: Arquitectures pures de séries temporals sense RNN, molt eficients.
- **TabNet**: Per als models tabulars clàssics, supera XGBoost en alguns benchmarks i és més interpretable.

---

### 3.3 Reinforcement Learning — Millores

**Estat actual:** PPO (discret) + SAC (continu). Entrenats amb simples retorns com a reward.

**Problemes actuals:**
1. L'entorn discret (PPO) inverteix el 100% de USDT en cada BUY. Massa agressiu.
2. Inconsistència entre l'entorn d'entrenament i la inferència al RLBot.
3. Sense mecanisme de position sizing dinàmic a l'agent discret.

**Millores per a RL:**

| Millora | Prioritat | Descripció |
|---|---|---|
| Reward shaping avançat | 🔴 Alta | Reward que penalitzi drawdown i trading excessiu (calmar-based) |
| Position sizing continu | 🔴 Alta | Que el SAC controli el sizing, no el DemoRunner |
| Multi-step returns (n-step) | 🟡 Mitjana | Millora la propagació de recompenses |
| Curriculum learning | 🟡 Mitjana | Entrenar primer en períodes de tendència clara, despres en rangs |
| Multi-asset RL | 🟢 Baixa | Un sol agent que gestioni BTC + ETH simultàniament |

**Nous agents a considerar:**
- **TD3** (Twin Delayed DDPG): Millor que SAC per a entorns continus financers amb molt soroll.
- **DreamerV3** / **MBPO**: Model-based RL. Aprèn un model del mercat i simula internament — molt mostres-eficient.
- **Alpha (Transformer-based RL)**: Atenció multi-cap sobre la sèrie temporal com a política.

---

### 3.4 Ensemble / Meta-capa — L'estratègia més potent

L'ensemble és on es pot guanyar molt sense gaire risc, simplement combinant el que ja tenim:

| Tipus d'ensemble | Com funciona |
|---|---|
| **Majority Vote** | Si >50% de bots diuen BUY → BUY |
| **Weighted Vote** | Pes proporcional al Sharpe dels últims N dies |
| **Unanimous** | Només actua si TOTS els bots coincideixen (molt conservador) |
| **Stacking (meta-model)** | Un model ML de 2a capa entrena sobre les prediccions dels bots base |
| **Dynamic Routing** | Selecciona automàticament el millor bot per règim de mercat (tendència, rang, alta volatilitat) |

---

### 3.5 Finestres temporals recomanades

No usar una sola finestra — la diversitat de timeframes és una font de diversificació:

| Timeframe | Tipus d'estratègia adequada |
|---|---|
| 1h | Momentum, Trend, ML de curt termini |
| 4h | Swing trading, ML de mig termini |
| 1d | Macro-tendència, DCA, RL (menys soroll) |
| Multi (1h+4h+1d) | TFT, ensemble, meta-models |

---

## PART 4 — Millores Estructurals Prioritzades

Ordre recomanat d'implementació:

### 4.1 Primer: Corregir les mètriques [CRÍTIC]
Sense mètriques correctes, tots els backtests i comparacions son inútils.
- Sharpe Ratio: `sqrt(252)` per a dades diàries, `sqrt(252*24)` per a dades horàries
- Calmar Ratio: calcular dies reals, no len(df)
- Win Rate: round-trips reals (buy→sell), no ticks amb ordre filled
- Afegir: Sortino Ratio, Profit Factor, Average Trade, Max Consecutive Losses

### 4.2 Segon: Walk-forward backtesting
El backtesting actual fa un sol split. Per a una validació honest:
- TimeSeriesSplit: N folds lliscants sobre el temps
- Out-of-sample obligatori: les últimes N setmanes mai s'utilitzen per a optimització
- Benchmark sempre inclòs: HoldBot s'executa automàticament en cada backtest

### 4.3 Tercer: Corregir el Demo Runner — sincronització de candles
El bot ha d'actuar quan tanca una candle, no cada minut. Implementar detecció de "nova candle tancada" i actualitzar la cache.

### 4.4 Quart: Restaurar estat intern dels bots en reinici
`_in_position` i `_tick_count` s'han de persistir a la DB i restaurar en reiniciar el DemoRunner.

### 4.5 Cinquè: DataSource abstraction — per a dades externes
Crear la infraestructura per afegir Fear&Greed, sentiment, etc. com a features.

### 4.6 Sisè: EnsembleBot
Crear la capa de meta-bot que combina senyals.

### 4.7 Setè: Tests complets
- Tests unitaris per a totes les mètriques
- Tests d'integració amb DB real (fixture automàtica)
- Backtests de referència guardats (regression tests: el bot X sobre el període Y ha de donar Z Sharpe)

### 4.8 Vuitè: Dashboard complet
Quan ja hi hagi dades de demo reals i de qualitat, construir el dashboard que mostri:
- Portfolios en temps real per bot
- PnL acumulat i diari
- Drawdowns visuals
- Trade log amb filtre per bot
- Comparativa vs BTC spot (benchmark)
- Correlació entre bots (matriu de correlació de retorns)

---

## PART 5 — Pla d'acció seqüencial

### Fase A — Fonaments (prioritat màxima, fer ara)
1. Corregir mètriques de backtesting (Sharpe, Calmar, Win Rate)
2. Implementar walk-forward backtesting / TimeSeriesSplit
3. Re-executar tots els backtests amb les mètriques correctes
4. Establir benchmark clar: tot bot ha de superar HoldBot en Sharpe i Calmar

### Fase B — Nous models i recerca
5. Afegir MeanReversionBot i MomentumBot (simples, clàssics)
6. Implementar TFT (Temporal Fusion Transformer) com a model ML
7. Millorar RL: reward shaping + position sizing + curriculum
8. Afegir Fear&Greed Index com a feature (API gratuïta, simple)
9. Fer una cerca exhaustiva de hiperparàmetres (Optuna) per a tots els models

### Fase C — Ensemble i validació
10. Implementar EnsembleBot (majority vote + weighted)
11. Seleccionar els 5-8 millors bots (diversitat obligatòria: no 5 models ML similars)
12. Backtesting de l'ensemble sobre el período out-of-sample (mai vist durant optimització)
13. Establir criteris clars de "prou bo per a demo": Sharpe > 1.5, Drawdown < 20%, supera HoldBot

### Fase D — Robustesa del sistema
14. Corregir sincronització candles al DemoRunner
15. Persistir _in_position i _tick_count a la DB
16. Implementar tests complets (unitaris + integració)
17. Configurar TimescaleDB compression i política de retenció de demo_ticks
18. Logging estructurat (JSON) per facilitar l'anàlisi posterior

### Fase E — Demo 24/7
19. Desplegar a servidor (Oracle Cloud Free Tier recomanat)
20. Cron per update-data cada hora
21. Systemd per restart automàtic del DemoRunner
22. Alertes de Telegram per caiguda del sistema (watchdog extern)

### Fase F — Dashboard i anàlisi
23. Dashboard complet amb portfolios, PnL, drawdowns, correlació entre bots
24. Exportació de dades a CSV/Excel per a anàlisi offline
25. Comparativa visual entre bots i vs BTC spot

---

## Resum visual de prioritats

```
ARA ──────────────────────────────────────────► FUTUR

[A] Mètriques correctes
    └─► [A] Walk-forward backtesting
            └─► [B] Nous models (TFT, MeanReversion)
                └─► [B] Fear&Greed feature
                    └─► [C] EnsembleBot
                        └─► [C] Validació out-of-sample
                            └─► [D] Demo robusta
                                └─► [E] Desplegament
                                    └─► [F] Dashboard
```

**La regla d'or:** No posar res en demo fins que en backtest out-of-sample superi HoldBot en Sharpe (ajustat correctament) i en Calmar Ratio, i que l'ensemble tingui correlació baixa entre components.
