# Benvingut al BTC Trading Bot

> **Comença aquí.** Aquest document és el teu mapa del projecte. Explica com llegir la documentació, el glossari de termes, i respon les preguntes que segurament tens ara mateix.

---

## Què és aquest projecte?

Un **laboratori de paper trading algorísmic** per a BTC/USDT. Experimenta amb tres famílies d'estratègies (clàssiques, ML supervisat, i Reinforcement Learning) i un sistema de trading avançat de 5 portes (Gate System). Tot corre en paral·lel 24/7 amb diners virtuals mentre acumules evidència estadística de quines estratègies funcionen.

**No és producció.** L'objectiu és validar estratègies rigorosament abans de considerar diners reals.

---

## Prerequisits

```
Python 3.11+         (recomanat: pyenv)
PostgreSQL 14+       (o Docker Compose inclòs)
Hardware recomanat:  Apple M4 Pro 24GB / equivalent
Temps inicial:       ~2h per setup + ~3-4h per entrenar tots els models
```

Variables d'entorn necessàries (fitxer `.env` a l'arrel):
```env
DATABASE_URL=postgresql://btc_user:btc_password@localhost:5432/btc_trading
TELEGRAM_TOKEN=...       # opcional, per notificacions
TELEGRAM_CHAT_ID=...     # opcional
```

---

## Ordre de lectura — tria el teu perfil

### "Vull entendre el projecte de zero"
```
00_START_HERE.md (ara)
  → 01_ARCHITECTURE.md        (com funciona tot)
  → 02_GATE_SYSTEM.md         (l'estratègia principal)
  → 03_ML_RL_MODELS.md        (els altres models)
  → examples/trade_walkthrough.md  (un trade real pas a pas)
```

### "Vull posar-ho en marxa"
```
00_START_HERE.md (ara)
  → 01_ARCHITECTURE.md §Quick Start
  → 04_CONFIGURATION.md
  → 07_OPERATIONS.md          (runbook d'operació)
```

### "Vull afegir un bot / model nou"
```
06_EXTENDING.md             (receptes pas a pas)
  → 01_ARCHITECTURE.md §BaseBot
  → 04_CONFIGURATION.md §YAML unificat
```

### "Vull entendre per què les coses estan fetes així"
```
08_DECISIONS.md             (11 Architecture Decision Records)
  → 02_GATE_SYSTEM.md §Decisions descartades
```

---

## Mapa del projecte

```
btc-trading-bot/
│
├── bots/                    ← Les estratègies
│   ├── classical/           5 bots clàssics + EnsembleBot
│   ├── ml/                  7 models ML (XGBoost, GRU, PatchTST...)
│   ├── rl/                  8 agents RL (PPO, SAC, TD3...)
│   └── gate/                Gate System (5 portes seqüencials)
│       ├── gates/           P1..P5 — cada porta en un fitxer
│       └── regime_models/   HMM + XGBoost per classificar règims
│
├── core/                    ← La infraestructura (no tocar per afegir bots)
│   ├── interfaces/          BaseBot, BaseExchange (contractes ABC)
│   ├── engine/              DemoRunner + BacktestEngine
│   ├── db/                  Models SQLAlchemy + repositori
│   └── models.py            Signal, Order, Trade (Pydantic)
│
├── config/                  ← Configuració
│   ├── settings.yaml        Globals (BD, exchange, dates walk-forward)
│   ├── demo.yaml            Quins bots corren en demo
│   └── models/              UN YAML per model (tot en un: params + training)
│
├── data/                    ← Càrrega i transformació de dades
│   ├── processing/          FeatureBuilder, TechnicalIndicators
│   └── observation/         ObservationBuilder (construeix observació per bot)
│
├── scripts/                 ← Punts d'entrada (executar des d'aquí)
│   ├── download_data.py
│   ├── optimize_bots.py / optimize_models.py
│   ├── train_models.py / train_rl.py / train_gate_regime.py
│   ├── run_comparison.py
│   └── run_demo.py
│
├── models/                  ← Models entrenats (*.pkl, *.pt, *.zip)
├── tests/                   ← smoke/ + unit/ + integration/
└── docs/                    ← Estàs aquí
```

### Com es connecten les peces

```
Binance API
    │ candles OHLCV
    ▼
PostgreSQL (taula: candles)
    │
    ▼
ObservationBuilder
    │ finestra de N candles + features tècniques
    ▼
Bot.on_observation(obs) ──── declara necessitats amb observation_schema()
    │ Signal (BUY / SELL / HOLD)
    ▼
PaperExchange ──── fees 0.1% + slippage 0.01%
    │ Order
    ▼
PostgreSQL (taules: signals, orders, trades, ticks)
    │
    ├── Telegram → notificació si trade
    └── MLflow → mètriques de backtest
```

---

## Glossari

| Terme | Significat |
|-------|-----------|
| **Walk-forward validation** | Tècnica per evitar lookahead bias: els models s'entrenen amb dades fins a una data de tall i es validen en dades mai vistes (post-tall) |
| **Near-miss** | Una oportunitat que ha passat P1+P2+P3 però ha fallat a P4 o P5. S'enregistra per anàlisi posterior |
| **Gate / Porta** | Un filtre seqüencial del Gate System. Si qualsevol tanca, no hi ha trade |
| **Règim** | L'estat macro del mercat classificat per P1: STRONG_BULL, WEAK_BULL, RANGING, WEAK_BEAR, STRONG_BEAR, UNCERTAIN |
| **Fractal pivot** | Un swing high o low on el preu forma un màxim/mínim local respecte N candles adjacents |
| **HVN (High Volume Node)** | Zona de preu on s'ha transaccionat molt volum. Actua com a suport/resistència fort |
| **Circuit breaker** | Sortida d'emergència quan el preu cau >3×ATR en una sola candle |
| **Trailing stop** | Stop de pèrdues que puja automàticament quan el preu avança a favor, però mai baixa |
| **Deceleration exit** | Sortida quan la velocitat del preu (d2) és negativa durant N candles consecutives |
| **Kelly criterion** | Fórmula per calcular la mida òptima d'una posició en funció del risc i la confiança |
| **ATR percentile** | La volatilitat actual (ATR-14) expressada com a percentil vs la seva història. 80 = ara és més volàtil que el 80% de períodes passats |
| **Stagnation** | Una posició que porta >6 dies en pèrdues sense estructura de suport vàlida → es redueix al 50% |
| **Paper trading / Demo** | Trading amb diners virtuals (simulació) per validar estratègies sense risc real |
| **Signal** | Objecte que retorna un bot: `{action, size, confidence, reason}` |
| **ObservationSchema** | Cada bot declara aquí quines dades necessita: features, timeframes, lookback |
| **Auto-discovery** | El sistema llegeix tots els `config/models/*.yaml` i carrega els bots automàticament, sense llistes hardcodejades |
| **best_params** | Secció del YAML on Optuna guarda els hiperparàmetres òptims trobats |
| **P2 multiplier** | Factor [0.0–1.0] que escala la mida de les posicions del Gate System. 0 = veto total |
| **D1 / D2** | Primera i segona derivada del preu suavitzat (EWM). D1 = velocitat, D2 = acceleració |
| **HMM** | Hidden Markov Model: model estadístic que descobreix estats ocults en una sèrie temporal |
| **Viterbi** | Algoritme per decodificar la seqüència d'estats més probable d'un HMM |
| **BIC** | Bayesian Information Criterion: mètrica per triar el nombre òptim d'estats del HMM (menys és millor) |
| **R:R** | Risk:Reward ratio. R:R 2.0 = el target és 2 vegades la distància al stop |
| **VETO** | Condició en P5 que impedeix obrir una nova posició |

---

## FAQ — Les preguntes que tothom fa

**Per on començo si vull córrer el bot?**
Setup → download dades → entrena models → backtest → activa a demo.yaml → `python scripts/run_demo.py`. Veure `07_OPERATIONS.md` per a les comandes exactes.

**Com sé si un model funciona prou bé per activar-lo?**
Ha de superar HoldBot (buy & hold) en Sharpe Ratio i Calmar Ratio en el període de test out-of-sample. Veure `01_ARCHITECTURE.md §Walk-Forward`.

**Quin és el bot que s'entrena diferent de tots els altres?**
El Gate System. No usa `train_models.py` ni `train_rl.py`. Té el seu propi pipeline: `python scripts/train_gate_regime.py`. Veure `02_GATE_SYSTEM.md §Entrenament`.

**Quan re-entreno els models?**
ML/RL: quan arriben dades noves suficients (cada mes o trimestre). Gate System P1 (HMM+XGBoost): mensualment. Gate System P2/P3/P4 no necessiten re-entrenament (són deterministes).

**Per què hi ha dos sistemes de models (Pydantic + SQLAlchemy)?**
Pydantic valida dades en memòria (lògica de negoci). SQLAlchemy gestiona la persistència (BD). Mesclar-los crearia acoblament fort. Veure `08_DECISIONS.md §1`.

**Per que els bots clàssics calculen els seus indicadors i no els agafen del Feature Store?**
Perquè Optuna varia els paràmetres dels indicadors (p.ex. ema_fast=47) i el Feature Store no pot regenerar-se per cada trial. Veure `08_DECISIONS.md §4`.

**Puc córrer en producció amb diners reals?**
No directament. El sistema usa `PaperExchange`. Per producció caldria implementar `LiveBinanceExchange(BaseExchange)`, que és la interfície abstracta prevista però no implementada.

---

## Índex complet de la documentació

| Document | Contingut |
|----------|-----------|
| **[00_START_HERE.md](./00_START_HERE.md)** | Aquest document: mapa, glossari, FAQ |
| **[01_ARCHITECTURE.md](./01_ARCHITECTURE.md)** | Arquitectura, flux de dades, bots disponibles, scripts, mètriques |
| **[02_GATE_SYSTEM.md](./02_GATE_SYSTEM.md)** | Gate System complet: les 5 portes, entrenament, exemples |
| **[03_ML_RL_MODELS.md](./03_ML_RL_MODELS.md)** | Tots els models ML i RL: paràmetres, decisions, entrenament |
| **[04_CONFIGURATION.md](./04_CONFIGURATION.md)** | Referència de tots els YAMLs i arguments CLI |
| **[05_DATABASE.md](./05_DATABASE.md)** | Esquema de la BD: taules, columnes, consultes útils |
| **[06_EXTENDING.md](./06_EXTENDING.md)** | Receptes per afegir nous bots, models, agents i fonts de dades |
| **[07_OPERATIONS.md](./07_OPERATIONS.md)** | Runbook: com engegar, monitoritzar i mantenir el sistema |
| **[08_DECISIONS.md](./08_DECISIONS.md)** | 11 Architecture Decision Records amb context i alternatives |
| **[ROADMAP.md](./ROADMAP.md)** | Tasques pendents i camí a la demo 24/7 |
| **[examples/trade_walkthrough.md](./examples/trade_walkthrough.md)** | Un trade del Gate System pas a pas |
| **[examples/near_miss_analysis.md](./examples/near_miss_analysis.md)** | Anàlisi d'una oportunitat que no s'ha executat |
