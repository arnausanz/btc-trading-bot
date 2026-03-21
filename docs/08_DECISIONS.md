# Decisions d'Arquitectura — Architecture Decision Records

> 11 decisions de disseny clau amb context, alternatives descartades i justificació.
> Per a l'arquitectura resultant: veure **[01_ARCHITECTURE.md](./01_ARCHITECTURE.md)**.

**Data:** 2026-03-01
**Estat:** Acceptat
**Autor:** Arnau Sanz

---

## Context

Plataforma de trading algorítmic per BTC construïda des de zero amb l'objectiu de ser un sistema de demo 24/7 que provi rendibilitat abans de considerar diners reals. El sistema ha de suportar múltiples tipus de bots (clàssics, ML, RL) de manera modular, sense haver de tocar el codi core quan s'afegeix un nou bot o estratègia.

---

## Decisions

### 1. Separació Pydantic (validació) vs SQLAlchemy (persistència)

**Decisió:** Dos sistemes de models paral·lels: `core/models.py` (Pydantic) i `core/db/models.py` (SQLAlchemy).

**Per què:** Pydantic valida dades en memòria i genera errors clars quan un bot retorna valors incorrectes. SQLAlchemy gestiona la persistència a PostgreSQL. Barrejar-los crearia acoblament fort entre la lògica de negoci i l'esquema de base de dades.

**Implicació:** Quan s'afegeix un nou camp a `Signal`, s'ha d'actualitzar `core/models.py` (Pydantic) i `core/db/models.py` (SQLAlchemy) + generar migració Alembic.

---

### 2. Interfícies ABC per a bots i exchanges

**Decisió:** `BaseBot` i `BaseExchange` com a classes abstractes (ABC).

**Per què:** El `Runner` i el `DemoRunner` no saben quin bot o exchange estan executant. Treballen sempre contra la interfície. Això permet afegir un `LiveBinanceExchange` en el futur sense tocar res del core.

**Implicació:** Tot bot ha d'implementar `observation_schema()` i `on_observation()`. Tot exchange ha d'implementar `send_order()`, `get_portfolio()`, `get_balance()`, `get_portfolio_value()`, `set_current_price()`.

---

### 3. ObservationSchema — el bot declara el que necessita

**Decisió:** Cada bot declara les seves necessitats de dades via `ObservationSchema` (features, timeframes, lookback).

**Per què:** El `Runner` no ha de saber res sobre les dades que necessita cada bot. El bot declara el que vol i l'`ObservationBuilder` ho construeix automàticament. Afegir un bot nou que necessita dades de 4 timeframes no requereix cap canvi al Runner.

**Implicació:** Si un bot necessita una feature nova (ex: `sentiment_score`), l'ha de declarar a `observation_schema()` i l'`ObservationBuilder` ha de saber com construir-la.

---

### 4. Bots calculen indicadors dinàmicament

**Decisió:** TrendBot i GridBot calculen les seves EMAs i Bollinger Bands sobre la finestra rebuda, no depenen de columnes pre-calculades.

**Per què:** Permet que Optuna provi qualsevol combinació de paràmetres (ema_fast=47, ema_slow=179) sense haver de re-calcular el Feature Store. Si els períodes venguessin del Feature Store, Optuna no podria variar-los lliurement.

**Implicació:** Hi ha un cost computacional petit per candle. Per a backtesting sobre 36k candles és negligible.

---

### 5. PaperExchange net per cada backtest

**Decisió:** `BacktestEngine` crea un `PaperExchange` nou per cada execució.

**Per què:** Evita estat residual entre backtests. Si el `PaperExchange` es reutilitzés, el segon backtest comenzaria amb el portfolio del primer.

**Implicació:** No es pot fer "continuació" de backtests. Cada backtest és independent i comença sempre amb el capital inicial configurat.

---

### 6. MLflow SQLite local (no PostgreSQL)

**Decisió:** MLflow amb SQLite (`mlflow.db`) en comptes de PostgreSQL.

**Per què:** Simplicitat. MLflow no necessita concurrència ni alta disponibilitat. SQLite és zero-config i suficient per a un projecte de demo. PostgreSQL seria overkill i afegiria complexitat al Docker Compose.

**Implicació:** El fitxer `mlflow.db` és local i no es fa backup automàtic. Fer `git ignore` d'aquest fitxer és correcte — les mètriques es poden regenerar re-executant els backtests.

---

### 7. Reward Registry pattern per a RL

**Decisió:** Les reward functions de RL es registren amb un decorator `@register("nom")` i es referencien pel nom al YAML.

**Per què:** Permet afegir reward functions custom (un fitxer Python nou) sense tocar cap fitxer del core. El YAML de configuració simplement posa `reward_type: "sortino"` i el sistema la troba automàticament.

**Implicació:** Per afegir una reward function nova: crear un fitxer a `bots/rl/rewards/`, implementar la funció amb el decorator `@register`, i importar el mòdul a `bots/rl/environment.py` perquè s'executi el registre.

---

### 8. Credencials al `.env`, no al YAML

**Decisió:** `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, i `DATABASE_URL` al fitxer `.env`.

**Per què:** Els YAMLs es comiten al repositori. Les credencials mai han d'estar en un repositori, ni privat. El `.env` és local i està al `.gitignore`.

**Implicació:** En un servidor nou, s'ha de crear el `.env` manualment. El projecte no funciona sense ell si el Telegram està activat.

---

### 9. Timestamps sempre en UTC a la DB

**Decisió:** Tots els timestamps es guarden en UTC a PostgreSQL. La conversió a timezone local (Europe/Madrid) només es fa al mostrar per pantalla.

**Per què:** UTC és l'estàndard per a sistemes distribuïts i evita problemes amb canvis d'hora (horari d'estiu). La DB no ha de saber on viu l'usuari.

**Implicació:** Quan es fan queries directament a la DB, els timestamps apareixen en UTC. Al dashboard i als logs es mostren en hora local.

---

### 10. Constant `MLFLOW_TRACKING_URI` centralitzada

**Decisió:** La URI de MLflow viu a `core/config.py` i s'importa des d'aquí.

**Per què:** Anteriorment estava duplicada a `BacktestEngine`, `RandomForestModel`, `XGBoostModel`, i `RLTrainer`. Si es canvia la ubicació de MLflow (ex: servidor remot), s'ha de canviar en un sol lloc.

**Implicació:** Qualsevol component nou que necesiti MLflow ha d'importar `from core.config import MLFLOW_TRACKING_URI`.

---

---

### 11. Gate System — Decisions d'Arquitectura (Març 2026)

**Context:** Implementació del Gate System, un bot de swing trading de nova generació basat en 5 portes seqüencials. Quatre decisions arquitecturals clau.

---

**11-A: Long-only en v1**

**Decisió:** V1 opera únicament en llarg (STRONG_BULL, WEAK_BULL, RANGING → entrades; BEAR/UNCERTAIN → flat).

**Per què:** Les posicions curtes requereixen gestió d'stop específica per a mercats en tendència decreixent i validació externa addicional. En un bull market estructural de BTC, el risc/recompensa de curts és asimètric. Es pot afegir en v2 un cop la infraestructura de portes estigui validada en paper trading.

**Implicació:** `p5.py` només genera ordres BUY. La lògica de SELL és exclusivament per tancar posicions llargues (stop, target, trailing).

---

**11-B: Sense news sentiment en v1**

**Decisió:** P2 usa Fear & Greed + funding rate com a proxy; news sentiment (LLM) exclòs.

**Per què:** El news sentiment requereix integració d'una API externa (CryptoPanic), un cron diari, un model LLM, i validació del seu senyal. Afegeix complexitat d'infraestructura que no és necessària per validar la hipòtesi central del Gate System. Es pot afegir com a component de P2 en v2 (ja documentat al ROADMAP.md §J).

**Implicació:** `p2_health.py` usa `_fg_score()` + `_funding_score()`. El `_news_score()` és un stub comentat.

---

**11-C: 14 features P1, sense exchange_netflow**

**Decisió:** Les 14 features de P1 inclouen EMA slopes, ADX, ATR percentile, funding rate (3d/7d), volume, Fear & Greed i returns. `exchange_netflow_7d` exclòs.

**Per què:** `exchange_netflow` requereix CryptoQuant o Glassnode (APIs de pagament, ~$50-$200/mes). Les 14 features seleccionades estan totes disponibles de fonts gratuïtes (Binance + alternative.me). Mantenir el sistema zero-cost és prioritat per a la fase de demo.

**Implicació:** `xgb_classifier.py::P1_FEATURES` conté exactament 14 entrades. Cap referència a `exchange_netflow`.

---

**11-D: Sense prefix, unió deduplificada a ObservationSchema**

**Decisió:** `GateBot.observation_schema()` declara les features sense prefix (no `4h_close`, `1d_close`); la unió de features_4h + features_1d és deduplificada (features compartides com `close`, `rsi_14` apareixen una sola vegada).

**Per què:** Els dfs de cada timeframe (`obs["4h"]`, `obs["1d"]`) ja estan separats per clau. El prefix seria redundant i complicaria els noms de columna. La deduplicació evita declarar la mateixa feature dues vegades.

**Implicació:** `ObservationBuilder.build()` construeix cada timeframe de forma independent. `GateBot` accedeix `obs["4h"]["features"]` i `obs["1d"]["features"]` per separat i selecciona les columnes que necessita per nom.

---

## Alternatives Descartades

| Alternativa | Per què es va descartar |
|---|---|
| TA-Lib per a indicadors | Problemes de compilació en alguns entorns; implementació pròpia amb pandas és suficient i portable |
| Shorts en Gate System v1 | Risc asimètric en bull market; complexitat d'stop addicional; v2 ho pot afegir un cop validada la infraestructura |
| News sentiment en P2 v1 | Requereix API externa de pagament + LLM; afegeix complexitat innecessària per validar la hipòtesi central |
| exchange_netflow com a feature P1 | API de pagament (CryptoQuant/Glassnode); prioritat zero-cost per a la fase de demo |
| Prefix en features multi-TF | Redundant quan cada TF té el seu propi df; la deduplicació és més neta que gestionar `4h_close` vs `1d_close` |
| Polars en comptes de pandas | L'ecosistema ML (sklearn, SB3) espera pandas/numpy; la conversió afegiria complexitat |
| Redis per a cache d'observacions | Overkill per a demo local; dict en memòria és suficient |
| FastAPI per a REST API | No necessari per a demo; Telegram és suficient per a interacció remota |
| DVC per a versionat de dades | La DB ja és la font de veritat; DVC afegiria complexitat sense benefici clar |
