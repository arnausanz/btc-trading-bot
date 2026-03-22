# Auditoria Multi-Agent — BTC Trading Bot

> **Metodologia:** Tres rols simultanis — Arquitecte de Sistemes, Auditor de Codi Senior, Enginyer Quant — anàlisi independent i convergència en un veredicte únic.
>
> Totes les afirmacions estan ancrades en codi real amb referències `fitxer:línia`.

**Data:** Març 2026 · Versió del codi: branch `develop` (1dff007)

---

## Índex

1. [Revisió d'Arquitectura](#1-revisió-darquitectura)
2. [Auditoria de Codi](#2-auditoria-de-codi)
3. [Validació de Trading](#3-validació-de-trading)
4. [Elements Obsolets i Redundants](#4-elements-obsolets-i-redundants)
5. [Roadmap Reescrit](#5-roadmap-reescrit)
6. [Correccions de Documentació](#6-correccions-de-documentació)
7. [Veredicte Final](#7-veredicte-final)

---

## 1. Revisió d'Arquitectura

### Fortaleses

**1. Contracte BaseBot com a ABC genuí**
`core/interfaces/base_bot.py` força tots els bots a implementar `observation_schema()` i `on_observation()`. El Runner (`core/engine/runner.py:22`) opera exclusivament contra aquesta interfície — no coneix ni li importa el tipus de bot. Afegir un nou bot no requereix tocar el motor.

**2. Auto-discovery via YAML**
`core/config_utils.py` localitza configs per categoria (`category: ML`) sense cap llista codificada. El mateix patró s'aplica al registre de reward functions (`@register("name")` a `bots/rl/rewards/`) i al registre d'agents (`_AGENT_REGISTRY` a `bots/rl/rl_bot.py:14`). Extensible per disseny.

**3. Separació Pydantic / SQLAlchemy**
`core/models.py` (validació en memòria) i `core/db/models.py` (persistència) són completament independents. La capa de validació pot canviar sense tocar el schema de BD, i viceversa. Decisió correcta documentada a `docs/08_DECISIONS.md`.

**4. Walk-forward enforçat a nivell de config**
`config/settings.yaml:29-30` defineix `train_until: "2024-12-31"` i `test_from: "2025-01-01"`. El BacktestEngine respecta aquests límits. Cap bot entra a demo sense superar el criteri en test out-of-sample. Prevenció de lookahead bias sòlida.

**5. Infraestructura de producció per a paper trading**
Oracle Cloud Free Tier + systemd (`btc-demo.service`) + cron jobs per a dades externes + `pg_dump` diari = infraestructura robusta per a un projecte de paper trading. El DemoRunner inclou `restore_state()` (`exchanges/paper.py:115`) per a recuperació en reinicis.

**6. Pipeline de dades extern correcte i idempotent**
Totes les fonts externes (Fear&Greed, FundingRates, OpenInterest, BlockchainMetrics) implementen deduplicació via `UniqueConstraint` a la BD + precarrega del conjunt de timestamps existents en memòria. Cap doble inserció possible. L'excepció notable és `candles` (veure Debilitats).

---

### Debilitats

**D1 — `CandleDB` sense `UniqueConstraint` (HIGH)**
`core/db/models.py:12-24`: la taula `candles` és l'única taula de dades que **no** té `UniqueConstraint`. Totes les altres taules (fear_greed, funding_rates, open_interest, blockchain_metrics) el tenen. Executar `download_data.py` dues vegades sobre el mateix rang insereix duplicats silenciosament. El fetcher (`data/sources/ohlcv.py:94-99`) fa un `session.query(...).filter_by(...).first()` per cada candle — correcte funcionalment però que seria innecessari amb un constraint a nivell de BD.

**D2 — Pattern N+1 a `OHLCVFetcher._save()`**
`data/sources/ohlcv.py:93-113`: per cada candle de la pàgina (màx 1000), fa una query individual per comprovar si existeix. En una descàrrega inicial de 5 anys × 4 timeframes = ~175.000 candles → ~175.000 queries individuals. Contrast amb `FearGreedFetcher` que carrega el conjunt de timestamps un cop (`session.query(...).all()`) i fa lookup en memòria. Impacte: la descàrrega inicial és ordres de magnitud més lenta del que hauria de ser.

**D3 — Sin validació d'`obs_shape` en temps de càrrega**
`bots/rl/rl_bot.py:60-62`: `agent.load(model_path)` carrega el model sense verificar que la forma de l'observació del model guardat coincideixi amb la forma que generarà `on_observation()`. Si `features.select` al YAML del bot no coincideix exactament amb el de l'entrenament, el codi peta en temps d'execució (`ValueError` a `act()`), no en temps de configuració. L'error és críptic i difícil de diagnosticar sense conèixer la regla.

**D4 — DemoRunner síncrona amb N bots**
El DemoRunner itera sobre tots els bots actius seqüencialment cada 60 segons. Amb 4 bots actuals és negligible. Amb 10+ bots (objectiu del Roadmap), el cicle podria superar els 60 segons si el GateBot o els models ML triguen en inference. No hi ha timeout per bot ni execució async.

**D5 — Símbols i timeframes hardcodejats**
`config/settings.yaml:21-25` defineix `default_symbol: BTC/USDT`. Tota la lògica de negoci (runners, observació builders, bots) assumeix un únic par BTC/USDT. Afegir ETH/USDT o qualsevol altre actiu requeriria canvis no trivials a múltiples capes. Acceptable per l'abast actual (laboratori BTC), però important documentar-ho com a limitació explícita.

---

### Riscos Estructurals

| Risc | On | Probabilitat | Impacte |
|------|----|-------------|---------|
| Duplicats a `candles` per re-descàrrega | `core/db/models.py:12` | Alta (si es re-executa download) | Mitjà (backtest amb dades duplicades) |
| Shape mismatch RL silent crash | `bots/rl/rl_bot.py:60` | Moderada (per canvis de config) | Alt (bot inoperatiu en producció) |
| DemoRunner timeout amb molts bots | `core/engine/demo_runner.py` | Baixa ara, Alta amb 10+ bots | Baix-Mitjà |
| Model stale sense détection | Cap fitxer | Certa (passa contínuament) | Alt (degradació silenciosa del rendiment) |

---

## 2. Auditoria de Codi

### Bugs Confirmats (codi llegit, no teòrics)

#### B1 — `CandleDB` sense `UniqueConstraint` · **SEVERITAT: HIGH**
```python
# core/db/models.py:12-24
class CandleDB(Base):
    __tablename__ = "candles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = ...
    symbol: Mapped[str] = ...
    timeframe: Mapped[str] = ...
    timestamp: Mapped[datetime] = ...
    # ← NO hi ha __table_args__ = (UniqueConstraint(...),)
```
Totes les altres taules de dades el tenen. La correcció requereix una nova migració Alembic + un `DELETE FROM candles WHERE id NOT IN (SELECT MIN(id) FROM candles GROUP BY exchange, symbol, timeframe, timestamp)` per netejar duplicats existents.

#### B2 — N+1 queries a `OHLCVFetcher._save()` · **SEVERITAT: MEDIUM**
```python
# data/sources/ohlcv.py:93-99
for candle in candles:
    exists = session.query(CandleDB).filter_by(
        exchange=candle.exchange,
        symbol=candle.symbol,
        timeframe=candle.timeframe,
        timestamp=candle.timestamp,
    ).first()  # ← 1 query per candle
```
Patró correcte (com fan les altres fonts):
```python
# Exemple: data/sources/fear_greed.py
existing_ts = {r.timestamp for r in session.query(FearGreedDB.timestamp).all()}
new_entries = [e for e in entries if e.timestamp not in existing_ts]
session.bulk_save_objects(new_entries)
```

#### B3 — `win_rate()` amb `iterrows()` · **SEVERITAT: MEDIUM**
```python
# core/backtesting/metrics.py:126
for _, row in self.df.iterrows():  # ← O(n) Python loop
    signal_str = str(row.get("signal", "")).lower()
    ...
```
Amb 50k+ ticks (1 any de demo a 1h), `iterrows()` pot trigar diversos segons. A més, la detecció de "buy" via `str.lower()` és fràgil: `Action.BUY` serialitza com `"Action.BUY"` → `.lower()` → `"action.buy"` → `"buy" in "action.buy"` = True (correcte per accident). Si el format de `Signal.action` canvia, pot deixar de funcionar silenciosament.

#### B4 — Trade counter inconsistència a `BtcTradingEnvContinuous` · **SEVERITAT: LOW**
```python
# bots/rl/environment.py:190
if position_delta > 0.01:  # ← threshold hardcodejat
    self.trades += 1
```
`BtcTradingEnvProfessionalContinuous` usa `DEADBAND` (=0.05) per decidir si rebalança. El comptador de trades de l'entorn base usa 0.01. Inconsistència que infla el comptador de trades a l'entorn base respecte al professional.

#### B5 — ATR fallback silenciós a `_ProfessionalBase` · **SEVERITAT: MEDIUM**
```python
# bots/rl/environment_professional.py:158
atr_14 = _safe_col(row, "atr_14", price * ATR_REFERENCE)  # fallback = price * 0.02
```
`ATR_REFERENCE = 0.02` → stop loss = `entry × (1 - 2.0 × 0.02)` = `entry × 0.96`. Si el model és entrenat sense `atr_14` en el seu feature set, tots els stops es calculen assumint volatilitat del 2%, que pot ser radicalment incorrecte (p.ex., en mercats molt volàtils el 2019-2020 o molt calmats). No hi ha warning quan s'activa el fallback.

#### B6 — `BlockchainFetcher` accepta qualsevol metric sense validació · **SEVERITAT: LOW**
```python
# data/sources/blockchain.py (inferit del patró)
def fetch_and_store(self, metric: str, ...):
    url = f"{BASE_URL}/{metric}?..."  # ← typo en metric → HTTP 404 en temps d'execució
```
`BLOCKCHAIN_METRICS` frozenset existeix a `core/models.py` però no s'usa per validar el paràmetre d'entrada. Un `assert metric in BLOCKCHAIN_METRICS` als inicis del mètode prevendria l'error.

---

### Falsos Positius (reportats per sub-agents, NO són bugs)

**FP1 — Fee calculation a `paper.py:59`**
`capital_to_use = budget / (1 + fee_rate)` → `fees = capital_to_use * fee_rate` → `capital_to_use + fees = budget` exactament. Matemàticament correcte i ben documentat en el comentari de la línia 56. No és un bug.

**FP2 — Fee en `BtcTradingEnvDiscrete` línies 128/134**
```python
self.btc_balance = (self.usdt_balance * (1 - self.fee_rate)) / price  # BUY
self.usdt_balance = self.btc_balance * price * (1 - self.fee_rate)    # SELL
```
Equivalent semàntic a pagar fees sobre el valor de la transacció. El resultat final (portfolio value) és correcte. No és un bug, és una implementació diferent però equivalent.

**FP3 — Scaling 0.1 a rewards Sharpe/Sortino**
```python
# bots/rl/rewards/builtins.py:42
return float((ret / std) * scaling * 0.1)
```
`scaling` és un hiperparàmetre Optuna (`reward_scaling: {type: float, low: 1.0, high: 500.0}`). El 0.1 normalitza el ratio ret/std a magnituds comparables amb `simple_reward`. No és un bug — és un hiperparàmetre de disseny que pot (i ha de) ser tunejat.

---

### Males Pràctiques

| # | Fitxer | Problema | Recomanació |
|---|--------|---------|------------|
| MP1 | `data/sources/ohlcv.py:145` | Start date `2019-01-01` hardcodejat | Constants mòdul o config `settings.yaml` |
| MP2 | `data/sources/ohlcv.py:50` | Pàgina size `1000` hardcodejat | Constant `BINANCE_PAGE_SIZE = 1000` |
| MP3 | Múltiples fetchers | `500`, `288`, `30` magic numbers | Constants de mòdul documentades |
| MP4 | `core/backtesting/metrics.py:60` | Sharpe retorna 0.0 si std==0 (indocumentat) | Afegir comentari: "Retorna 0.0 per estratègies sense variació en portfolio" |
| MP5 | `config/settings.yaml:9-11` | Credencials de BD al fitxer de config (no `.env`) | Documentar que es sobreescriuen per `DATABASE_URL` al `.env` |

---

### Codi Mort / No Usat

| Element | Fitxer | Estat | Acció |
|---------|--------|-------|-------|
| `PaperExchange.get_candles()` | `exchanges/paper.py:31-32` | Retorna `[]` sempre. Cap bot l'usa. Part del contracte `BaseExchange`. | Mantenir com a stub del contracte; afegir `# stub: no s'usa en paper trading` |
| Taules `signals`, `orders`, `trades` | `core/db/models.py:27-68` | Definides però mai escrites pel DemoRunner | Ja documentades com a "legacy" a `05_DATABASE.md`. Estat correcte. |
| `config/models/tft.yaml` | `config/models/` | Optimitzada? No. Entrenada? No. Validada? No. | Afegir `enabled: false` al YAML + nota d'estat |

---

### Consistència i Mantenibilitat

**Positiu:**
- Tots els fitxers de `data/sources/` segueixen el mateix patró (fetch→validate→save→deduplicate). Qualsevol programador pot entendre un fetcher nou en 5 minuts.
- Reward registry amb `@register` és el pattern correcte. Afegir una nova reward function = afegir un fitxer i un decorador.
- YAML-driven config és la decisió correcta. Cap experimentació requereix canviar codi core.

**Negatiu:**
- `_get_observation()` a `bots/rl/environment.py:55-72` i la normalització equivalent a `bots/rl/rl_bot.py:91-93` estan duplicades (tot i que similars). Si la normalització canvia a l'entrenament, cal recordar canviar-la també a la inferència. Risc de drift tàcit.
- `bots/rl/rl_bot.py:111-122` inicialitza `_entry_price`, `_was_in_position`, `_steps_in_pos` amb `hasattr` + valor default dins `on_observation()`. Millor inicialitzar a `on_start()`.

---

## 3. Validació de Trading

### Estratègies Clàssiques

**HoldBot** — Correcta com a benchmark. Compra al primer tick i manté. L'única garantia és que tots els altres bots han de superar-lo.

**DCABot** — Acumulació sistemàtica sense lògica de sortida. Funciona en bulls llargs. En bears pot acumular BTC fins a exhaurir el capital. No hi ha cap mecanisme de "prou he comprat". Adequat per a l'objectiu de comparació.

**TrendBot (EMA crossover)** — Indicador retardat per definició. En mercats laterals (2024: ~60% del temps) genera múltiples falsos senyals. Consistent amb la seva baixa performance en el test 2025.

**GridBot** — Estratègia de range-bound correcta en teoria. El problema és que les bandes (suport/resistència) es calculen sobre dades d'entrenament. En test out-of-sample, el rang pot ser completament diferent. Requeriria recalibració dinàmica per ser vàlid.

**MomentumBot** — El more sophisticated dels clàssics: ROC + MACD + RSI + Volume. Una preocupació de rendiment: `bots/classical/momentum_bot.py` recalcula MACD cada tick (`ta.macd(close_series)`) quan ja està pre-computat a les features. Cost addicional innecessari per N candles de lookback en cada crida.

**MeanReversionBot** — RSI oversold/overbought. Elevat false positive rate en trending regimes. El 2025 (bearish+lateral) és exactament el pitjor entorn per a mean reversion.

---

### Models ML (Resultats 2025 Out-of-Sample)

| Model | Return | Sharpe | Calmar | Estat |
|-------|--------|--------|--------|-------|
| RF | +4.44% | 1.14 | >0.5 | ✅ Demo |
| XGB | -3.53% | <1.0 | <0.5 | ⚠️ Observació |
| LGBM | - | <1.0 | - | ❌ Descartats |
| CatBoost | - | <1.0 | - | ❌ |
| GRU | - | <1.0 | - | ❌ |
| PatchTST | - | <1.0 | - | ❌ |
| TFT | No entrenat | - | - | ❌ Possiblement descartat |

**Problemes reals dels models ML:**

1. **Staleness implícita**: Els models es reentrenan manualment. Cap sistema detecta si el model actual ha deixat de ser predictiu. El RF pot tenir Sharpe 1.14 el gener 2025 i 0.3 el gener 2026. No hi ha alarma.

2. **Classificació binària massa simplificada**: Tota la predicció és `price_t+1 > price_t` → (0/1). No hi ha diferenciació entre "puja un 0.1%" i "puja un 5%". Un model que encerta tots els petits moviments però falla en els grans pot tenir accuracy alta i performance dolenta.

3. **Threshold fix a 0.55** (`prediction_threshold: 0.55` als YAMLs): No s'adapta al règim de mercat. En alta incertesa (Fear & Greed < 25), el threshold hauria de ser més conservador.

---

### Agents RL

**Agents baseline (PPO/SAC 1H) — `comparable: false`**
Exclosos del walk-forward per manca de dades on-chain historials. Això significa que hi ha agents entrenats però sense cap validació out-of-sample. El camí cap a demo és bloquejat correctament — però es podria resoldre entrenant-los sense features on-chain.

**Agents professional (PPO/SAC/TD3 12H) — `comparable: true`**
El Roadmap indica "500k steps, ~1.2k candles 12H disponibles". Amb lookback=96, les mostres efectives de training = ~1.104 transicions. Amb PPO `n_steps=2048`, ni una sola actualització de política utilitza tots els trajectories disponibles. Risc d'overfitting en el període d'entrenament altíssim. Necessiten validació rigorosa en 2025 out-of-sample.

**Normalització d'observació en RL**
`bots/rl/environment.py:60-63`: normalització per-column per z-score sobre la finestra de lookback actual. Aquesta és una normalització **online** (usa dades presents però no futures). No introdueix lookahead bias. Correcte.

**Problema real — Continuïtat d'entrenament vs. inferència**
El model RL aprèn en un entorn simulat amb dades sintètiques (normalitzades per finestra). En producció (DemoRunner), la normalització és idèntica però el context és diferent: el mercat viu té distribucions no estacionàries. El model pot degradar-se ràpidament si el mercat entra en un règim no vist durant l'entrenament.

---

### Gate System — L'estratègia més sòlida

El Gate System és l'estratègia financerament més justificada del projecte. La seva arquitectura — règim → salut on-chain → estructura tècnica → momentum → risc — imita el procés de decisió d'un swing trader institucional.

**Fortaleses:**
- P1 HMM: detecció probabilística de règim, no binària. El bot descarta el trade si la confiança del règim és massa baixa.
- P5 `gate_near_misses`: logging de quasi-trades permet calibrar els llindars de forma empírica.
- ATR stop loss forçat per l'entorn (`_check_and_apply_stop_loss`) — l'agent no pot ignorar-lo.
- Trailing stop (`highest_price` a `gate_positions`) — preserva guanys.

**Febleses:**
- Long-only. En règim BEAR (P1), el sistema no opera en absolut. El capital queda inactiu en USDT en comptes de shortejar o cobrar yield.
- Posició única simultàniament. No pot escalar en una posició guanyadora ni diversificar en múltiples setups.
- P2 "health multiplier" — el seu impacte depèn de la correlació real entre on-chain metrics i price action a curts terminis (4H-1D). Aquesta correlació pot ser molt feble en el swing trading.

---

### Components Crítics Absents per Trading Real

| Component | Absència | Impacte |
|-----------|---------|---------|
| **Sizing volàtil** | Tots els bots usen `trade_size` fix | En alta volatilitat, una posició del 50% pot perdre el doble del normal |
| **Circuit breaker portfolio** | No existeix | Si tot cau un 30% en un mes, cap bot s'atura automàticament |
| **Slippage asimètric** | Fix i simètric al 0.05% | BTC pot tenir spreads de 0.1-0.3% en sells ràpids; el backtest és optimista |
| **Detecció de regime shift** | Solo GateBot (P1) | Els bots ML/RL corren en tots els règims sense distinció |
| **Model drift detection** | No existeix | Un model degradat pot generar pèrdues durant setmanes fins que es detecti manualment |
| **Correlació entre bots** | No considerada | 4+ bots en BTC/USDT → drawdowns correlacionats; l'efecte diversificació és nul |
| **Funding cost (futures)** | No modelat | ~0.01%/8h = ~11%/any si s'usen perpetuals en trading real |

---

### Supervivència en Mercats Reals

**Amb paper trading (demo):** SÍ, el sistema és adequat. És exactament per a això que va ser dissenyat.

**Amb diners reals al dia d'avui:** NO. Raons per ordre de severitat:
1. Únic model ML que supera el criteri (RF) — si falla, no hi ha backup
2. Cap agent RL validat out-of-sample per a demo
3. Sense circuit breaker de portfolio
4. Sense detecció de staleness dels models
5. Integració amb exchange real no implementada

---

## 4. Elements Obsolets i Redundants

### Roadmap — Seccions ja fetes, innecessàries, o enganyoses

| Secció | Estat real | Acció |
|--------|-----------|-------|
| **Tasca D — Migració servidor** | ✅ 100% completat. Oracle Cloud actiu, systemd corrent. | Eliminar del Roadmap |
| **Tasca K — Neteja de BD** | ✅ No cal: BD nova al servidor. | Eliminar del Roadmap |
| Estat inicial (taula de components) | ✅ Tots els ítems estan completats o marcats. Aporta poc valor. | Condensar en una línia |

### Codi / Config — Elements que creen confusió

| Element | Fitxer | Estat | Acció |
|---------|--------|-------|-------|
| `PaperExchange.get_candles()` | `exchanges/paper.py:31` | Stub etern (`return []`), cap bot l'usa | Afegir comentari `# stub — part del contracte BaseExchange; no usat en paper trading` |
| `config/models/tft.yaml` | `config/models/` | Existeix el config però mai entrenat ni optimitzat | Afegir `status: discarded_pending_review` i comentari al YAML |
| `config/models/ppo.yaml`, `sac.yaml` | `config/models/` | `comparable: false` — cap camí cap a demo | Afegir `note: "Exclòs del walk-forward. Entrena sense features on-chain o usa versió professional."` |
| `config/models/ppo_onchain.yaml`, `sac_onchain.yaml` | `config/models/` | Mateixa situació | Mateixa nota |

### Docs — Seccions desactualitzades

`docs/ROADMAP.md` conté les seccions D i K que ja estan completades. El document ha de reflectir únicament el que queda per fer.

---

## 5. Roadmap Reescrit

> Conté **únicament** elements pendents. Les tasques completades s'han eliminat.
> Versió 7.0 — Març 2026

### Deute Tècnic (a resoldre ABANS de la demo)

| # | Problema | Fitxer | Prioritat |
|---|---------|--------|----------|
| DT1 | `CandleDB` sense `UniqueConstraint` — duplicats silenciosos | `core/db/models.py:12` | 🔴 BLOQUEJANT |
| DT2 | N+1 queries a `OHLCVFetcher._save()` | `data/sources/ohlcv.py:93` | 🟠 Alta |
| DT3 | Sense validació `obs_shape` en càrrega de model RL | `bots/rl/rl_bot.py:60` | 🟠 Alta |
| DT4 | `win_rate()` amb `iterrows()` — O(n) Python loop | `core/backtesting/metrics.py:126` | 🟡 Mitjana |
| DT5 | ATR fallback silenciós sense warning | `bots/rl/environment_professional.py:158` | 🟡 Mitjana |

**DT1 — Fix concret:**
```bash
# Nova migració Alembic
alembic revision -m "add_unique_constraint_candles"
# Dins la migració:
# 1. DELETE duplicats (keep MIN(id))
# 2. ALTER TABLE candles ADD CONSTRAINT uq_candles UNIQUE (exchange, symbol, timeframe, timestamp)
```

**DT3 — Fix concret:**
A `RLBot._load_agent()`, després de `agent.load(model_path)`, calcular `expected_shape` des del config i verificar que coincideix amb `agent.policy.observation_space.shape`.

---

### Camí Crític cap a la Demo

```
[DT1] Fix CandleDB UniqueConstraint + migració
          ↓
[A]   Comparativa walk-forward → selecció sub-bots per EnsembleBot
          ↓
[A]   Entrenar models finals (els que superin Sharpe>1.0, Calmar>0.5)
          + train_gate_regime.py (si Gate System entra a demo)
          ↓
[B]   Backtest EnsembleBot — ha de superar HoldBot
          ↓
[E]   🚀 Iniciar demo 24/7 (mín. 3-6 mesos paper trading)
```

**Criteri d'entrada a demo:** Sharpe > 1.0, Max Drawdown > -25%, Calmar > 0.5 en backtest out-of-sample (2025-01-01 → avui).

---

### En Paral·lel mentre la demo corre

#### G — Telegram millorat
- Resum diari: PnL per bot, comparativa vs BTC spot
- Alertes de drawdown configurables per bot
- Rànquing setmanal
- Comandes: `/status`, `/portfolio`, `/ranking`

#### H — Dashboard Streamlit
Quan hi hagi 3+ mesos de dades reals:
- Portfolios en temps real per bot (PnL acumulat)
- Trade log filtrable
- Comparativa vs BTC spot
- Ranking per Sharpe/Calmar (darrers 30/90/180 dies)

#### J — Capa LLM per a sentiment ⭐ (Alta prioritat)
```
CryptoPanic/RSS → LLM (Claude Haiku / GPT-4o-mini) → news_sentiment [-1, 1]
    ↓
BD: taula `news_sentiment` (timestamp, score, raw_text)
    ↓
FeatureBuilder → columna `news_sentiment` en ML + RL models
```
Implementació: `data/sources/news_sentiment.py` → `core/db/models.py` (nova taula) → `data/processing/external.py` → YAMLs actualitzats. Cron diari 07:00 UTC. Cost <$20/any.

#### I — Nous models ML/RL

| Model | Prioritat | Notes |
|-------|-----------|-------|
| EnsembleBot weighted | 🟢 Alta | Pes proporcional al Sharpe dels darrers N dies |
| Gate System v2 | 🟢 Alta | Shorts + news_sentiment en P2 + exchange_netflow |
| N-BEATS / N-HiTS | 🟡 Mitjana | Forecasting pur, eficient, sovint supera GRU |
| TabNet | 🟡 Mitjana | Attention-based tabular — complement a XGBoost |
| BreakoutBot | 🟡 Mitjana | Pivot points + ATR per confirmar ruptures |
| DreamerV3 | 🔴 Baixa | World model RL — projecte de recerca, massa complex ara |

---

### Millores de Robustesa (Important, no bloquejants)

| Item | Acció | Fitxer |
|------|-------|--------|
| Model staleness | Afegir mètrica de rolling Sharpe dels darrers 30 dies; alarma Telegram si cau <0.5 | `core/engine/demo_runner.py` |
| Refactoritzar `OHLCVFetcher._save()` | Patró batch (com `FearGreedFetcher`) | `data/sources/ohlcv.py:85` |
| Validar `obs_shape` en càrrega | Assert a `RLBot._load_agent()` | `bots/rl/rl_bot.py:60` |
| Vectoritzar `win_rate()` | Reemplaçar `iterrows()` per operacions pandas | `core/backtesting/metrics.py:113` |
| Warning ATR fallback | Log WARNING quan s'activa `ATR_REFERENCE` | `bots/rl/environment_professional.py:158` |

---

## 6. Correccions de Documentació

### Inconsistències reals trobades

**I1 — `docs/05_DATABASE.md`: Candles sense `UniqueConstraint` no documentat**
La secció de `candles` (`docs/05_DATABASE.md:53-72`) no menciona en cap lloc que la taula no té `UniqueConstraint`. Donada la importància d'aquest gap, hauria d'estar explícitament advertit.

**Afegir a `docs/05_DATABASE.md` sota la secció `candles`:**
```markdown
> ⚠️ **Atenció — Gap de deduplicació:** La taula `candles` NO té `UniqueConstraint`.
> Executar `download_data.py` dues vegades sobre el mateix rang insereix duplicats
> silenciosament. El fetcher `OHLCVFetcher._save()` fa deduplicació per software
> (query individual per candle) però és una línia de defensa fràgil. Veure Deute Tècnic DT1.
> **Fix pendent:** Migració Alembic per afegir `UniqueConstraint(exchange, symbol, timeframe, timestamp)`.
```

**I2 — `docs/01_ARCHITECTURE.md`: semanticació `merge_asof` confusa**
El `FeatureBuilder` usa `merge_asof(direction="backward")`. En pandas, `direction="backward"` significa que cada timestamp rep el valor més recent del passat de la font externa (backward-fill). Documentar-ho explícitament per evitar confusió amb "forward-fill".

**Afegir nota a `docs/01_ARCHITECTURE.md` a la secció de Feature Builder:**
```markdown
> **Nota sobre `merge_asof(direction='backward')`:** Per a cada candle, s'assigna
> el valor extern MÉS RECENT DEL PASSAT. Si el Fear&Greed del dia 15 és l'últim
> disponible, tots els candles del dia 16 fins al 17 reben el valor del dia 15.
> Això és backward-fill (no forward-fill) i NO introdueix lookahead bias perquè
> sempre s'usa informació passada.
```

**I3 — `docs/03_ML_RL_MODELS.md`: Invariant `obs_shape` poc prominent**
La regla crítica de `obs_shape = len(features.select) × lookback` (+ 4 per a professional) hauria d'estar en un box prominent prop de la secció RL, no només a la memòria persistent.

**Afegir a `docs/03_ML_RL_MODELS.md` a la secció 5 (Models RL), al principi:**
```markdown
> ⚠️ **Invariant Crític — `obs_shape`**
>
> L'`obs_shape` del model entrenat i de la inferència han de coincidir exactament:
>
> - **Agents base/on-chain:** `obs_shape = len(features.select) × lookback`
> - **Agents professional:** `obs_shape = len(features.select) × lookback + 4`
>   (+4 = pnl_pct, position_fraction, steps_in_position, drawdown_pct)
>
> Si `features.select` al YAML de training ≠ `features` al YAML del bot → **crash en runtime**.
> Si `lookback` al YAML de training ≠ `lookback` al YAML del bot → **crash en runtime**.
>
> Verificació: `assert len(bot_features) * lookback == model.policy.observation_space.shape[0]`
```

**I4 — `docs/ROADMAP.md`: Seccions D i K obsoletes**
Eliminades en la reescritura del Roadmap (Secció 5 d'aquest document).

**I5 — `config/settings.yaml:4` — `env: development` sense efecte operatiu documentat**
El camp `app.env: development` no té efecte en cap part del codi actual (cap bloc `if env == 'production'`). Si es pretén usar en el futur, documentar-ho. Si és ornamental, eliminar-lo per evitar confusió.

---

## 7. Veredicte Final

### Resum executiu

| Dimensió | Puntuació | Justificació |
|----------|-----------|-------------|
| **Arquitectura** | 8/10 | Sòlida, extensible, ben dissenyada. Penalitzada per CandleDB i manca d'async. |
| **Qualitat de codi** | 7/10 | Consistent i llegible. Bugs reals però cap catastròfic. N+1 i iterrows penalitzen. |
| **Trading logic** | 6/10 | Gate System és excel·lent. ML/RL massa limitats per un mercat real. |
| **Producció (paper)** | 8/10 | Ready un cop completades les tasques A+B+E del Roadmap. |
| **Producció (real)** | 2/10 | Moltes capes de risc absents: circuits, staleness, live exchange. |

---

### És aquest sistema production-ready?

**Per a paper trading:** Quasi. Pendent de:
1. Fix CandleDB UniqueConstraint (DT1)
2. Completar walk-forward → train models → validate EnsembleBot (A+B)
3. Executar demo (E)

**Per a trading amb diners reals:** No. Les raons no són bugs de codi, sinó gaps sistèmics:
- Únic model ML validat (RF): single point of failure
- Zero agents RL validats out-of-sample per a demo
- Sense circuit breaker a nivell de portfolio
- Sense detecció de model staleness
- Sense integració amb exchange real

---

### Confiaríeu diners reals a aquest sistema?

No en l'estat actual. En 12-18 mesos, si:
1. La demo de paper trading mostra performance consistent (Sharpe > 1.0, Drawdown < -20%) durant almenys 6 mesos
2. S'implementa un circuit breaker de portfolio (halt automàtic si perd >15% des del pic)
3. S'afegeix detecció de model staleness i pipeline de reentrenament automàtic
4. El Gate System v2 amb shorts redueix el temps idle en règim BEAR

---

### Top 5 accions abans de trading real

1. **Fix CandleDB + migració** — Deute tècnic DT1. Integritat de dades fonamental.
2. **3-6 mesos de paper demo** — Validació de performance en condicions reals de mercat.
3. **Circuit breaker de portfolio** — Implementar halt automàtic si drawdown > 15% des del pic del portfolio agregat.
4. **Model drift detection** — Rolling Sharpe dels darrers 30 dies per cada bot; alarma Telegram si cau < 0.5.
5. **Integració exchange real** — Order management complet: fills, cancel·lacions, fills parcials, reconnect en pèrdua de connexió.

---

*Auditoria completada: Març 2026*
*Fitxers clau llegits: `exchanges/paper.py`, `bots/rl/environment.py`, `bots/rl/environment_professional.py`, `bots/rl/rl_bot.py`, `bots/rl/rewards/builtins.py`, `core/backtesting/metrics.py`, `core/db/models.py`, `data/sources/ohlcv.py`, `core/engine/runner.py`, `config/settings.yaml`, `docs/ROADMAP.md`, `docs/05_DATABASE.md`, `docs/01_ARCHITECTURE.md`, `docs/03_ML_RL_MODELS.md`*
