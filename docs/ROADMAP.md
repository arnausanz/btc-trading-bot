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
| Tests (smoke + unit) | ✅ 123 tests | Smoke + unit passen |
| Dashboard (Streamlit) | ⚠️ Bàsic | Només preus de la BD |
| Feature Store | ❌ Pendent | Placeholder reservat |
| Risk Manager | ❌ Pendent | Placeholder reservat |
| Dades externes | ❌ Pendent | Fear & Greed, on-chain |
| EnsembleBot | ❌ Pendent | Meta-capa de combinació |
| Tests d'integració | ⚠️ Buits | Necessiten BD real |

---

## Prioritats — ordre recomanat

### 🔴 Crític (fer primer)

#### 1. Corregir les mètriques de backtesting
Les mètriques actuals estan mal calculades i fan que tots els resultats siguin incomparables amb benchmarks estàndard.

- **Sharpe Ratio:** ha d'usar `sqrt(252*24)` per a dades horàries (no `sqrt(365)`)
- **Calmar Ratio:** cal calcular dies reals, no `len(df)`
- **Win Rate:** cal mesurar round-trips (buy→sell), no canvis positius entre ticks
- **Afegir:** Sortino Ratio, Profit Factor, Max Consecutive Losses, Average Trade Duration

_Fitxers afectats:_ `core/backtesting/metrics.py`

---

#### 2. Walk-forward backtesting real
El backtesting actual fa un sol split train/test. Cal implementar validació amb finestra lliscant.

- Implementar `TimeSeriesSplit` amb N folds sobre el temps
- Assegurar que les últimes N setmanes mai s'usen per a optimització
- Benchmark automàtic: HoldBot s'executa en cada fold

_Fitxers afectats:_ `core/backtesting/optimizer.py`, `scripts/run_comparison.py`

---

#### 3. Re-executar tots els backtests amb mètriques correctes
Quan les mètriques siguin correctes, els resultats anteriors no seran vàlids. Cal reiniciar els backtests.

---

### 🟡 Prioritat mitjana

#### 4. Fear & Greed Index com a feature

**API disponible (gratuïta):** `https://api.alternative.me/fng/`

- Descarregar historial complet (des de 2018): `?limit=2000&format=json`
- Valor actual: `?limit=1&format=json`
- Retorna un valor 0-100 i una classificació: `Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`

**Implementació:**
1. Crear `data/sources/fear_greed.py` amb `fetch_history()` i `fetch_current()`
2. Crear taula `fear_greed` a la BD: `(timestamp, value, classification)`
3. Afegir script `scripts/update_fear_greed.py` (cridat per cron diàriament)
4. Integrar com a feature numèrica al `DatasetBuilder` i `ObservationBuilder`
5. Els models ML reben `fear_greed_value` com a columna del feature vector

Veure recepte complet a **[EXTENDING.md → Secció 5](./EXTENDING.md)**.

---

#### 5. EnsembleBot

Meta-bot que combina senyals de múltiples sub-bots sense necessitat d'entrenament addicional.

**Polítiques a implementar:**
- `majority_vote`: si >50% de bots diuen BUY → BUY
- `weighted`: pes proporcional al Sharpe dels últims N dies
- `unanimous`: només actua si TOTS coincideixen (molt conservador)
- `stacking`: un model ML de 2a capa entrena sobre les prediccions dels bots base

**Fitxers nous:** `bots/ensemble/ensemble_bot.py`, `config/bots/ensemble.yaml`

Veure disseny a **[EXTENDING.md → Secció 6](./EXTENDING.md)**.

---

#### 6. Nous bots clàssics

| Bot | Lògica | Prioritat |
|-----|--------|-----------|
| MeanReversionBot | RSI extrems (<20 / >80) + Z-score del preu | 🟡 |
| MomentumBot | Rate of Change (ROC) + confirmació de volum | 🟡 |
| BreakoutBot | Suports/resistències (pivot points) + ATR per confirmar | 🟢 |

---

#### 7. Nous models ML

| Model | Tipus | Descripció |
|-------|-------|-----------|
| Temporal Fusion Transformer (TFT) | Deep Learning | Estat de l'art per séries temporals financeres; suporta covarites externes, multi-horizon |
| N-BEATS / N-HiTS | Deep Learning | Arquitectures pures sense RNN; molt eficients per forecasting |
| TabNet | Tabular | Supera XGBoost en alguns benchmarks; interpretable via atenció |

---

#### 8. Millores RL

| Millora | Descripció | Prioritat |
|---------|-----------|-----------|
| Reward shaping | Penalitzar drawdown i trading excessiu (calmar-based reward) | 🔴 |
| Position sizing PPO | L'agent discret hauria de poder invertir fraccions, no sempre el 100% | 🔴 |
| TD3 (Twin Delayed DDPG) | Millor que SAC per a entorns molt sorollosos | 🟡 |
| Curriculum learning | Entrenar primer en tendències clares, després en rangs | 🟡 |
| Multi-step returns | Millora la propagació de recompenses a llarg termini | 🟡 |
| DreamerV3 | Model-based RL; aprèn un model intern del mercat | 🟢 |

---

### 🟢 Futur (quan hi hagi dades demo de qualitat)

#### 9. Correccions del DemoRunner

- **Sincronització de candles:** el bot ha d'actuar quan tanca una candle, no cada minut. Implementar detecció de "nova candle tancada"
- **Persistir estat intern:** `_in_position` i `_tick_count` han de persistir a la BD i restaurar-se en reiniciar

---

#### 10. Infraestructura de dades externes

Crear `DataSourceRegistry`: un sistema d'extensions on cada font de dades s'integra com a mòdul independent.

**Fonts candidates:**
- Fear & Greed Index (gratuïta) ← **PRIORITAT ALTA** (veure punt 4)
- Sentiment de Twitter/X (API de pagament)
- On-chain: SOPR, MVRV, exchange flows (Glassnode API — de pagament)
- Dominàncid BTC (CoinGecko API — gratuïta)
- Google Trends (pytrends — gratuïta)
- Funding rates perpetus (Binance API — gratuïta)

---

#### 11. Dashboard complet

Quan hi hagi dades de demo reals:
- Portfolios en temps real per bot
- PnL acumulat i diari (gràfic)
- Drawdowns visuals
- Trade log amb filtre per bot
- Comparativa vs BTC spot (benchmark)
- Matriu de correlació de retorns entre bots
- Alertes de drawdown configurable

---

#### 12. Feature Store

Pre-calcular TOTES les features una sola vegada i emmagatzemar-les. Avui cada bot recalcula les mateixes EMAs i RSIs. El Feature Store elimina la duplicació i assegura consistència.

---

#### 13. Desplegament 24/7

- Oracle Cloud Free Tier (o equivalent)
- Cron per `update_data.py` cada hora
- Systemd per restart automàtic del DemoRunner
- Watchdog extern amb Telegram per caigudes del sistema

---

## Seqüència recomanada d'implementació

```
[CRÍTIC] Corregir mètriques → Re-executar backtests
              ↓
[CRÍTIC] Walk-forward real
              ↓
[IMPORTANT] Fear & Greed Index (dades + feature + script)
              ↓
[IMPORTANT] EnsembleBot (majority vote primer)
              ↓
[IMPORTANT] MeanReversionBot + MomentumBot
              ↓
[ML] TFT / TD3 / Reward shaping RL
              ↓
[SISTEMA] Corregir DemoRunner · Feature Store
              ↓
[DEPLOY] Servidor 24/7 · Dashboard complet
```

---

*Última actualització: Març 2026 · Versió 1.1*
