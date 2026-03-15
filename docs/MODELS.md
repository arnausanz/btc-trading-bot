# Models i Estratègies — Referència

> Descripció concisa de cada model, paràmetres clau i consideracions pràctiques.
> Per afegir un model nou: veure **[EXTENDING.md](./EXTENDING.md)**.
> Arquitectura compartida dels models tree-based: `bots/ml/base_tree_model.py`.
> Dataset PyTorch compartit (GRU/PatchTST): `data/processing/torch_dataset.py`.

---

## Classical Bots

Regles deterministes. Ràpids, interpretables, sense entrenament.

### DCABot
Compra una quantitat fixa cada N ticks, sense vendre mai. Dollar-Cost Averaging pur.
Ideal com a benchmark d'acumulació passiva; no té lògica de sortida.

| Paràmetre | Descripció | Default |
|-----------|-----------|---------|
| `buy_every_n_ticks` | Cada quantes candles compra | 24 |
| `buy_size` | Fracció del capital per compra | 0.1 |

---

### TrendBot
EMA crossover (ràpida > lenta → BUY) filtrat per RSI per evitar zones de sobrecompra.
El bot clàssic de seguiment de tendència; funciona bé en mercats direccionals.

| Paràmetre | Descripció | Default |
|-----------|-----------|---------|
| `ema_fast` | Períodes EMA ràpida | 9 |
| `ema_slow` | Períodes EMA lenta | 21 |
| `rsi_period` | Períodes RSI | 14 |
| `rsi_oversold` | Llindar oversold (BUY) | 40 |
| `rsi_overbought` | Llindar overbought (SELL) | 60 |

---

### GridBot
Compra quan el preu cau N% des d'un nivell, ven quan puja N%. Basada en Bollinger Bands.
Funciona bé en mercats en rang (laterals); perd en tendències forts.

| Paràmetre | Descripció | Default |
|-----------|-----------|---------|
| `bb_period` | Períodes Bollinger | 20 |
| `bb_std` | Desviació estàndard | 2.0 |
| `rsi_filter` | Filtre RSI per evitar falsos senyals | 50 |
| `level_size` | Fracció del capital per nivell | 0.2 |

---

> **Nota d'implementació:** Random Forest, XGBoost, LightGBM i CatBoost hereten
> de `BaseTreeModel` (`bots/ml/base_tree_model.py`), que implementa el bucle
> d'entrenament complet (TimeSeriesSplit 5-fold, MLflow, StandardScaler, pickle).
> Cada model fill aporta ~50 LOC: la construcció del model, `from_config()` i
> 3 mètodes d'1 línia per a l'experiment name, els paràmetres i el label de display.

---

### HoldBot
Compra tot el capital en la primera candle i no fa res més. Buy & Hold pur.
**Benchmark de referència.** Tot bot ha de superar-lo en Sharpe i Calmar.

_Sense paràmetres configurables._

---

### MeanReversionBot
Z-score del preu vs. mitjana mòbil + RSI extrem + filtre de volum.
Captura sobreextensions estadístiques que tendeixen a revertir; complementa GridBot i TrendBot.
Funciona bé en mercats laterals i en pull-backs dins de tendències.

**Estat de l'art per BTC (1h):** el Z-score és més precís que Bollinger Bands per detectar sobreextensió real. El filtre de volum protegeix de "catching a falling knife" en panic sells. Documentat en literatura 2020–2025 com un dels millors senyals de reversió per a cripto.

| Paràmetre | Descripció | Default |
|-----------|-----------|---------|
| `zscore_window` | Finestra per a la mitjana i σ | 30 |
| `zscore_entry` | Compra si Z < -zscore_entry | 1.8 |
| `zscore_exit` | Ven si Z > zscore_exit | 0.3 |
| `rsi_oversold` | Filtre RSI: no compra si RSI ≥ valor | 30 |
| `rsi_overbought` | Ven si RSI supera el valor | 70 |
| `volume_filter_multiplier` | No compra si volum > N × avg | 2.5 |
| `trade_size` | Fracció del capital | 0.4 |

---

### MomentumBot
Rate of Change (ROC) + confirmació de volum + MACD histogram.
Captura tendències nascents quan el preu accelera en volum confirmat.
Funciona bé en bull markets i en breakouts; complementa TrendBot (senyal més ràpid i selectiu).

**Estat de l'art per BTC (1h):** el momentum a 12–24h és un dels factors de retorn més documentats en cripto (Jegadeesh & Titman adaptat). La combinació ROC + volum confirmat + MACD redueix ~40% els falsos senyals respecte ROC sol. El filtre RSI evita entrades en sobrecompra.

| Paràmetre | Descripció | Default |
|-----------|-----------|---------|
| `roc_period` | Períodes per al ROC | 14 |
| `roc_threshold` | Compra si ROC > valor (%) | 1.5 |
| `volume_window` | Finestra per al volum mitjà | 20 |
| `volume_multiplier` | Volum ha de ser > N × avg | 1.3 |
| `macd_fast` | EMA ràpida del MACD | 12 |
| `macd_slow` | EMA lenta del MACD | 26 |
| `macd_signal` | Línia senyal del MACD | 9 |
| `rsi_max_entry` | No entra si RSI ≥ valor | 72 |
| `trade_size` | Fracció del capital | 0.55 |

---

## ML Models (via MLBot)

Tots usen `DatasetBuilder` per construir features + target. El target per defecte és `price_up_1pct_in_24h` (puja >1% en les properes 24h). S'accedeixen via `MLBot` amb `model_type` al YAML.

### Random Forest
Ensemble d'arbres de decisió; robusts a outliers i features irrelevants.
Ràpid, interpretable via feature importance, bon punt de partida.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `n_estimators` | 50–300 |
| `max_depth` | 3–10 |
| `min_samples_split` | 2–10 |
| `max_features` | sqrt, log2 |

---

### XGBoost
Gradient boosting optimitzat; estat de l'art per a dades tabulars.
Millor que RF en molts benchmarks; requereix més tuning.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `n_estimators` | 100–500 |
| `max_depth` | 3–8 |
| `learning_rate` | 0.01–0.3 |
| `subsample` | 0.6–1.0 |

---

### LightGBM
Gradient boosting de Microsoft; molt ràpid i eficient en memòria.
Similar a XGBoost però 5-10x més ràpid; preferit per datasets grans.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `n_estimators` | 100–500 |
| `max_depth` | 3–8 |
| `learning_rate` | 0.01–0.3 |
| `num_leaves` | 20–100 |

---

### CatBoost
Gradient boosting de Yandex; natiu per a features categòriques.
Menys tuning necessari; bona performance out-of-the-box.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `iterations` | 100–500 |
| `depth` | 4–8 |
| `learning_rate` | 0.01–0.3 |
| `l2_leaf_reg` | 1–10 |

---

### GRU (Gated Recurrent Unit)
Xarxa recurrent que modela dependències temporals seqüencials.
Capta patrons temporals que els models tabulars no veuen; lent d'entrenar.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `seq_len` | 24–168 |
| `hidden_size` | 32–256 |
| `num_layers` | 1–3 |
| `learning_rate` | 1e-4–1e-2 |
| `dropout` | 0.0–0.5 |

---

### PatchTST
Transformer per a séries temporals que divideix la seqüència en "patches".
Estat de l'art per a forecasting de séries temporals; necessita GPU per ser pràctic.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `patch_len` | 8–32 |
| `d_model` | 64–256 |
| `n_heads` | 4–8 |
| `n_layers` | 2–6 |
| `learning_rate` | 1e-4–1e-3 |

---

## RL Agents (via RLBot)

Aprenen per interacció amb un entorn Gym que simula el mercat. Entrenen 500k timesteps. Guarden l'agent a `models/`.

### PPO (Proximal Policy Optimization)
Agent de política discreta (BUY / SELL / HOLD). Estable i fàcil d'entrenar.
Bona primera opció per a RL; `BtcTradingEnvDiscrete`, inverteix el 100% del USDT disponible.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `n_steps` | 512–4096 |
| `batch_size` | 64–512 |
| `gamma` | 0.9–0.999 |
| `learning_rate` | 1e-4–1e-3 |
| `ent_coef` | 0.0–0.1 |

---

### SAC (Soft Actor-Critic)
Agent de política contínua: retorna un valor [-1.0, 1.0] que representa sizing dinàmic.
Més flexible que PPO; `BtcTradingEnvContinuous`, permet posicions parcials.

| Paràmetre | Rang Optuna |
|-----------|------------|
| `learning_rate` | 1e-4–1e-3 |
| `batch_size` | 64–512 |
| `tau` | 0.001–0.05 |
| `gamma` | 0.9–0.999 |
| `buffer_size` | 10k–100k |

---

### PPO Professional
Agent PPO de swing trading sense apalancament. **Dissenyat per aguantar posicions de 5–7 dies** aprofitant els cicles de BTC. Timeframe 12H per minimitzar l'impacte de les comissions.

Extensions sobre PPO baseline: acció `Discrete(5)` (HOLD / BUY_FULL / BUY_PARTIAL / SELL_PARTIAL / SELL_FULL), stop loss ATR dur (forçat per l'entorn, l'agent no el pot moure), position state a l'observació (PnL%, fracció, steps, drawdown), reward `professional_risk_adjusted` amb escalat ATR i penalitzacions de drawdown progressives.

**Features (18):** indicadors tècnics + règim (ADX-14, vol_ratio) + sentiment (Fear & Greed, funding rate). `obs_shape = 18 × 90 + 4 = 1.624`.

Pre-requisit de dades:
```bash
python scripts/download_data.py       # candles 12h
python scripts/download_fear_greed.py
python scripts/update_futures.py      # funding rate
```

Entrenament: `python scripts/train_rl.py --agents ppo_professional`

| Paràmetre | Valor / Rang Optuna |
|-----------|-------------------|
| `learning_rate` | 1e-4–1e-3 |
| `n_steps` | 1024–4096 |
| `batch_size` | 64–256 |
| `gamma` | 0.95–0.999 |
| `stop_atr_multiplier` | 1.5–3.0 |
| `lookback` | 60 o 90 |

---

### SAC Professional
Agent SAC de swing trading sense apalancament. Equivalent professional de `ppo_professional` però amb política contínua: l'agent decideix exactament quina fracció del capital posar en BTC en cada step de 12H.

Diferències clau: `action_space = Box([0,1])`, deadband ±0.05 (evita micro-rebalanceigs), replay buffer off-policy (aprèn de transicions passades). Hipòtesi: SAC pot aprendre sizing òptim fi (p. ex. 37% en BTC) mentre PPO pren decisions discretes més llegibles.

Mateixos features i pre-requisits que `ppo_professional`. `obs_shape = 18 × 90 + 4 = 1.624`.

Entrenament: `python scripts/train_rl.py --agents sac_professional`

| Paràmetre | Valor / Rang Optuna |
|-----------|-------------------|
| `learning_rate` | 1e-4–1e-3 |
| `batch_size` | 128–512 |
| `gamma` | 0.95–0.999 |
| `stop_atr_multiplier` | 1.5–3.0 |
| `lookback` | 60 o 90 |
| `buffer_size` | 100k (fix) |

---

## Comparativa ràpida

| Model | Velocitat | Paràmetres | Dades mínimes | Interpretable |
|-------|-----------|-----------|--------------|--------------|
| DCA | ⚡⚡⚡ | 2 | Cap | ✅ |
| Trend | ⚡⚡⚡ | 5 | 50 candles | ✅ |
| Grid | ⚡⚡ | 5 | 20 candles | ✅ |
| Hold | ⚡⚡⚡ | 0 | 1 candle | ✅ |
| MeanReversion | ⚡⚡⚡ | 7 | 60 candles | ✅ |
| Momentum | ⚡⚡⚡ | 9 | 80 candles | ✅ |
| RF | ⚡⚡ | 4 | 500+ | ✅ (feature imp) |
| XGBoost | ⚡⚡ | 4 | 500+ | ✅ (SHAP) |
| LightGBM | ⚡⚡⚡ | 4 | 500+ | ✅ (SHAP) |
| CatBoost | ⚡⚡ | 4 | 500+ | ✅ (SHAP) |
| GRU | 🐢 | 5 | 2000+ | ❌ |
| PatchTST | 🐢🐢 | 5 | 2000+ | ❌ |
| PPO | 🐢🐢 | 5 | 500k steps | ❌ |
| SAC | 🐢🐢 | 5 | 500k steps | ❌ |
| PPO Professional | 🐢🐢 | 6 | 500k steps · dades 12H | ❌ |
| SAC Professional | 🐢🐢 | 6 | 500k steps · dades 12H | ❌ |

---

## Models futurs (ROADMAP)

| Model | Tipus | Prioritat | Notes |
|-------|-------|-----------|-------|
| EnsembleBot | Meta | 🔴 | Combina senyals de múltiples bots |
| TFT | ML | 🟡 | Temporal Fusion Transformer |
| TD3 | RL | 🟡 | Millor que SAC en entorns sorollosos |
| DreamerV3 | RL | 🟢 | Model-based RL, molt eficient |

Veure **[ROADMAP.md](./ROADMAP.md)** per a detalls complets.

---

*Última actualització: Març 2026 · Versió 1.3*
