# Nova Política RL — Document de Disseny
**Estat:** Proposta per aprovació
**Data:** 2026-03-14
**Versió:** 2.0 (revisada)
**Basat en:** `docs/decisions/trading_policy_reference.md` + anàlisi de l'arquitectura existent

---

## 0. Objectiu d'aquest document

Aquest document explica **quines decisions de disseny** es prenen per implementar la nova política RL de trading professional, **per què** es prenen, i **com** encaixen amb l'arquitectura existent. No conté codi: és la base conceptual per revisar i aprovar abans de programar.

---

## 1. Diagnòstic de l'estat actual

### 1.1 Agents existents (PPO i SAC)

El sistema actual té dos agents RL operatius:

| | PPO | SAC |
|---|---|---|
| **Action space** | Discret: HOLD / BUY / SELL | Continu: fracció [0, 1] del capital en BTC |
| **Timeframe** | 1H | 1H |
| **Observation** | 9 features tècnics × 96 lookback | 9 features tècnics × 96 lookback |
| **Reward** | `sharpe` | `sortino` |
| **Position state** | No inclòs a l'observació | No inclòs |
| **Regime detection** | No | No |
| **Dades on-chain** | No | No |

### 1.2 Limitacions identificades

**Limitació 1 — Timeframe massa curt (1H).** Sense apalancament, les comissions de 0.2% per round-trip representen una fracció excessiva del profit esperat en moviments de 1H. BTC té cicles naturals de swing de 3–7 dies; operar a 1H és competir contra bots professionals d'alta freqüència amb accés a order flow en temps real, sense tenir cap avantatge. La freqüència ideal per a no-leveraged swing és de 12H.

**Limitació 2 — L'agent no "sap" on és.** L'agent veu l'historial de preus però no sap si té una posició oberta, quin és el seu P&L actual, ni quant temps fa que hi és. Sense aquest context, no pot aprendre a gestionar el trade un cop obert: quan sortir, quan aguantar, quan reduir exposició.

**Limitació 3 — No hi ha detecció de règim.** El mercat alterna entre tendència, rang i caos. Les estratègies òptimes són radicalment diferents per a cada règim. L'agent actual no té cap feature que li indiqui en quin règim es troba.

**Limitació 4 — El reward no penalitza el risc de ruïna.** Sharpe i Sortino premia el profit ajustat per volatilitat però no penalitzen activament drawdowns grans. Un drawdown del 20% requereix un +25% per recuperar-se; un del 50%, un +100%. Aquesta asimetria no és present als rewards actuals.

**Limitació 5 — L'action space no permet sortida parcial.** L'estratègia professional de "tancar 50% al primer objectiu i deixar córrer la resta" no es pot aprendre si l'agent només pot estar 100% dins o 100% fora.

**Limitació 6 — Les dades on-chain i el Fear & Greed Index no s'usen.** El sistema ja té totes aquestes dades a la DB (taules `futures`, `blockchain`, `fear_greed`). No les usar és deixar informació valuosa sobre la taula, especialment el Funding Rate i el Fear & Greed Index, que a 12H i en swing trading aporten un context molt rellevant.

---

## 2. Filosofia de la nova política

**Estil objectiu:** Swing trading sense apalancament sobre BTC/USDT, amb horitzó natural de 1–7 dies per posició.

**Freqüència:** Candles de **12H**. A aquesta freqüència:
- Comissions (0.2% round-trip) representen <10% d'un moviment típic de BTC de 2–4% per candle.
- El soroll intradiari queda filtrat per la pròpia granularitat.
- L'agent disposa de 45 dies d'historial amb un lookback de 90 candles.

**Durada de les posicions:** La posició es pot mantenir oberta tants steps com el mercat indiqui. No hi ha cap cap fix ("si portes +N candles, tanca sí o sí"). El `steps_in_position` és un *senyal informatiu* a l'observation —l'agent sap quant temps fa que hi és— però la decisió de sortir la pren sempre ell. BTC té cicles de swing de 5–7 dies (10–14 candles a 12H); la política ha de poder participar en moviments d'aquesta durada. L'única sortida forçada és el stop loss basat en ATR.

**Dos agents en paral·lel:** S'implementaran i entrenaran **dues variants** amb la mateixa arquitectura d'observació, reward i entorn, però diferent action space:
- `ppo_professional` — PPO amb Discrete(5)
- `sac_professional` — SAC amb continu [0, 1]

Ambdues competiran en el backtest de comparació per determinar quina és superior per a aquest règim específic.

---

## 3. Decisió 1 — Dos agents en paral·lel: PPO i SAC

En comptes d'escollir un sol agent a priori, es dissenyaran i entrenaran els dos. Compartiran l'observation space, el reward i l'entorn base, i diferiran exclusivament en l'action space i l'algorisme d'aprenentatge.

**Arguments per PPO (Discrete):**
- On-policy: aprèn del comportament que realment executa, millora la coherència de la política per a entorns no-estacionaris.
- Accions discretes faciliten la interpretació posterior ("per quin motiu l'agent ha triat BUY_PARTIAL i no BUY_FULL?").
- Més estable en les primeres fases d'entrenament, convergeix amb menys timesteps.

**Arguments per SAC (Continu):**
- L'acció única [0, 1] = "fracció de capital en BTC" és conceptualment la representació més directa del problema. 0.0 = fora, 1.0 = tot dins, 0.35 = posició parcial.
- Off-policy: pot aprendre de experiències antigues (replay buffer), molt eficient en mostres.
- No cal definir manualment els llindars del 50% — l'agent aprèn el sizing òptim directament.
- Millor per a entorns on la granularitat fina del sizing importa.

**Hipòtesi de partida:** SAC podria ser superior si el sizing continu li aporta avantatge real (aprèn "ara vull un 37% d'exposició"). PPO podria ser superior si la claredat de les decisions discretes facilita l'aprenentatge de la gestió del trade. L'experiment decidirà.

---

## 4. Decisió 2 — Action Spaces

### 4.1 PPO Professional: Discret(5)

| ID | Acció | Descripció | Equivalent humà |
|---|---|---|---|
| 0 | `HOLD` | No fer res | Esperar confirmació o mantenir posició sense canvis |
| 1 | `BUY_FULL` | Obrir posició al 100% del capital disponible | Entrada de convicció alta |
| 2 | `BUY_PARTIAL` | Obrir posició al 50% del capital disponible | Entrada exploratòria o mida reduïda per alta volatilitat |
| 3 | `SELL_PARTIAL` | Tancar el 50% de la posició oberta | Recollida de profit al primer objectiu |
| 4 | `SELL_FULL` | Tancar el 100% de la posició oberta | Stop activat o objectiu final assolit |

Nota: `BUY_PARTIAL` sobre una posició parcialment oberta = complet la posició al 100%. `SELL_PARTIAL` sobre posició plena = tanco el 50%. La mecànica exacta la gestiona l'entorn.

### 4.2 SAC Professional: Continu [0, 1]

Una sola dimensió d'acció: la **fracció objectiu del capital que ha d'estar en BTC** en tot moment.

- Si l'acció és 0.0 i no hi ha posició → no passa res (HOLD implícit).
- Si l'acció puja de 0.0 a 0.8 → l'entorn executa una compra fins arribar al 80%.
- Si l'acció baixa de 0.8 a 0.4 → l'entorn executa una venda parcial fins al 40%.
- Si l'acció baixa a 0.0 → tanca completament.

L'entorn aplica un **deadband** petit (±0.05) per evitar rebalancejos continus per fluctuacions mínimes de l'acció, que generarien costos excessius de comissions.

---

## 5. Decisió 3 — Timeframe i Lookback

**Timeframe:** 12H per a totes les features de mercat.

**Lookback:** 90 candles × 12H = **45 dies** d'historial per a cada observació. Aquest és el rang on es concentren els cicles de swing rellevants de BTC. Amb 22 features × 90 lookback → **obs_shape = 1.980 dimensions** (inferior als 2.112 originals gràcies a la reducció de lookback).

**Sobre la durada de les posicions:** `steps_in_position` es normalitza amb referència a 14 candles (= 7 dies), que és el màxim de durada raonablament esperat per a un swing trade de BTC. El valor es passa a l'observation com un float [0, 1+] (pot superar 1.0 si la posició dura més), i l'agent aprèn lliurement a interpretar-lo. **No hi ha cap tancament forçat per temps**, excepte l'ATR stop loss.

---

## 6. Decisió 4 — Observation Space: Position-Aware + Regime-Aware

L'observation space és idèntica per als dos agents (PPO i SAC). Es compon de quatre blocs:

### Bloc A — Features tècnics (12H)

| Feature | Descripció |
|---|---|
| `close` | Preu de tancament normalitzat |
| `rsi_14` | RSI 14 períodes |
| `macd` | Línia MACD |
| `macd_signal` | Senyal MACD |
| `macd_hist` | Histograma MACD (momentum) |
| `atr_14` | ATR 14 períodes (volatilitat base) |
| `atr_5` | ATR 5 períodes (volatilitat curta per vol_ratio) |
| `bb_upper_20` | Banda superior Bollinger |
| `bb_lower_20` | Banda inferior Bollinger |
| `bb_middle_20` | Banda central Bollinger |
| `ema_21` | EMA 21 — suport/resistència curt termini |
| `ema_50` | EMA 50 — suport/resistència mitjà termini |
| `volume` | Volum brut |
| `volume_norm_20` | Volum normalitzat (vol / rolling_mean_20) — distingeix ruptures reals de fakeouts |

### Bloc B — Indicadors de règim (NOUS)

| Feature | Càlcul | Interpreta |
|---|---|---|
| `adx_14` | Average Directional Index, 14 períodes | > 25 = tendència; < 20 = rang |
| `vol_ratio` | `atr_5 / atr_14` | > 1.2 = vol s'expandeix; < 0.8 = vol es comprimeix |

**Nota sobre el Hurst Exponent:** Descartat per a V1. Mesura la persistència estadística d'una sèrie temporal (H > 0.5 = trending, H < 0.5 = mean-reverting) i és conceptualment superior al `vol_ratio` per a detecció de règim. El motiu del descart és computacional: requereix l'anàlisi R/S sobre múltiples longituds de subfinestra (8, 16, 32, 64, 128...), amb una complexitat de O(n² log n) per a cada nova candle. Sobre 36.000 candles amb 15 trials d'Optuna, aquest càlcul es convertiria en el coll d'ampolla del preprocessing. El `vol_ratio` captura la mateixa intuïció (el mercat s'expandeix o es comprimeix) amb dos simples EMA. Candidat per a V2 si s'implementa amb una versió eficient (e.g., Hurst per DFA — Detrended Fluctuation Analysis).

### Bloc C — Dades on-chain i sentiment (existents a la DB)

| Feature | Font | Interpreta | Rellevància a 12H |
|---|---|---|---|
| `funding_rate` | `futures` table | > 0.1% = longs paguen (mercat sobrecomprat) | Alta: el funding s'acumula per sessions de 8H, molt llegible a 12H |
| `oi_btc` | `futures` table | OI creixent + preu pujant = tendència confirmada | Alta: captura la sobreextensió de longs/shorts en swing |
| `fear_greed_value` | `fear_greed` table | 0–25 = por extrema; 75–100 = cobdícia extrema | Alta: indicador diari, molt útil per a swing de dies |

*Nota tècnica:* El Fear & Greed Index és diari. A 12H, ambdues candles del dia prendran el mateix valor (forward-fill). Això és acceptable i coherent: el sentiment de mercat no canvia dos cops en un dia.

### Bloc D — Position State (NOU — el canvi arquitectural més important)

| Feature | Càlcul | Per què |
|---|---|---|
| `pnl_pct` | `(current_price - entry_price) / entry_price` si en posició, 0 si no | L'agent sap si guanya o perd en el trade actual |
| `position_fraction` | `btc_value / total_portfolio_value` ∈ [0, 1] | L'agent sap quina part del capital té en BTC (0 = fora, 1 = tot dins) |
| `steps_in_position` | Passos des de l'entrada / 14 (referència = 7 dies) | L'agent sap quant temps fa que hi és — aprèn time-based exit |
| `drawdown_pct` | `(portfolio - peak_portfolio) / peak_portfolio` ∈ [-1, 0] | L'agent veu el drawdown acumulat de la sessió |

Sense position state, l'agent pren decisions cegues respecte al seu estat. Amb aquests 4 valors, pot aprendre patrons com "si porto 10+ candles (5 dies) en posició amb P&L negatiu → SELL_FULL" o "si el drawdown supera -8% → reduir exposició". **És el canvi més crític de tota la nova política.**

### Resum del nou observation space

| Bloc | Features | Comptador |
|---|---|---|
| Tècnics 12H | close, rsi_14, macd, macd_signal, macd_hist, atr_14, atr_5, bb_upper_20, bb_lower_20, bb_middle_20, ema_21, ema_50, volume, volume_norm_20 | 14 |
| Règim | adx_14, vol_ratio | 2 |
| On-chain + sentiment | funding_rate, oi_btc, fear_greed_value | 3 |
| Position state | pnl_pct, position_fraction, steps_in_position, drawdown_pct | 4 |
| **TOTAL** | | **23 features** |

**obs_shape = 23 × 90 = 2.070 dimensions.**

*Les features del Bloc D s'injecten directament des de l'entorn (representen l'estat de la simulació, no el mercat), i es concatenen a l'observation vector en cada step.*

---

## 7. Decisió 5 — Reward Function: `professional_risk_adjusted`

La mateixa reward function s'usa per als dos agents (PPO i SAC). Codifica els principis del document de referència.

### Components

```
reward = base_pnl_reward
       + drawdown_penalty
       + overtrading_penalty
       + chaos_penalty
       + patience_bonus
```

**Component 1 — Base PnL (ATR-scaled)**

El PnL del step s'escala per l'invers de la volatilitat normalitzada (ATR). Quan la volatilitat és alta, un +1% de profit és relativament modest; quan és baixa, és significatiu. Això incentiva buscar trades amb millor R:R en entorns de baixa volatilitat.

Conceptualment: `base = (pnl_step / atr_normalized) × scaling`

**Component 2 — Drawdown penalty (quadràtica)**

Quan el drawdown acumulat supera el 5%, s'aplica una penalització progressiva:

- Drawdown < 5%: sense penalització
- Drawdown 5–10%: penalització lineal suau
- Drawdown > 10%: penalització quadràtica forta
- Drawdown > 20%: penalització molt agressiva

La progressió quadràtica reflecteix l'asimetria matemàtica real: un -20% requereix un +25% per recuperar; un -50%, un +100%.

**Component 3 — Overtrading penalty**

A 12H, el ritme natural per a swing trading és d'1–2 trades per setmana. La penalització s'activa si l'agent supera un màxim de 3 trades en finestres de 14 candles (= 1 setmana a 12H). Força selectivitat i evita el "revenge trading".

**Component 4 — Chaos penalty**

Quan `vol_ratio > 1.5` (la volatilitat s'ha expandit >50% respecte a la seva mitjana recent) i l'agent obre una posició nova, rep una penalització. Codifica la regla "Don't trade the news / reduce size in chaos" del document de referència.

**Component 5 — Patience bonus**

Un petit bonus quan l'agent fa HOLD i el `vol_ratio < 0.8` (mercat comprimit, breakout no confirmat). Ha de ser molt petit —una fracció del cost d'una comissió— per no crear incentius perversos d'inacció permanent.

### Per què no simplement Sharpe/Sortino?

Els rewards actuals mesuren qualitat del retorn a posteriori sobre una finestra curta (20 steps). No penalitzen drawdowns grans explícitament, no distingeixen entre règims de mercat, i no donen cap senyal de si l'agent hauria d'operar o esperar. El nou reward té components prospectius (no operes en caos) i retrospectius (drawdown causat), que s'alineen millor amb la filosofia "Plan the trade, trade the plan".

---

## 8. Decisió 6 — Entorn Compartit: `BtcTradingEnvProfessional`

Un sol entorn base (`BtcTradingEnvProfessional`) amb dues variants d'action space, configurables per YAML. Les extensions clau respecte als entorns actuals:

### Extensió A — Gestió de posicions parcials

L'entorn trackeja `position_fraction` (0.0, 0.5, 1.0 per PPO; qualsevol valor [0,1] per SAC) i aplica la mecànica de compres i vendes parcials amb costos de comissió per cada transacció parcial.

### Extensió B — Position state a l'observation

Els 4 features del Bloc D s'injecten com a últimes 4 dimensions del vector d'observació en cada step, normalitzats correctament.

### Extensió C — ATR Stop Loss (hard, sense acció de l'agent)

L'entorn calcula un stop loss implícit en l'obertura de cada posició:

```
stop_price = entry_price × (1 - stop_atr_multiplier × atr_14_at_entry / entry_price)
```

Si el preu cau per sota del stop, l'entorn executa un `SELL_FULL` automàtic independentment de l'acció de l'agent. L'agent no té cap acció per moure el stop. El multiplicador (`stop_atr_multiplier`, default 2.0 per a 12H) s'exposarà al `search_space` d'Optuna.

A 12H, el stop basat en 2×ATR_14 dóna prou marge per a la variabilitat normal de swing sense ser aturat per soroll intradiari.

### Extensió D — Deadband per SAC

Per a la variant SAC, l'entorn aplica un deadband de ±0.05 sobre l'acció. Si la nova `target_fraction` difereix en menys de 0.05 de la `current_fraction`, no s'executa cap ordre. Evita rebalancejos continus i costos de comissions excessius.

---

## 9. Nous indicadors tècnics a implementar

### 9.1 ADX (Average Directional Index)

Mesura la força d'una tendència independentment de la direcció. Base: Directional Movement (+DM, -DM) i True Range sobre 14 períodes. Retorna columna `adx_14`. Rang: 0–100 (> 25 = tendència; < 20 = rang lateral).

### 9.2 ATR curt (ATR-5)

Variant de 5 períodes de l'ATR existent per calcular el `vol_ratio`. Retorna columna `atr_5`.

### 9.3 Vol Ratio

`atr_5 / atr_14`. Retorna columna `vol_ratio`. Rang típic: 0.4–2.5. No requereix implementació pròpia, és una divisió de les dues columnes anteriors.

### 9.4 Volume Normalized

`volume / volume.rolling(20).mean()`. Retorna columna `volume_norm_20`. Indica si el volum actual és alt (>1) o baix (<1) respecte a la seva mitjana dels últims 20 períodes.

---

## 10. Fitxers a crear i modificar

### Fitxers nous

| Fitxer | Contingut |
|---|---|
| `bots/rl/environment_professional.py` | Classe `BtcTradingEnvProfessional` amb suport per Discrete(5) i Continuous. Gestió parcial, position state, ATR stop, deadband SAC. |
| `bots/rl/rewards/professional.py` | Funció `professional_risk_adjusted` registrada amb `@register("professional")`. |
| `config/models/ppo_professional.yaml` | Config PPO: Discrete(5), 12H, 23 features × 90 lookback, reward `professional`. |
| `config/models/sac_professional.yaml` | Config SAC: Continuous[0,1], 12H, 23 features × 90 lookback, reward `professional`. |

### Fitxers a modificar

| Fitxer | Canvi |
|---|---|
| `data/processing/technical.py` | Afegir `adx(period=14)`, `atr(period=5)`, `volume_normalized(period=20)`. Calcular `vol_ratio` com a columna derivada. Afegir tot a `compute_features()`. |
| `bots/rl/trainer.py` | Registrar `"ppo_professional"` i `"sac_professional"` com a entorns vàlids. |
| `bots/rl/rl_bot.py` | Gestionar les noves accions 3 i 4 (SELL_PARTIAL, SELL_FULL) per PPO, i el valor continu per SAC professional. |
| `scripts/train_rl.py` | Afegir els dos nous agents a `AVAILABLE_AGENTS`. |
| `scripts/run_comparison.py` | Afegir `ppo_professional` i `sac_professional` a `BOT_REGISTRY`. |

---

## 11. Resum de decisions i alternatives descartades

| Decisió presa | Alternative descartada | Per què s'ha descartat |
|---|---|---|
| Dos agents: PPO Discrete(5) + SAC Continu | Escollir un sol agent a priori | Cap de les dues opcions és clarament superior sense dades; l'experiment decideix |
| Timeframe 12H | 1H (existent), 4H, 1D | 1H: massa comissions per swing no-leveraged. 1D: massa poc granular per aprendre timing. 12H: balanç òptim. |
| Posicions obertes fins que l'agent decideixi (o stop ATR) | Cap fix d'1 dia | BTC té cicles de 5–7 dies. Un cap artificial eliminaria la capacitat de participar en els moviments principals. |
| 23 features, 1 sola timeframe (12H) | Multi-timeframe (12H + 1D) | Multi-timeframe requeriria reestructurar completament l'entorn (finestres de lookback paral·leles). V2. |
| Vol_ratio com a proxy de règim | Hurst Exponent | Hurst és computacionalment car (O(n² log n) rolling). Vol_ratio captura intuïció equivalent en O(n). V2 si s'implementa DFA. |
| ATR Stop Loss a l'entorn | Stop loss al reward | Un hard stop al simulador és més fidel a la realitat; el reward no hauria de gestionar restriccions operatives. |
| Deadband ±0.05 per SAC | Cap deadband | Sense deadband, SAC genera rebalancejos constants per fluctuacions petites de l'acció, destruint el P&L en comissions. |
| Fear & Greed a 12H (forward-fill) | No usar-lo | A 12H i amb swing trading, el sentiment diari és molt rellevant. La pèrdua de resolució (forward-fill) és acceptable. |
| Drawdown penalty quadràtica | Penalty lineal | Reflecteix l'asimetria matemàtica real del drawdown (−50% necessita +100% per recuperar). |
| Position state a l'observation | Cap position state | Sense saber on és, l'agent no pot aprendre gestió del trade. És el canvi conceptual més important de V1. |

---

## 12. Riscos i mitigacions

**Risc 1 — L'agent PPO aprèn a no operar mai.** Amb 5 accions i el patience_bonus, pot col·lapsar en HOLD constant. Mitigació: monitorar `val_trades` durant Optuna. Si < 5 trades en tot el backtest de validació, ajustar el pes del patience_bonus cap avall.

**Risc 2 — El SAC oscil·la contínuament per sobre del deadband.** Si l'agent aprèn a fer petites modificacions de posició cada step, els costos de comissions s'acumulen. Mitigació: el deadband de 0.05 és el primer filtre. Si no és suficient, l'overtrading_penalty actua com a segon filtre.

**Risc 3 — El stop ATR tanca massa posicions prematurament.** A 12H, la volatilitat típica de BTC és alta. Un stop de 2×ATR_14 pot ser suficientment ample, però caldria verificar-ho. Mitigació: el multiplicador del stop (`stop_atr_multiplier`) entra al `search_space` d'Optuna (rang: 1.5–3.0).

**Risc 4 — Gaps a les dades on-chain.** El Funding Rate i l'OI poden tenir períodes sense dades. Mitigació: verificar la cobertura temporal de les taules `futures` i `fear_greed` abans d'iniciar l'entrenament. El `merge_asof` del FeatureBuilder fa forward-fill automàticament.

**Risc 5 — obs_shape elevada alenteix l'entrenament.** 2.070 dimensions vs. les 864 actuals. Mitigació: provar primer amb `lookback=60` (23×60=1.380) per als primers experiments de validació. Pujar a 90 per a l'entrenament definitiu.

---

## 13. Mètriques d'èxit

Per considerar la nova política millor que els agents actuals, ha de superar-los en **almenys tres** de les quatre mètriques durant el backtest de validació:

| Mètrica | Descripció | Target |
|---|---|---|
| `val_return_pct` | Retorn total en el període de validació | ≥ millor entre PPO i SAC actuals |
| `val_max_drawdown_pct` | Màxim drawdown en el període de validació | ≤ 80% del pitjor entre PPO i SAC actuals |
| `profit_factor` | Suma de guanys / suma de pèrdues (nou) | > 1.5 |
| `val_trades` | Nombre de trades (eficiència) | Rang raonable: 1 trade cada 3–7 dies |

La mètrica `profit_factor` és nova i requerirà afegir-la al codi d'avaluació del trainer/backtester.

---

## 14. Seqüència d'implementació

Un cop aprovat el disseny:

1. **Indicadors tècnics** — ADX, ATR-5, vol_ratio, volume_norm_20 a `technical.py`. Base de tot; sense això les features no existeixen.
2. **Reward function** — `professional_risk_adjusted` a `bots/rl/rewards/professional.py`. Testar amb valors sintètics.
3. **Entorn** — `BtcTradingEnvProfessional` amb totes les extensions. Validar amb smoke test.
4. **Configs YAML** — `ppo_professional.yaml` i `sac_professional.yaml`.
5. **Registre** — Actualitzar `trainer.py`, `rl_bot.py`, `train_rl.py`, `run_comparison.py`.
6. **Entrenament de prova (smoke)** — 50k steps cada agent per verificar que l'entorn no té bugs i la reward és raonable.
7. **Entrenament complet + Optuna** — 500k steps, 15 trials per agent.
8. **Comparació final** — `run_comparison.py` amb PPO, SAC, ppo_professional i sac_professional.

---

*Document preparat per a revisió. Versió 2.0 — Pendent d'aprovació per Arnau Sanz.*
