# Professional Trading Knowledge → RL Policy Reference
**Document d'investigació per al disseny d'un agent RL de trading BTC/USDT**

---

## 0. Com llegir aquest document

Aquest document sintetitza com opera un trader professional i el tradueix parcialment a les preguntes clau que cal respondre per dissenyar una **política RL**. Cada secció conté:
- **Coneixement del trader** → el que fa un humà i per què
- **Implicació per a RL** → com afecta el disseny de l'agent

L'objectiu no és implementar directament, sinó tenir la base conceptual sòlida per prendre decisions de disseny informades.

---

## 1. Gestió del Risc Professional

### 1.1 Position Sizing

Un trader professional mai arriba al mercat pensant "quant puc guanyar?" — sempre pensa primer "quant puc perdre?".

**Mètodes principals:**

| Mètode | Fórmula | Quan s'usa |
|---|---|---|
| Fixed fractional | Risc = 1-2% del capital total | Base per a la majoria |
| Kelly Criterion | `f = (bp - q) / b` | Quan tens estadístiques fiables |
| Kelly fraccionat (half-Kelly) | `f_k / 2` | Cripto (alta incertesa) |
| Volatilitat ajustada | `size = risk_capital / (ATR_multiplier × price)` | Règim de vol alt |

- `b` = odds (guany potencial / pèrdua potencial)
- `p` = probabilitat de guany estimada
- `q` = 1 - p

**Ajustos dinàmics:**
- Alta convicció (confluència de senyals) → mida 1.5–2× la base
- Alta volatilitat → mida reduïda inversament a l'ATR
- Correlació > 0.7 amb posició oberta → NO obrir nova posició al mateix actiu/sector

**Límit de ruïna:**
- Drawdown >10-20% → reduir exposició
- Pèrdua del 50% → requereix +100% per recuperar (asimetria crítica)

### 1.2 Stop-Loss: Tipologies

**ATR-based stop:**
```
stop = entry - (1.5 × ATR_14)   # per a longs
stop = entry + (1.5 × ATR_14)   # per a shorts
```
Filtra el soroll de mercat sense ajustar-se a l'estructura.

**Structure-based stop:**
Més enllà de màxims/mínims de swing, nivells de suport/resistència clau. Respecta la lògica del mercat.

**Time-based stop:**
Si la posició no es mou a favor en N barres, sortida per oportunitat cost — el capital té un cost.

**Hard vs. Mental stops:**
- Hard stops: ordre automàtica al exchange. Necessaris en mercats ràpids (cascades cripto, liquidacions).
- Mental stops: flexibilitat per a il·líquids, però exigeixen disciplina de ferro. En cripto, pràcticament sempre hard.

### 1.3 Gestió de Cartera (Multi-trade)

- **VaR agregat**: la suma d'exposicions no ha de superar el 6-10% del capital
- **Heatmaps de correlació**: si dos actius es mouen junts (ρ > 0.7), comptar-los com una sola posició de facto
- **Covariance matrix**: per ajustar mides quan hi ha múltiples posicions obertes

---

## 2. Entrades: Anatomy d'una Bona Entrada

### 2.1 Multi-timeframe Confluence (HTF → LTF)

El trader professional **mai** mira un sol timeframe. El flux de decisió és:

```
HTF (4H / Daily / Weekly)
    ↓ determina el biaix direccional (tendència alcista / baixista / neutre)
MTF (1H / 4H)
    ↓ identifica la zona d'entrada (suport, Fibonacci, estructura)
LTF (15m / 5m)
    ↓ cerca el trigger exacte d'entrada (ruptura, rebuig de nivell, patró de vela)
```

**Regla fonamental**: el LTF mai contradiu el HTF. Si el Daily és bajista, no busques longs en 15m.

### 2.2 Confirmació vs. Anticipació

| | Confirmació | Anticipació |
|---|---|---|
| **Quan entres** | Després de la ruptura validada | Abans de la ruptura, al punt clau |
| **Avantatge** | Menys fakeouts, més certesa | Millor R:R, stop més ajustat |
| **Desavantatge** | Pots perdre el moviment inicial | Més probabilitat de parada prematura |
| **Usat per** | Swing traders, bots conservadors | Scalpers, traders experimentats |

### 2.3 Com Distingir Ruptura Real de Fakeout

Senyals de ruptura real:
- Volum >2× la mitjana en la vela de ruptura
- Tancament de vela més enllà del nivell (no sols intradía)
- Order flow: agressió compradora al footprint chart
- Retest del nivell com a nova suport/resistència (amb volum baix en el pull-back)

Senyals de fakeout:
- Volum baix en la ruptura
- Ràpida reversió a la zona rompuda
- Manca de seguiment en les barres posteriors
- Liquidació visible (spike breu i retorn)

### 2.4 Risk/Reward Mínim Exigit

```
R:R = (target - entry) / (entry - stop)
```

- Mínims acceptats: R:R > 1:2 (perds 1 per guanyar 2+)
- Entrada invalidada si R:R < 1:1.5
- Afegit al filtre: win_rate × avg_win > (1 - win_rate) × avg_loss

---

## 3. Sortides: Quan i Com Tancar

### 3.1 Estratègies de Sortida

**Fixed target (R-múltiples):**
```
target = entry + (stop_distance × R)   # on R = 2, 3, etc.
```
Simple, definit a priori. Millor per backtest net.

**Trailing stop (Chandelier Exit):**
```
stop = highest_high - (3 × ATR)   # per a longs
```
Deixa córrer guanys en tendències fortes. El risc és ser aturat en una correcció normal.

**Structure-based exit:**
Sortida quan el preu perd un màxim/mínim de swing clau del HTF. Menys mecànic, més contextual.

### 3.2 Escala de Sortida (Partial Profit-Taking)

Estratègia típica de professional:
1. Tancar **50% de la posició** en el primer objectiu (1R o 1.5R)
2. Moure el stop a break-even per a la resta
3. Deixar córrer la resta amb trailing stop o fins a objectiu HTF

Benefici: assegura profit, elimina el risc de pèrdua en el trade, permet participar en tendències grans.

### 3.3 El Trade "Correcte però Prematur"

Un dels reptes més difícils: tens raó en la direcció però el mercat no s'ha mogut encara.

Opcions del professional:
- Si el HTF segueix intacte → aguantar amb trailing stop ample
- Si el context ha canviat → sortida parcial i reobertura en millors condicions
- Time-based exit: si en N barres no s'ha activat la tesi, tancar i revisar

### 3.4 Asimetria Fonamental

Un sistema amb R:R = 1:3 pot ser rendible amb **només un 33% de win rate**:
```
Expected Value = (0.33 × 3R) - (0.67 × 1R) = 0.99R - 0.67R = +0.32R per trade
```

Implicació: tallar pèrdues ràpid i deixar guanys créixer és **matemàticament superior** a intentar tenir raó el màxim possible.

---

## 4. Estils de Trading: Comparativa

### 4.1 Taula Comparativa

| | Scalping | Momentum/Intraday | Swing |
|---|---|---|---|
| **Timeframes** | 1m – 5m | 15m – 1H | 4H – Daily |
| **Durada posició** | Segons – minuts | Hores | Dies – setmanes |
| **Win rate necessari** | >60% | >50% | >40% |
| **R:R típic** | 1:1 – 1:1.5 | 1:1.5 – 1:2.5 | 1:3 – 1:5 |
| **Senyals principals** | Order flow, tape, L2 | MACD acceleration, EMA cross + vol | HTF structure, macro, on-chain |
| **Millors condicions** | Alta liquiditat, rang estret | Tendència intraday, expansió de vol | Tendència persistent, baixa soroll |
| **Fallades** | Notícies, alta vol, spread ample | Fade post-clímax, gaps d'obertura | Whipsaws en rang, reversals |

### 4.2 Signals per Estil (Detall)

**Scalping:**
- Desequilibri al carnet d'ordres (bid/ask imbalance)
- Liquidacions de perps visibles en temps real
- Repetició de patrons a nivells de liquiditat (equal highs/lows)

**Momentum Intraday:**
- MACD histogram accelerant en la mateixa direcció que el preu
- EMA de curta longitud creuant les de llarga en expansió de vol
- Primer 15-30 minuts post-obertura (London/NY overlap per a cripto: 14h-18h UTC)

**Swing:**
- Ruptura d'estructures HTF amb tancament mensual/setmanal per sobre
- Dades on-chain: flux net a exchanges, SOPR > 1 (realitzant beneficis = possible cim)
- Context macro: decisió Fed, CPI, aprovacions ETF

---

## 5. Regime Detection: El Context Ho Canvia Tot

### 5.1 Els Tres Règims Principals

| Règim | Característiques | Estratègia òptima |
|---|---|---|
| **Tendència** | ADX > 25, Hurst > 0.5, vol expansió | Seguir tendència, stops amples, objectius grans |
| **Rang** | ADX < 20, preu oscil·lant entre S/R, vol contracta | Mean reversion, R:R ajustats, sortida ràpida |
| **Alta Volatilitat/Caos** | ATR > 2× avg, moviments erràtics | Reduir mida, no operar o hedging |

### 5.2 Indicadors de Detecció

**ADX (Average Directional Index):**
- >25 → tendència present (no diu la direcció)
- <20 → mercat en rang

**Hurst Exponent:**
- >0.5 → persistència (tendència)
- = 0.5 → random walk
- <0.5 → mean-reverting (rang)

**Volatility Ratio:**
```
vol_ratio = ATR_short / ATR_long   # e.g., ATR_5 / ATR_20
```
- Ratio pujant → règim canviant, vol s'expandeix
- Ratio baixant → compressió (breakout imminent o rang prolongat)

### 5.3 Adaptació Mid-Trade

Si el règim canvia mentre tens una posició oberta:
- Vol s'expandeix inesperat → ampliar stop 50%, tancar parcialment
- ADX cau per sota 20 → tendència s'esgota, sortida parcial o total
- Ruptura de l'estructura HTF → invalidació de la tesi, sortida

### 5.4 Senyals de Fi de Tendència

- **Climax volume**: volum molt superior al normal en la darrera vela
- **Failed retest**: el preu torna al nivell romputs però no aguanta
- **Open Interest (OI) spike en perps**: sobreextensió de longs/shorts
- **Divergència RSI/MACD**: preu fa nou màxim però els indicadors no

---

## 6. Indicadors: Humans vs. Bots

### 6.1 Que Usen els Traders Discrecionals

**Alta prioritat:**
- Price action pur (estructura de màxims/mínims, veles individuals)
- Volume profile / VPOC (on s'ha negociat més volum)
- VWAP (preu mitjà ponderat per volum, intraday)
- EMA ribbons (suport/resistència dinàmics, 21, 50, 200)
- ATR (calibrar volatilitat i mides)

**Ús com a filtre, no com a trigger:**
- RSI (detectar sobrecompra/sobrevenda en context HTF)
- MACD (confirmar momentum, divergències)

### 6.2 Que Usen els Bots Algorítmics

- Bollinger Bands squeezes (compressió → breakout)
- RSI mean-reversion amb entrades mecàniques
- EMA crossovers simples
- Z-score de spreads (per a estratègies de parells)
- Correlació estadística d'actius

### 6.3 El Gap: On els Humans Guanyen als Bots

| Capacitat humana | Dificultat d'encodar |
|---|---|
| Tape reading (microstructura en temps real) | Molt alta |
| Reconeixement de contexte (notícia + preu + hora) | Alta |
| Time-of-day edges (London/NY overlap) | Mitjana (encodable) |
| "Sensació de mercat" (flux ordenat vs. caòtic) | Molt alta |
| Adaptació instantània a règim canviant | Alta |

**Consell dels professionals**: els indicadors son lagging per definició. Usar-los com a filtres de context, no com a triggers d'entrada. Over-reliance → paràlisi o entrades tardanes.

---

## 7. Cripto-Específic (BTC/USDT)

### 7.1 Diferències Estructurals vs. Equities/Forex

- **24/7**: no hi ha tancament. El risc overnight és real i difícil de hedgejar.
- **Liquiditat variable**: diumenge nit UTC → spread amplíssim, slippage elevat.
- **Liquidació en cascada**: quan es liquiden posicions apalancades, el preu cau abruptament i reactiva més liquidacions. Predictible amb dades de Coinglass.
- **Sense market maker institucional**: el BTC pot moure's 5-10% en minuts sense notícia externa.

### 7.2 Senyals On-Chain rellevants

| Signal | Interpretació |
|---|---|
| Flux net a exchanges (entrada BTC) | Possible venda imminent, pressió baixista |
| Flux net des d'exchanges (sortida BTC) | Acumulació, pressió alcista |
| SOPR > 1 | Inversors realitzant beneficis (pot ser sostre) |
| SOPR < 1 | Vendes en pèrdues (possible fons de capitulació) |
| OI (Open Interest) pujant amb preu pujant | Tendència confirmada |
| OI pujant amb preu baixant | Shorts acumulant (possible squeeze) |
| Funding rate > 0.1% | Perps longs pagant: mercat sobrecomprat |
| Funding rate < -0.1% | Perps shorts pagant: mercat sobrevenut |

### 7.3 Gestió d'Esdeveniments d'Alta Volatilitat

- CPI, decisió Fed, notícies ETF → spreads s'amplien, slippage empitjora
- Estratègia professional: 3-5× stop normal O simplement no operar 30 min abans/després
- "Don't trade the news" és la regla 1 per a molts professionals

---

## 8. Frameworks Mentals (Traslladables a Policy)

### 8.1 Checklist Pre-Trade del Professional

```
1. Quin és el biaix del HTF? (alcista / baixista / neutre)
2. Estic operant en la direcció del HTF?
3. El R:R és > 1:2?
4. On col·loco el stop? Té sentit estructuralment?
5. Quin % del capital arrisca aquest trade?
6. La meva exposició total amb aquest trade és acceptable?
7. Quin és el règim actual? Adapto la mida i els targets?
8. Hi ha un event de risc imminent (macro, on-chain)?
```

### 8.2 Principis Operatius dels Guanyadors (Market Wizards)

- **"Plan the trade, trade the plan"**: les decisions s'han de prendre ABANS d'entrar, no durant.
- **Asimetria**: buscar situacions on si tens raó guanyes molt, si t'equivoques perds poc.
- **Adaptabilitat**: el mercat canvia. Una estratègia fixa sense adaptació al règim fracassa.
- **Preservació del capital**: l'objectiu principal és sobreviure per poder operar demà.

### 8.3 Errors Comuns (Fins i Tot en Experts)

- Revenge trading: intentar recuperar pèrdues immediatament → cicle destructiu
- Sizing up after wins: l'eufòria porta a sobre-exposar-se just quan el mercat pot revertir
- Ignorar correlació: creure que tens 5 posicions diversificades quan en realitat tens la mateixa posició 5 vegades
- FOMO entries: entrar tard en un moviment gran → el risc és màxim i el R:R és mínim
- Moure el stop: la pitjor temptació. Canviar el stop per "donar-li marge" sol amplificar les pèrdues.

---

## 9. Síntesi per al Disseny de la Política RL

### 9.1 Preguntes Clau per Respondre Abans de Codificar

**Sobre l'Observation Space:**
- Quants timeframes incloc? (mínim recomanat: 2 — LTF per entrada, HTF per biaix)
- Incloc indicadors de règim (ADX, Hurst, vol ratio)?
- Incloc senyals on-chain (funding rate, OI, flux d'exchanges)?
- Incloc informació de la posició actual (preu d'entrada, P&L, temps obert)?

**Sobre l'Action Space:**
- Discret (HOLD / LONG / SHORT / CLOSE) o continu (mida de posició)?
- Permeto apalancament? Quant màxim?
- Permeto operació parcial (tancar 50%)?

**Sobre el Reward:**
- Recompenso profit brut o profit ajustat per risc (Sharpe, Sortino)?
- Penalitzo per drawdown? Per trades freqüents (comissions)?
- Recompenso per seguir el règim correcte (no operar en caos)?

**Sobre la Policy:**
- Implemento lògica de position sizing dins l'agent o és fix externament?
- Com encodo el regime detection? Com a input o com a subpolítica?
- Implemento curriculum learning (primer tendències clares, després rangs complexos)?

### 9.2 Arquitectura Recomanada (Alt Nivell)

```
Capes d'informació:
┌─────────────────────────────────────┐
│  HTF Context (4H/Daily)            │  → Biaix direccional, règim
│  - Estructura de mercat             │
│  - ADX, Hurst, vol ratio           │
├─────────────────────────────────────┤
│  LTF Signals (15m/1H)             │  → Trigger d'entrada/sortida
│  - OHLCV, indicadors               │
│  - Volume, order flow proxy        │
├─────────────────────────────────────┤
│  On-chain / Market Microstructure  │  → Context cripto
│  - Funding rate, OI                │
│  - Exchange flows (si disponible)  │
├─────────────────────────────────────┤
│  State de la Posició               │  → Gestió del trade obert
│  - Preu entrada, P&L actual        │
│  - Temps obert, distància al stop  │
└─────────────────────────────────────┘
          ↓
    Policy Network (PPO / SAC)
          ↓
    Action: [direcció, mida, stop_offset, target_offset]
```

### 9.3 Reward Shaping Inspirat en el Trader Professional

```python
# Pseudocodi conceptual de reward shaping

reward = 0

# 1. Profit realitzat (ajustat per comissions)
reward += realized_pnl - transaction_costs

# 2. Penalització per drawdown excessiu
if current_drawdown > MAX_DRAWDOWN_THRESHOLD:
    reward -= drawdown_penalty * current_drawdown

# 3. Premi per R:R correcte en l'entrada
if trade_opened:
    rr = (target - entry) / (entry - stop)
    if rr >= 2.0:
        reward += rr_bonus

# 4. Penalització per operar en règim caòtic
if regime == "chaotic" and position_opened:
    reward -= chaos_penalty

# 5. Premi per preservar capital (no operar = bo si no hi ha edge)
if no_position and regime == "ranging":
    reward += patience_bonus * small_constant

# 6. Penalització per moure el stop en la direcció dolenta
if stop_moved_against_trade:
    reward -= stop_manipulation_penalty
```

### 9.4 Errors de Disseny RL Equivalents als Errors del Trader

| Error del trader | Equivalent en RL |
|---|---|
| Revenge trading | Agent aprèn a sobre-operar després de pèrdues |
| Sizing up en eufòria | Mida de posició no regularitzada → vol màxim |
| Ignorar correlació | No incloure state de posicions obertes a l'observation |
| FOMO entries | Reward massa lligat a no perdre moviments |
| Moure el stop | Action space permet moure stop; reward no ho penalitza |
| Over-reliance en indicadors | Observation space només conté indicadors, no price action |

---

## 10. Lectures i Fonts Recomanades

**Llibres fonamentals:**
- *Market Wizards* (Jack Schwager) — entrevistes a traders top, principis psicològics
- *Trading in the Zone* (Mark Douglas) — mindset i probabilitats
- *Advances in Financial Machine Learning* (Marcos López de Prado) — implementació quantitativa
- *The Art and Science of Technical Analysis* (Adam Grimes) — fonaments de price action

**Recursos tècnics per a RL en trading:**
- [Denny Britz — Learning to Trade with RL](https://dennybritz.com/posts/wildml/learning-to-trade-with-reinforcement-learning/)
- [ML4Trading — Deep RL Chapter](https://www.ml4trading.io/chapter/21)
- [ClementPerroud — RL Trading Agent (GitHub)](https://github.com/ClementPerroud/RL-Trading-Agent)
- [NeuralArB — RL in Dynamic Crypto Markets](https://www.neuralarb.com/2025/11/20/reinforcement-learning-in-dynamic-crypto-markets/)

---

*Document preparat per a ús intern com a base de disseny de política RL. Versió 1.0*
