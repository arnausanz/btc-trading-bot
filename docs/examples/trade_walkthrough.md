# Exemple: Un Trade del Gate System — Pas a Pas

> Il·lustració de com el Gate System avalua i executa un trade complet.
> Per a la documentació completa del Gate System: veure **[02_GATE_SYSTEM.md](../02_GATE_SYSTEM.md)**.

---

## Context: Dijous 16 gener 2025, 08:00 UTC

BTC/USDT cota a **95.420 USDT**. El mercat ha corregit un 8% en els darrers 4 dies des del màxim de 103.800 i ha format una base a la zona 94.000-96.000. La candle de les 8:00 UTC acaba de tancar a 95.420.

El DemoRunner crida `GateBot.on_observation(obs)` amb les candles de les darreres 300× 4H i 300×1D.

---

## P1 — Règim: "Com està el mercat?"

**Dades d'entrada:**
- EMA-200 diària: 72.400 → preu (95.420) molt per sobre ✓
- EMA-50 diària: 89.600 → preu per sobre ✓
- ADX-14 diari: 28 (tendència moderada)
- Volatilitat anualitzada (20 dies): 62%

**Pipeline P1:**

```
1. HMM Viterbi sobre retorns log[90 dies]:
   → Seqüència d'estats: [2, 2, 1, 2, 2, 2, 1, 2]  ← últim estat = 2
   → Estat 2 = "Expansió alcista" (identificat durant l'entrenament)

2. XGBoost de confirmació:
   Features: [close=95420, ema50=89600, ema200=72400, adx=28, atr_pct=1.8%,
              volatility_20d=62%, rsi14_1d=58, fear_greed=62, funding_rate=0.01%]
   → Probabilitat WEAK_BULL: 0.71
   → Probabilitat STRONG_BULL: 0.18
   → Probabilitat RANGING: 0.08
   → Probabilitat BEAR: 0.03
   → Classe final: WEAK_BULL (conf=0.71 > 0.60) ✓

3. Resultat P1: WEAK_BULL, confiança 0.71
```

**P1 PASSA.** El règim és apte per a operacions long.

---

## P2 — Salut del Mercat: "Està sa el mercat?"

**Dades d'entrada:**
- Fear & Greed Index: 62 (Greed, però no extrem)
- Funding rate perpetuals: +0.01% (neutre-positiu)

**Avaluació P2:**

```
Fear & Greed Index = 62 → zona GREED
  Rang    [0,25]:  Extreme Fear   → multiplier = 1.2  (oportunitat)
  Rang   [25,40]:  Fear           → multiplier = 1.1
  Rang   [40,60]:  Neutral        → multiplier = 1.0
  Rang   [60,75]:  Greed          → multiplier = 0.85 ← aplica
  Rang   [75,90]:  Extreme Greed  → multiplier = 0.7
  Rang  [90,100]:  Extreme Greed  → multiplier = 0.0 (VETO)

Funding rate = +0.01%
  [-0.05%, +0.03%]: neutre → ajust = 0
  [+0.03%, +0.10%]: lleument positiu → ajust = -0.05
  [+0.10%, +0.20%]: molt positiu → ajust = -0.15 (VETO si >+0.20%)

Multiplier final P2 = 0.85 + 0 = 0.85
```

**P2 PASSA.** Multiplier 0.85 (no és veto, però reduirà lleugerament la mida de la posició).

---

## P3 — Estructura de Mercat: "Hi ha un bon preu?"

**Dades d'entrada:** 300 candles 4H + 300 candles 1D. Preu actual: 95.420.

**Pipeline P3:**

**Step 1: Fibonacci sobre el swing principal**
```
Swing High (màxim local): 103.800 (identificat 4 dies enrere, pivot fractal validat)
Swing Low (mínim local):   87.500 (identificat 11 dies enrere, pivot fractal validat)
Rang swing: 103.800 - 87.500 = 16.300

Nivells Fibonacci:
  0.382: 103.800 - 16.300 × 0.382 = 97.573
  0.500: 103.800 - 16.300 × 0.500 = 95.650  ← preu actual 95.420 ≈ aquí!
  0.618: 103.800 - 16.300 × 0.618 = 93.726
  0.786: 103.800 - 16.300 × 0.786 = 91.987
```

**Step 2: Volume Profile (50 bins)**
```
Rang de preu: [87.500, 103.800]
Resolució per bin: 326 USDT/bin

Bin més transaccionat (High Volume Node):
  94.800 - 95.126: volum = 18.420 BTC  ← HVN principal
  95.126 - 95.452: volum = 16.380 BTC  ← HVN secundari (preu actual aquí!)

El preu actual cau dins d'una zona HVN (volum alt = suport fort)
```

**Step 3: Confluència i força del nivell**
```
Nivell candidat: 95.500 (zona Fib 50% + HVN)
  Fib 50%:  ✓ (distància 95.650 → 95.500: 0.16%, dins tolerància ATR×0.5)
  HVN:      ✓ (HVN secundari a 95.126-95.452, bin adjacent)
  Touches:  4 (el preu ha tocat aquesta zona 4 vegades en les últimes 300 candles)

Força del nivell = 0.5 (Fib 50%) × 0.7 (4 touches) × 0.9 (HVN) = 0.63
Llindar mínim: min_level_strength = 0.40 ✓
```

**P3 PASSA.** Nivell accionable: 95.500 (Fib 50% + HVN), força 0.63.
Stop Loss proposat: 93.726 (Fib 61.8%). Distància: 95.500 - 93.726 = 1.774.

---

## P4 — Momentum: "Hi ha momentum ara?"

**Dades d'entrada:** últimes 20 candles 4H, preu actual 95.420.

**Pipeline P4:**

```
Preu suavitzat (EWM span=3):
  t-4: 94.120 → t-3: 94.580 → t-2: 94.920 → t-1: 95.150 → t: 95.420

D1 (velocitat):    (95.420 - 95.150) / 95.150 = +0.0028  (+0.28%) ✓ POSITIVA
D2 (acceleració): ΔD1 = (0.0028 - 0.0023) = +0.0005  ✓ POSITIVA

RSI-2 Connors:
  Últimes 2 candles: retorn acumulat = +0.32%... període curt
  RSI-2 = 8  ← sobrevenut extrem (< 10 = senyal fort de reversió)

Volume confirmation:
  Volum candle actual: 1.248 BTC
  Volum SMA-20:        0.987 BTC
  Ratio: 1.26 > 1.0 ✓ (volum per sobre de la mitjana)
```

**Avaluació P4:**
- D1 > 0: velocitat positiva ✓
- D2 > 0: acceleració positiva ✓ (el moviment s'accelera)
- RSI-2 = 8 < 10: sobrevenut extrem (compra contrarian validada) ✓
- Volum > SMA: confirmació de volum ✓

**P4 PASSA.** Momentum confirmat: preu pujant amb acceleració, RSI-2 sobrevenut, volum per sobre.

---

## P5 — Gestió del Risc: "Quant arriscar?"

**Dades d'entrada (de P3):**
- Nivell d'entrada: 95.500
- Stop Loss: 93.726 (Fib 61.8%)
- Distància al stop: 1.774 (1.86%)

**Pipeline P5:**

**Step 1: Target i R:R**
```
Règim: WEAK_BULL → min_rr = 2.0
ATR-14 (4H): 920

Target (trailing trailing stop passarà per aquí):
  Mètode: entrada + 2.0 × distància_stop = 95.500 + 2.0 × 1.774 = 99.048

R:R = (99.048 - 95.500) / (95.500 - 93.726) = 3.548 / 1.774 = 2.0 ✓
```

**Step 2: Càlcul Kelly**
```
Win rate estimat (del backtest del Gate System):    55%
Odds (R:R):                                         2.0

Kelly complet:
  f* = (p × b - q) / b = (0.55 × 2.0 - 0.45) / 2.0 = 0.325

Kelly fraccionari (factor seguretat 1/3):
  f_frac = 0.325 / 3 = 0.108 = 10.8% del capital en risc

Verificació amb max_risk_pct = 0.01 (1% del capital per trade):
  Capital: 10.000 USDT
  Risc màxim: 10.000 × 0.01 = 100 USDT
  Distància stop: 95.500 × 0.0186 = 1.776 USDT/BTC
  Mida màxima per risc: 100 / 1.776 = 56.3 USDT → 0.590 BTC equivalent capital

  Però Kelly suggereix 10.8% del capital = 1.080 USDT en risc
  → El límit de risk_pct (100 USDT < 1.080 USDT) és el factor limitant

  Mida final per Kelly (cap al risk limit):
    size = risk_cap / distància_stop_pct = 0.01 / 0.0186 = 0.538 (53.8% del capital)
```

**Step 3: Aplicar P2 multiplier**
```
Mida base (Kelly): 0.538
P2 multiplier: 0.85

Mida final: 0.538 × 0.85 = 0.457 → arrodonit a 0.46 (46% del capital)
```

**Step 4: Verificar límits de risc**
```
max_exposure_pct = 0.95 → exposició total no pot superar 95%
Posicions actuals: 0  → exposició actual: 0%
Exposició si s'obre: 46% < 95% ✓

max_open_positions = 2 → posicions actuals: 0 < 2 ✓

weekly_drawdown_limit = 0.05 → pèrdua setmanal actual: 0% < 5% ✓
```

**P5 PASSA.** Mida: 0.46 (46% del capital, ~4.600 USDT). Cap VETO actiu.

---

## Resultat final: BUY

```python
Signal(
    bot_id     = "gate_v1",
    action     = Action.BUY,
    size       = 0.46,
    confidence = 0.71,      # confiança de P1
    reason     = "WEAK_BULL|P2=0.85|Fib50%+HVN@95500|D1+D2+RSI2=8|Kelly=0.46|RR=2.0"
)
```

**PaperExchange executa:**
```
ORDER BUY: 0.46 × 10.000 USDT = 4.600 USDT
Price: 95.420 USDT
Fee (0.1%): 4.60 USDT
Posició oberta: 0.04820 BTC @ 95.420 USDT
Stop Loss: 93.726 USDT (pèrdua màxima: ~90 USDT = 0.9% del capital)
Trailing stop actiu: quan preu > 96.340 (95.420 + 1×ATR 920)
```

**Notificació Telegram:**
```
📈 GATE BUY
0.04820 BTC @ 95.420 USDT
Règim: WEAK_BULL (0.71)
Nivell: Fib50%+HVN 95.500
R:R: 2.0 | Mida: 46%
Stop: 93.726 | Target: ~99.048
```

---

## Seguiment de la posició — dies posteriors

### Divendres 17 gener, 08:00 UTC — Preu: 97.180

```
GateBot._manage_open_positions() cridat:
  Posició: LONG 0.04820 BTC @ 95.420
  PnL: (97.180 - 95.420) / 95.420 = +1.85%
  Trailing stop: activat (preu > 96.340)
    ATR percentil actual: 45 (vol normal)
    trailing_atr_multiplier[normal_vol] = 2.0
    Nou stop level: 97.180 - 2.0 × 920 = 95.340
    Stop puja de 93.726 → 95.340 ✓
  Circuit breaker: |candle| = 0.9% < 3×ATR% = 2.9% → no activa
  D2 check: +0.0002 → no desacceleració
  → HOLD: continuar amb trailing stop a 95.340
```

### Dilluns 20 gener, 16:00 UTC — Preu: 99.560

```
GateBot._manage_open_positions():
  Posició: LONG 0.04820 BTC @ 95.420
  PnL: +4.35%
  Trailing stop: 99.560 - 2.0 × 920 = 97.720 (puja de 95.340 → 97.720) ✓
  D2 check: -0.0003 (desacceleració!)
    Règim WEAK_BULL → decel_exit_candles = 3
    Comptador desacceleració: 1/3 → no activa sortida encara
  → HOLD
```

### Dimarts 21 gener, 00:00 UTC — Preu: 98.340 (correcció)

```
GateBot._manage_open_positions():
  Posició: LONG 0.04820 BTC @ 95.420
  PnL: +3.06%
  Trailing stop actual: 97.720
  Preu (98.340) > Stop (97.720) → posició oberta ✓
  D2 check: -0.0008 (desacceleració continua)
    Comptador: 2/3
  → HOLD
```

### Dimarts 21 gener, 04:00 UTC — Preu: 97.650 → **Stop tocat!**

```
GateBot._manage_open_positions():
  Posició: LONG 0.04820 BTC @ 95.420
  Preu (97.650) ≤ Stop (97.720) → STOP LOSS TOCAT

  Signal: SELL (size=1.0, "trailing_stop_hit@97.720")
  PaperExchange: SELL 0.04820 BTC @ 97.650
  PnL: (97.650 - 95.420) / 95.420 = +2.34%
  Benefici net: 4.600 × 0.0234 - fees = 107.64 - 9.20 = +98.44 USDT (+0.98%)
```

**Notificació Telegram:**
```
📉 GATE SELL (trailing stop)
0.04820 BTC @ 97.650 USDT
PnL: +2.34% (+98.44 USDT)
Durada: 4.5 dies
Stop final: 97.720 (trailing ATR×2.0)
```

---

## Resum del trade

| Concepte | Valor |
|---------|-------|
| Entrada | 95.420 USDT |
| Sortida | 97.650 USDT |
| Mida | 46% del capital (4.600 USDT) |
| Durada | 4.5 dies |
| PnL brut | +2.34% |
| Fees | -0.20% |
| **PnL net** | **+2.14%** |
| **Guany absolut** | **+98.44 USDT** |
| R:R aconseguit | ~1.25 (vs target 2.0 — sortida per trailing stop) |

**Observació:** El target teòric era 99.048 (R:R 2.0). El trailing stop a 97.720 va tancar el trade amb R:R 1.25. És el comportament esperat: el trailing stop protegeix les guanyes i evita que un bon trade es converteixi en pèrdua, però pot tancar la posició abans d'arribar al target si hi ha una correcció.

---

*Per a l'exemple d'un trade rebutjat per P4 (near-miss), veure **[near_miss_analysis.md](./near_miss_analysis.md)**.*
