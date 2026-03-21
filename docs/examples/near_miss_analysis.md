# Exemple: Anàlisi d'un Near-Miss — P4 Bloqueja el Trade

> Exemple de com el Near-Miss Logger registra una oportunitat que ha passat P1+P2+P3
> però ha estat bloquejada per P4 (o P5). Útil per calibrar si P4 és massa restrictiu.
> Per a la documentació del Gate System: veure **[02_GATE_SYSTEM.md](../02_GATE_SYSTEM.md)**.

---

## Context: Dimecres 5 febrer 2025, 12:00 UTC

BTC/USDT cota a **97.820 USDT**. El mercat s'ha consolidat en un rang estret (96.500-98.500) durant 3 dies. L'estructura de mercat és interessant: hi ha un nivell de suport clar a 97.500 (HVN + Fib 38.2% d'un swing anterior).

---

## P1 — Règim: WEAK_BULL ✓

```
HMM estat: "Expansió moderada" (conf=0.68)
XGBoost: WEAK_BULL (prob=0.68 > 0.60)
→ P1 PASSA
```

---

## P2 — Salut: multiplier 0.90 ✓

```
Fear & Greed: 55 (Neutral → multiplier = 1.0)
Funding rate: +0.025% (lleument positiu → ajust = 0)
Multiplier P2: 1.0 → 0.90 (arrodonit per conservadorisme)
→ P2 PASSA
```

---

## P3 — Estructura: nivell accionable ✓

```
Nivell candidat: 97.500 (Fib 38.2% + HVN)
  Fib 38.2%: ✓ (distància 0.32%, dins tolerància)
  HVN: ✓ (volum alt al bin 97.300-97.620)
  Touches: 6 (zona molt ben testada)

Força nivell: 0.78 > 0.40 ✓
Stop Loss proposat: 96.200 (Fib 50%)
Distància stop: 97.500 - 96.200 = 1.300

→ P3 PASSA. Near-miss logger activa el registre.
```

---

## P4 — Momentum: **FALLA** ✗

```
Preu suavitzat (EWM span=3):
  t-4: 98.120 → t-3: 97.950 → t-2: 97.780 → t-1: 97.820 → t: 97.820

D1 (velocitat):    (97.820 - 97.820) / 97.820 = 0.0000  ← ZERO
D2 (acceleració): ΔD1 = 0.0000 - (-0.0005) = +0.0005

RSI-2 Connors:
  Últimes 2 candles: preu lateral
  RSI-2 = 38  ← NO sobrevenut (llindar = 10)

Volume confirmation:
  Volum candle actual: 0.743 BTC
  Volum SMA-20: 0.981 BTC
  Ratio: 0.76 < 1.0 ← volum per sota de la mitjana

Avaluació P4:
  D1 > 0: ✗ (D1 = 0, preu completament pla)
  RSI-2 < 10: ✗ (RSI-2 = 38, no sobrevenut)
  Volum > SMA: ✗

→ P4 FALLA. El preu no es mou cap amunt; no hi ha momentum confirmat.
```

**No hi ha trade.** El bot torna Signal(action=HOLD, reason="P4_failed_no_momentum").

---

## Registre al Near-Miss Logger

```python
# Automàticament registrat a la taula gate_near_misses:
NearMiss(
    timestamp       = "2025-02-05T12:00:00Z",
    symbol          = "BTC/USDT",
    regime          = "WEAK_BULL",
    regime_conf     = 0.68,
    p2_multiplier   = 0.90,
    p3_level        = 97500.0,
    p3_strength     = 0.78,
    p3_stop         = 96200.0,
    p4_passed       = False,           # ← motiu del near-miss
    p4_d1           = 0.0000,
    p4_rsi2         = 38.0,
    p4_volume_ratio = 0.76,
    p5_passed       = None,            # no s'ha evaluat
    current_price   = 97820.0,
)
```

---

## Resultat posterior: va ser encertada la decisió?

**12 hores més tard** (17:00 UTC), el preu cota a 97.350:
- Caiguda des de 97.820: -0.48%
- El trade hypothètic hauria quedat en pèrdues lleus (sense tocar el stop)

**36 hores més tard** (el dia 6/02, 00:00 UTC), el preu cota a 98.640:
- Recuperació des del suport 97.500 confirmada
- Si hagués esperat a que P4 passes, el trade s'hauria obert a 97.500-98.000
- Guany teòric: ~1.2% (menys que si hagués obert a 97.820, però amb millor confirmació)

**Conclusió:** P4 va ser **conservadora però correcta**. El moviment va venir, però no immediatament; esperar la confirmació de momentum hauria millorat el preu d'entrada.

---

## Anàlisi de near-misses acumulats

Amb múltiples setmanes de demo, la consulta SQL següent mostra el patró:

```sql
-- Distribució de near-misses per motiu i règim (darrers 30 dies)
SELECT
    regime,
    p4_passed,
    COUNT(*) as count,
    AVG(p4_d1) as avg_d1,
    AVG(p4_rsi2) as avg_rsi2,
    AVG(p4_volume_ratio) as avg_vol_ratio
FROM gate_near_misses
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY regime, p4_passed
ORDER BY regime, p4_passed;
```

**Exemple de resultat (hipotètic):**

| regime | p4_passed | count | avg_d1 | avg_rsi2 | avg_vol_ratio |
|--------|-----------|-------|--------|----------|---------------|
| WEAK_BULL | false | 23 | 0.0001 | 28.4 | 0.81 |
| WEAK_BULL | true | 8 | 0.0028 | 9.2 | 1.24 |
| RANGING | false | 31 | -0.0002 | 41.2 | 0.72 |
| RANGING | true | 3 | 0.0031 | 7.8 | 1.31 |

**Interpretació:** En WEAK_BULL, 23 near-misses van ser bloquejats per P4 (D1 ~ 0, RSI-2 ~ 28, vol ~ 0.81). Dels 8 que van passar P4, el D1 era 10× més gran i el RSI-2 estava a 9 (sobrevenut extrem). P4 discrimina bé: els que passen tenen condicions molt més fortes.

---

## Quan recalibrar P4?

**Considerar relaxar P4 si:**
- >50% dels near-misses bloquejats per P4 acaben pujant >2% en les 24h posteriors
- El `avg_rsi2` dels bloquejos és sistemàticament alt (>30) → RSI-2 mai arriba a sobrevenut
- El `avg_vol_ratio` dels bloquejos és >1.0 → el volum SÍ confirma però P4 falla per D1

**Considerar endurir P4 si:**
- Els trades que passen P4 perden sistemàticament en les primeres 8h
- El `avg_d1` dels que passen és molt baix (el filtre D1 no discrimina prou)

**Paràmetres de calibratge (gate.yaml):**
```yaml
p4:
  ewm_span: 3           # augmentar → menys soroll, D1 més suau
  rsi2_oversold: 10     # abaixar a 5 → P4 més restrictiu; pujar a 15 → més permissiu
```

---

*Per a l'exemple d'un trade que passa totes les portes: veure **[trade_walkthrough.md](./trade_walkthrough.md)**.*
