# Roadmap — Camí a la Demo 24/7

> **Regla d'or:** cap bot va a demo fins que en backtest out-of-sample superi HoldBot en Sharpe i Calmar.

---

## Estat del sistema

| Component | Estat | Notes |
|-----------|-------|-------|
| Classical bots (6) | ✅ | Trend, DCA, Grid, Hold, MeanReversion, Momentum |
| ML models (7) | ✅ | RF, XGB, LGBM, CB, GRU, PatchTST, TFT |
| RL agents — baseline | ✅ | PPO, SAC (1H, 500k steps) |
| RL agents — on-chain | ✅ | PPO + SAC amb Fear&Greed, funding rate, hash-rate |
| RL agents — professional | ✅ | PPO + SAC (12H, ATR stop, position state, reward professional) |
| RL agents — TD3 | ✅ | td3_professional (12H) + td3_multiframe (12H+4H) |
| Dades externes | ✅ | Fear&Greed, funding rates, open interest, blockchain |
| Optimize workflow | ✅ | Optuna + best_params in-place al YAML base |
| BacktestEngine + MLflow | ✅ | Walk-forward, mètriques Sharpe/Calmar |
| DemoRunner | ✅ | Persistència + Telegram |
| Documentació | ✅ | PROJECT, MODELS, EXTENDING, CONFIG, DATABASE, DECISIONS |
| **EnsembleBot v1** | ✅ | majority_vote — `bots/classical/ensemble_bot.py` |

---

## Camí crític cap a la demo

```
[✅]   EnsembleBot v1 implementat (majority vote)
[ara]  Optimització Optuna de tots els models (en curs)
          ↓
[A]    Entrenar tots els models amb hiperparàmetres òptims
          ↓
[B]    Backtest EnsembleBot — ha de superar HoldBot
          ↓
[C]    Correccions DemoRunner (sync candles + persistir estat)
          ↓
[D]    Migrar a servidor 24/7
          ↓
[E]    🚀 Iniciar demo — 3-6 mesos de paper trading
```

A partir de [F], les tasques de Telegram i Dashboard es fan **en paral·lel** mentre la demo corre. No bloquegen l'inici.

---

## Tasques pendents

---

### A — Entrenar tots els models (pendent entrenament òptim)

Abans d'entrenar cal acabar l'optimització Optuna.

**Recomanació de steps per a entrenament RL:**

| Agent | Steps recomanats | Justificació |
|-------|-----------------|-------------|
| PPO / SAC baseline (1H) | **1M–1.5M** | ~35k candles training → 500k = 14 passes; 1M = 28 passes, punt òptim |
| PPO / SAC on-chain (1H) | **1M–1.5M** | Igual que baseline; les features addicionals necessiten més training |
| PPO / SAC professional (12H) | **500k** | ~1.2k candles → ja 420 passes amb 500k; augmentar seria overfitting |
| TD3 professional / multiframe (12H) | **500k** | Igual que professional |

```bash
# Entrenar tot d'una (un cop l'Optuna hagi acabat)
python scripts/train_models.py          # tots els ML
python scripts/train_rl.py              # tots els RL (modifica total_timesteps als YAMLs primer)
```

---

### B — Backtest EnsembleBot (validació pre-demo)

EnsembleBot v1 ja implementat (`bots/classical/ensemble_bot.py`, política `majority_vote`).

Abans d'activar-lo a la demo, cal confirmar que supera HoldBot en backtest out-of-sample:

```bash
python scripts/run_comparison.py --bots hold trend mean_reversion momentum \
    ml_xgb ml_lgbm ml_rf rl_ppo rl_sac ensemble
```

Criteri mínim d'entrada a la demo: **Sharpe > 1.0 i Drawdown màx. < -25%** en backtest out-of-sample.

**Quins bots entren a l'ensemble:** edita `config/models/ensemble.yaml` (secció `sub_bots`). Descomenta els que hagin superat el criteri. Reinicia el DemoRunner per aplicar el canvi.

**Polítiques futures (mentre la demo corre):**

| Política | Quan implementar |
|---------|-----------------|
| `weighted` | Pes proporcional al Sharpe dels darrers N dies de cada sub-bot |
| `stacking` | Model ML de 2a capa entrenat sobre les prediccions dels sub-bots |

---

### C — Correccions DemoRunner

Dues correccions necessàries per a un demo fiable:

**C1. Sincronització de candles** — el bot ha d'actuar quan TANCA una candle (1 cop/hora), no cada 60 s sobre la mateixa candle oberta. Evita que el bot operi múltiples vegades sobre el mateix tick.

**C2. Persistir estat intern** — `_in_position`, `_tick_count` i l'estat del portfolio han de restaurar-se des de la BD si el procés es reinicia. Ara es perden si el DemoRunner s'atura.

---

### D — Migrar a servidor 24/7

Per a la demo necessites un servidor que estigui sempre actiu. Opcions:

| Opció | Cost | RAM/CPU | Ideal per a |
|-------|------|---------|------------|
| **Oracle Cloud Free Tier** | **Gratuït per sempre** | 4 vCPU ARM · 24 GB RAM · 200 GB SSD | ✅ **Recomanat** — la millor relació preu/prestacions |
| Hetzner CX32 | ~10€/mes | 4 vCPU · 8 GB RAM · 80 GB SSD | Si prefereixes pagar per fiabilitat europea |
| Raspberry Pi 5 | ~100€ únic | 4 vCPU ARM · 8 GB RAM | Si tens internet fiable a casa |

**Oracle Cloud** és la recomanació clara: ARM Ampere A1 (mateixa arquitectura que Apple Silicon — tots els paquets Python/ML funcionen perfectament), 24 GB de RAM (sobrat per executar PostgreSQL + DemoRunner + inference), i és **gratuït per sempre**.

**Infraestructura mínima al servidor:**
```bash
# Cron jobs
0 * * * *   python scripts/update_data.py        # candles cada hora
0 8 * * *   python scripts/update_fear_greed.py  # F&G diari
0 */4 * * * python scripts/update_futures.py     # funding rate
0 6 * * *   python scripts/update_blockchain.py  # on-chain diari

# Servei systemd (restart automàtic)
/etc/systemd/system/btc-demo.service → python scripts/run_demo.py

# Backup
0 3 * * *   pg_dump btc_trading > /backups/btc_$(date +%Y%m%d).sql
```

---

### E — Inici de la demo

Un cop [A-D] completats:

```bash
python scripts/run_demo.py
```

**Objectiu:** mínim **3-6 mesos** de paper trading en temps real, tots els bots que hagin superat el criteri de backtest actius simultàniament, registre complet a la BD.

---

## En paral·lel mentre la demo corre

### G — Telegram millorat

- Resum diari: PnL per bot, comparativa vs BTC spot
- Alertes de drawdown configurables per bot (ex: alerta si drawdown > 10%)
- Rànquing setmanal
- Comandes interactives: `/status`, `/portfolio`, `/ranking`

### H — Dashboard Streamlit complet

Quan hi hagi mesos de dades reals:
- Portfolios en temps real per bot (gràfic PnL acumulat)
- Trade log filtrable per bot i data
- Comparativa vs BTC spot
- Ranking per Sharpe/Calmar dels darrers 30/90/180 dies

### I — Nous models (opcional, baix impacte en demo)

| Model | Tipus | Notes |
|-------|-------|-------|
| BreakoutBot | Clàssic | Pivot points + ATR per confirmar ruptures |
| N-BEATS / N-HiTS | Deep Learning | Arquitectures eficients sense RNN |
| TabNet | Tabular | Competitiu amb XGBoost, interpretatiu |
| EnsembleBot weighted | Meta | Pes proporcional al Sharpe de cada bot |

### J — Neteja de BD per a la demo

La BD de dades de mercat (OHLCV, fear_greed, etc.) s'ha de conservar íntegra.
El que cal netejar és l'estat de demo anterior (si n'hi ha):

```sql
-- Executar si vols iniciar la demo des de zero
TRUNCATE TABLE demo_portfolio;  -- estat del portfolio anterior
TRUNCATE TABLE demo_trades;     -- historial de trades anteriors
-- Les taules de candles i dades externes no s'han de tocar
```

*(Verifica els noms exactes de taules amb `\dt` a psql)*

---

## Resposta a la pregunta clau

> *"El desenvolupament de Telegram i Dashboard el puc fer mentre la demo ja vagi corrent?"*

**Sí, completament.** El DemoRunner és independent del Telegram i del Dashboard. Pots iniciar la demo amb el Telegram bàsic que ja tens i millorar-lo mentre les dades s'acumulen. El Dashboard és fins i tot millor fer-lo un cop tens mesos de dades reals — no té sentit construir gràfics bonics sense dades.

**Ordre de prioritats real:**
1. Entrenar models (ara)
2. EnsembleBot v1
3. Correcció DemoRunner
4. Servidor + demo
5. Tot the rest mentre la demo corre

---

*Última actualització: Març 2026 · Versió 3.0*
