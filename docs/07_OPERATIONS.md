# Runbook d'Operació

> Document de referència per engegar, monitoritzar i mantenir el sistema.
> Per a l'arquitectura general: veure **[01_ARCHITECTURE.md](./01_ARCHITECTURE.md)**.
> Per als detalls de configuració: veure **[04_CONFIGURATION.md](./04_CONFIGURATION.md)**.

---

## Guia ràpida d'inici (Quick Start)

```bash
# 1. Setup de l'entorn
cp .env.example .env          # editar DATABASE_URL, TELEGRAM_TOKEN, etc.
pip install -r requirements.txt

# 2. Base de dades
docker-compose up -d          # PostgreSQL via Docker (recomanat)
# o: psql -c "CREATE DATABASE btc_trading;"  # si ja tens PostgreSQL instal·lat

# 3. Esquema de BD
alembic upgrade head

# 4. Descarregar dades
python scripts/download_data.py          # candles OHLCV (~2019-avui)
python scripts/download_fear_greed.py    # Fear & Greed Index
python scripts/update_futures.py         # Funding rate

# 5. Validar que tot és correcte
python scripts/validate_data.py

# 6. Engegar paper trading (bots base sense entrenar)
python scripts/run_demo.py
```

---

## Cicle de vida complet — de zero a demo

### Fase 0 — Dades (una sola vegada)

```bash
python scripts/download_data.py
python scripts/download_fear_greed.py
python scripts/update_futures.py
python scripts/validate_data.py
```

**Temps estimat:** ~30-60 min per la descàrrega inicial (dades 2019-avui).

---

### Fase 1 — Optimitzar hiperparàmetres (Optuna)

```bash
# Bots clàssics (DCA, TrendBot, GridBot, etc.)
python scripts/optimize_bots.py --bots dca trend grid mean_reversion momentum

# Models ML (arbres + neuronals)
python scripts/optimize_models.py --no-rl --trials 30

# Models RL (seqüencial — no en paral·lel)
python scripts/optimize_models.py --model ppo_professional
python scripts/optimize_models.py --model sac_professional
python scripts/optimize_models.py --model td3_professional
python scripts/optimize_models.py --model td3_multiframe
```

**Temps estimat total (M4 Pro 24GB):** ~30-50 hores.

---

### Fase 2 — Entrenar models

```bash
# Models ML
python scripts/train_models.py

# Models RL
python scripts/train_rl.py

# Gate System (pipeline propi — diferent de ML/RL)
alembic upgrade head                  # crea taules gate_positions + gate_near_misses
python scripts/train_gate_regime.py   # HMM K=2..6 + XGBoost Optuna (~30-45 min)

# Verificar que els models s'han creat
ls models/
# Ha d'incloure: *.pkl, *.pt, *.zip, gate_hmm.pkl, gate_xgb_regime.pkl
```

---

### Fase 3 — Validar (walk-forward backtest)

```bash
# Backtest de tots els bots comparables (clàssics + ML + RL; exclou comparable:false)
python scripts/run_comparison.py

# Sense RL (més ràpid si els models RL no estan entrenats)
python scripts/run_comparison.py --no-rl

# Selecció manual
python scripts/run_comparison.py --bots hold trend xgboost ppo_professional
```

**Criteri mínim per activar un bot a la demo:**
- Sharpe Ratio > 1.0 en el període de TEST (2025-01-01 en avant)
- Max Drawdown > -25% en el període de TEST
- Ha de superar HoldBot (buy & hold) en Sharpe i Calmar

---

### Fase 4 — Activar a demo

```bash
# Editar config/demo.yaml: canviar enabled: false → enabled: true
# per a cada bot que hagi superat el criteri de backtest
nano config/demo.yaml

# Iniciar paper trading 24/7
python scripts/run_demo.py
```

---

## Health checks — verificar que el sistema funciona

### Verificació ràpida

```bash
# 1. La BD té dades recents?
psql $DATABASE_URL -c "
  SELECT MAX(timestamp), COUNT(*) FROM candles
  WHERE symbol='BTC/USDT' AND timeframe='1h';
"
# Esperat: MAX(timestamp) = avui o ahir, COUNT(*) > 50000

# 2. El DemoRunner corre?
ps aux | grep run_demo.py
# Si no apareix → reiniciar

# 3. Hi ha activitat recent?
psql $DATABASE_URL -c "
  SELECT bot_id, action, timestamp
  FROM demo_signals
  ORDER BY timestamp DESC
  LIMIT 10;
"
```

### Verificació completa

```bash
# Validar integritat de les dades de mercat
python scripts/validate_data.py

# Test de smoke (66 tests ràpids sense deps exteriors)
cd tests && python -m pytest smoke/ -v

# Test unitaris (57 tests amb mocks)
python -m pytest unit/ -v
```

---

## Interpretació dels logs

### Log normal del DemoRunner

```
[2026-03-21 10:00:01] DemoRunner: tick BTC/USDT 65420.50
[2026-03-21 10:00:01] TrendBot: HOLD (ema_fast < ema_slow)
[2026-03-21 10:00:01] MLBot(xgboost): HOLD (confidence=0.52 < 0.65)
[2026-03-21 10:00:01] GateBot: HOLD (P1=RANGING → not actionable)
```

**Interpretació:** Comportament normal. Cada bot avalua i emet el seu senyal. HOLD és el més comú.

---

### Log d'un trade del GateBot

```
[2026-03-21 10:00:01] GateBot: P1=WEAK_BULL (conf=0.82) ✓
[2026-03-21 10:00:01] GateBot: P2 multiplier=0.85 ✓
[2026-03-21 10:00:01] GateBot: P3 level=64200 (strength=0.71) ✓
[2026-03-21 10:00:01] GateBot: P4 d1=+0.003 rsi2=8 ✓
[2026-03-21 10:00:01] GateBot: BUY size=0.34 (Kelly: risk=0.01, rr=2.1)
[2026-03-21 10:00:01] PaperExchange: ORDER BUY 0.34 BTC @ 65420.50
[2026-03-21 10:00:01] TelegramNotifier: 📈 GATE BUY 0.34 BTC @ 65420.50
```

**Interpretació:** Les 4 primeres portes han passat. P5 ha calculat la mida i obert posició.

---

### Alertes que requereixen atenció

| Log | Causa probable | Acció |
|-----|---------------|-------|
| `ConnectionError: PostgreSQL` | BD inactiva | `docker-compose up -d` o reiniciar PostgreSQL |
| `Missing data for timeframe 4h` | Candles buides | `python scripts/download_data.py` |
| `Model file not found: models/gate_hmm.pkl` | Gate no entrenat | `python scripts/train_gate_regime.py` |
| `TelegramNotifier: HTTPError` | Token invàlid | Verificar `TELEGRAM_TOKEN` al `.env` |
| `ValueError: Unexpected observation shape` | Model RL i YAML desincronitzats | Re-entrenar el model amb el YAML actual |
| `OperationalError: too many connections` | Connexions BD exhaurides | Reiniciar DemoRunner; revisar `pool_size` a `session.py` |

---

## Re-entrenament — quan i com

### Calendari recomanat

| Component | Freqüència | Mecanisme | Motiu |
|-----------|-----------|-----------|-------|
| **Candles OHLCV** | Cada hora | **DemoRunner intern** (`_update_data()`) | Integrat al runner — no cal cron |
| **Fear & Greed** | Diàriament | Cron | Actualitzar sentiment (1 valor/dia) |
| **Funding rate** | Cada 4h | Cron | Actualitzar dades de futures |
| **Blockchain / on-chain** | Diàriament | Cron | Hash-rate, exchange flows, etc. |
| **Gate System P1 (HMM+XGBoost)** | Mensualment | Manual | El règim de mercat canvia gradualment |
| **Models ML (XGBoost, GRU, etc.)** | Cada 1-3 mesos | Manual | Quan arriben dades noves suficients |
| **Models RL (PPO, SAC, TD3)** | Cada 3-6 mesos | Manual | Molt costós; prioritat baixa |

### Cron jobs per a la demo 24/7

> **Les candles OHLCV no necessiten cron.** El `DemoRunner` les actualitza internament cada hora via `_update_data()` (crida `OHLCVFetcher`, `FuturesFetcher` i `FearGreedFetcher` de cop). Afegir un cron independent crearia una condició de carrera (cron escrivint mentre el runner llegeix).

```bash
# Afegir a crontab: crontab -e
# Les dades externes de baixa freqüència SÍ necessiten cron:
0 8 * * *   cd /path/to/btc-trading-bot && python scripts/download_fear_greed.py
0 */4 * * * cd /path/to/btc-trading-bot && python scripts/update_futures.py
0 6 * * *   cd /path/to/btc-trading-bot && python scripts/update_blockchain.py
```

### Systemd per mantenir el DemoRunner actiu

```ini
# /etc/systemd/system/btc-demo.service
[Unit]
Description=BTC Trading Bot Demo Runner
After=postgresql.service

[Service]
Type=simple
WorkingDirectory=/path/to/btc-trading-bot
ExecStart=/usr/bin/python scripts/run_demo.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable btc-demo
systemctl start btc-demo
systemctl status btc-demo
```

---

## Checklist de diagnòstic si el bot es para

Seguir en ordre:

```
[ ] 1. Verificar que PostgreSQL corre
        systemctl status postgresql
        # o: docker-compose ps

[ ] 2. Verificar connexió a la BD
        python -c "from core.db.session import get_session; print('OK')"

[ ] 3. Verificar que hi ha dades recents
        python scripts/validate_data.py

[ ] 4. Verificar que els models existeixin
        ls models/ | grep -E "pkl|pt|zip"

[ ] 5. Executar smoke tests
        python -m pytest tests/smoke/ -v

[ ] 6. Iniciar el DemoRunner en mode verbose per veure l'error exacte
        python scripts/run_demo.py 2>&1 | head -50

[ ] 7. Revisar els logs de la BD (si hi ha errors de connexió)
        tail -n 50 /var/log/postgresql/postgresql-*.log
```

---

## Monitorització — mètriques clau

### Consultes SQL útils per a monitorització

```sql
-- Activitat d'avui per bot
SELECT bot_id, action, COUNT(*) as senyals
FROM demo_signals
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY bot_id, action
ORDER BY bot_id, action;

-- Trades de la darrera setmana
SELECT bot_id, action, size, price, pnl_pct, timestamp
FROM demo_trades
WHERE timestamp > NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;

-- PnL acumulat per bot
SELECT bot_id, SUM(pnl_pct) as total_pnl,
       COUNT(*) as n_trades,
       AVG(pnl_pct) as avg_pnl
FROM demo_trades
GROUP BY bot_id
ORDER BY total_pnl DESC;

-- Near-misses del Gate System (darrers 7 dies)
SELECT regime, p4_passed, COUNT(*) as count
FROM gate_near_misses
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY regime, p4_passed
ORDER BY count DESC;
```

### Mètriques de backtest

Els resultats de cada backtest es mostren per consola i es guarden al log estàndard (`logs/`). Les mètriques principals: `sharpe_ratio`, `max_drawdown_pct`, `win_rate_pct`, `total_return_pct`, `calmar_ratio`.

---

## Backups — què guardar

### Dades críticament importants

```bash
# 1. BD completa (candles + features + trades)
pg_dump btc_trading > backup_$(date +%Y%m%d).sql

# 2. Models entrenats
tar czf models_backup_$(date +%Y%m%d).tar.gz models/

# 3. Configuració
tar czf config_backup_$(date +%Y%m%d).tar.gz config/
```

### Backup automàtic diari (afegir al crontab)

```bash
0 3 * * * pg_dump btc_trading | gzip > /backups/btc_$(date +\%Y\%m\%d).sql.gz
0 3 * * * tar czf /backups/models_$(date +\%Y\%m\%d).tar.gz /path/to/models/
```

### Restaurar la BD

```bash
psql btc_trading < backup_YYYYMMDD.sql
# o des de gzip:
gunzip -c backup_YYYYMMDD.sql.gz | psql btc_trading
```

---

## Iniciar la demo des de zero (neteja de dades de demo)

Quan vols reiniciar les estadístiques de paper trading sense perdre les dades de mercat.

**Opció 1 — Via Telegram (recomanat mentre la demo corre):**
```
/reset all        → demana confirmació, reinicia tots els bots
/reset ppo_professional  → reinicia un bot concret
```

**Opció 2 — Via CLI:**
```bash
python scripts/reset_demo.py              # tots els bots (demana confirmació)
python scripts/reset_demo.py ppo_professional  # un bot concret
python scripts/reset_demo.py --yes        # saltar confirmació (scripts)
```

**Opció 3 — Via SQL directe:**
```sql
-- IMPORTANT: no tocar les taules candles ni les de dades externes!
TRUNCATE TABLE demo_signals;
TRUNCATE TABLE demo_trades;
TRUNCATE TABLE demo_ticks;
-- Les gate_positions i gate_near_misses es poden netejar si convé:
TRUNCATE TABLE gate_positions;
TRUNCATE TABLE gate_near_misses;
```

Després reiniciar el DemoRunner (si estava parat):
```bash
python scripts/run_demo.py
```

---

## Migrar a un servidor 24/7

Per a la demo continuada, es recomana **Oracle Cloud Free Tier** (4 vCPU ARM · 24 GB RAM · 200 GB SSD — gratuït per sempre). Veure [ROADMAP.md](./ROADMAP.md) §D per a les instruccions detallades.

Configuració mínima al servidor:
1. Instal·lar Python 3.11+, PostgreSQL 14+, les dependències
2. Copiar el projecte i els models entrenats
3. Configurar les variables d'entorn (`.env`)
4. Configurar cron jobs per a l'actualització de dades
5. Configurar systemd per al DemoRunner (restart automàtic)
6. Configurar cron job per a backups diaris

---

---

## Comandes de Telegram

El DemoRunner exposa comandes d'informació i de control a través de Telegram.  Totes les comandes d'informació envien el missatge amb un botó **🔄 Actualitzar** que edita el missatge en el lloc sense necessitat de reescriure la comanda.

### Comandes d'informació

| Comanda | Descripció |
|---------|-----------|
| `/status` | Taula compacta de tots els bots: retorn %, alpha vs B&H, capital en € |
| `/portfolio` | Detall interactiu per bot amb tabs inline. Inclou: capital, retorn, alpha, max drawdown, Sharpe, win-rate, nre trades, posició actual BTC/USDT. Botons: tab per bot + **📈 Gràfic** + **🔄 Actualitzar** |
| `/compare` | Envia dos missatges: (1) foto del gràfic d'evolució de retorn acumulat (tots els bots + B&H) i (2) taula rànquing live amb botó 🔄. El gràfic és un snapshot estàtic; la taula es pot refrescar en el lloc. |
| `/trades` | Últims 15 trades de tots els bots combinats, amb P&L i % de confiança. Inclou resum del dia (total trades + P&L net). Botó 🔄 per refrescar. |
| `/health` | Estat de fonts de dades (OHLCV, Fear & Greed, DB, Binance API) + estat de cada bot (actiu/inactiu). Botó 🔄 per refrescar. |
| `/help` | Llista totes les comandes disponibles |

### Botó 📈 Gràfic (dins /portfolio)

Quan visualitzes un bot a `/portfolio`, el botó **📈 Gràfic** envia una nova foto amb la corba d'equity individual del bot: línia del bot, referència B&H puntejada, zones de guany/pèrdua acolorides, i marcadors ▲/▼ per als BUY/SELL executats.

### Patró de refresh

Cada comanda d'informació s'envia amb un botó inline `🔄 Actualitzar`.  En prémer-lo, Telegram fa una crida `editMessageText` que reemplaça el contingut del missatge en el lloc amb dades fresques — sense crear un missatge nou ni haver de reescriure la comanda.  `/compare` és l'excepció: el refresh actualitza la taula de text però **no** regenera el gràfic (la foto és un snapshot immutable).

### Alertes proactives (sense comanda)

| Alerta | Condició |
|--------|---------|
| ⚠️ Inactivitat | Bot sense activitat > `inactivity_hours` (default: 4h) |
| ⚠️ Dades obsoletes | Candles OHLCV > `data_stale_minutes` (default: 90 min) |
| ⚠️ Drawdown | Drawdown < `drawdown_alert_pct` (default: -10%) |
| 🟢/🔴 Trade executat | Cada BUY o SELL amb preu, mida, portfolio, confiança i F&G actual |
| 🌅 Resum diari | BTC % avui, total trades (BUY/SELL), rànquing bots, snapshot F&G |

### Comandes de control (amb confirmació de doble clic)

| Comanda | Acció |
|---------|-------|
| `/pause <bot_id>` | Pausa el bot (no genera nous senyals; les posicions obertes es mantenen) |
| `/resume <bot_id>` | Reprèn un bot pausat |
| `/close <bot_id>` | Força venda immediata de tota la posició BTC del bot |
| `/restart` | Reinicia el DemoRunner (estat conservat a la BD) |
| `/reset [bot_id]` | Reinicia el capital al valor inicial. Sense argument, reinicia tots els bots. |

Totes les comandes de control mostren primer un missatge de confirmació amb botons **✅ Confirmar** / **❌ Cancel·lar** per evitar accions accidentals.

---

*Última actualització: Març 2026 · v3 (Telegram refresh buttons · gràfic per bot · /compare dos missatges · F&G en alertes · resum diari enriquit)*
