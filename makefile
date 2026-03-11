# Makefile — BTC Trading Bot
# Ús: make <target> [ARGS="..."]
# Exemple: make compare ARGS="--bots hold ppo sac --no-ml"
.PHONY: help \
        setup up down logs db-shell \
        data-download data-update data-check \
        train train-rl train-rl-onchain \
        optimize optimize-rl optimize-rl-onchain optimize-bots \
        compare \
        demo dashboard mlflow \
        test lint clean

# ── Utilitats ─────────────────────────────────────────────────────────────────
PYTHON = poetry run python
PYTEST = poetry run pytest
RUFF   = poetry run ruff

# ── Ajuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║             BTC Trading Bot — Makefile                  ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  INFRAESTRUCTURA                                         ║"
	@echo "║    make setup            Instal·la deps + aixeca Docker  ║"
	@echo "║    make up               Aixeca serveis Docker           ║"
	@echo "║    make down             Para serveis Docker             ║"
	@echo "║    make logs             Logs en temps real              ║"
	@echo "║    make db-shell         Shell PostgreSQL                ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  DADES                                                   ║"
	@echo "║    make data-download    Descàrrega inicial completa     ║"
	@echo "║    make data-update      Actualitza totes les fonts      ║"
	@echo "║    make data-check       Comprova completesa de dades    ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  ENTRENAMENT                                             ║"
	@echo "║    make train            Entrena tots els models ML      ║"
	@echo "║    make train-rl         Entrena PPO + SAC (baseline)    ║"
	@echo "║    make train-rl-onchain Entrena PPO + SAC on-chain      ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  OPTIMITZACIÓ (Optuna)                                   ║"
	@echo "║    make optimize         Optimitza tots els models ML    ║"
	@echo "║    make optimize-rl      Optimitza PPO + SAC baseline    ║"
	@echo "║    make optimize-rl-onchain  Optimitza on-chain          ║"
	@echo "║    make optimize-bots    Optimitza bots clàssics         ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  AVALUACIÓ                                               ║"
	@echo "║    make compare          Compara tots els bots           ║"
	@echo "║    make compare ARGS=... Compara selecció (veure baix)   ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  DEMO I MONITORATGE                                      ║"
	@echo "║    make demo             Llança el DemoRunner            ║"
	@echo "║    make dashboard        Streamlit dashboard             ║"
	@echo "║    make mlflow           MLflow UI (port 5001)           ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  QUALITAT                                                ║"
	@echo "║    make test             Executa tots els tests          ║"
	@echo "║    make lint             Ruff linter                     ║"
	@echo "║    make clean            Elimina artefactes temporals    ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  ARGS addicionals per a 'compare':                       ║"
	@echo "║    --bots hold ppo sac mean_reversion momentum           ║"
	@echo "║    --no-rl  --no-ml  --no-classic                       ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""

# ── Infraestructura ───────────────────────────────────────────────────────────
setup:
	poetry install
	docker compose up -d
	@echo "Esperant que la DB estigui ready..."
	@sleep 5
	@echo "Setup completat!"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

db-shell:
	docker exec -it btc_trading_db psql -U btc_user -d btc_trading

# ── Dades ─────────────────────────────────────────────────────────────────────
data-download:
	@echo "=== Descàrrega inicial de dades OHLCV ==="
	$(PYTHON) scripts/download_data.py
	@echo "=== Descàrrega de Fear & Greed ==="
	$(PYTHON) scripts/download_fear_greed.py
	@echo "=== Descàrrega de Futures (funding + OI) ==="
	$(PYTHON) scripts/download_futures.py
	@echo "=== Descàrrega de Blockchain metrics ==="
	$(PYTHON) scripts/download_blockchain.py
	@echo "=== Descàrrega de Binance Vision (OI 5m) ==="
	$(PYTHON) scripts/download_binance_vision.py

data-update:
	@echo "=== Actualització incremental de totes les fonts ==="
	$(PYTHON) scripts/update_data.py
	$(PYTHON) scripts/update_fear_greed.py
	$(PYTHON) scripts/update_futures.py
	$(PYTHON) scripts/update_blockchain.py
	$(PYTHON) scripts/update_binance_vision.py

data-check:
	$(PYTHON) scripts/check_data_completeness.py

# ── Entrenament ───────────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train_models.py $(ARGS)

train-rl:
	$(PYTHON) scripts/train_rl.py --agents ppo sac $(ARGS)

train-rl-onchain:
	$(PYTHON) scripts/train_rl.py --agents ppo_onchain sac_onchain $(ARGS)

# ── Optimització ──────────────────────────────────────────────────────────────
optimize:
	$(PYTHON) scripts/optimize_models.py --no-rl $(ARGS)

optimize-rl:
	$(PYTHON) scripts/optimize_models.py --agents ppo sac $(ARGS)

optimize-rl-onchain:
	$(PYTHON) scripts/optimize_models.py --agents ppo_onchain sac_onchain $(ARGS)

optimize-bots:
	$(PYTHON) scripts/optimize_bots.py $(ARGS)

# ── Avaluació ─────────────────────────────────────────────────────────────────
compare:
	$(PYTHON) scripts/run_comparison.py $(ARGS)

# ── Demo i monitoratge ────────────────────────────────────────────────────────
demo:
	$(PYTHON) scripts/run_demo.py $(ARGS)

dashboard:
	poetry run streamlit run monitoring/dashboard.py

mlflow:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# ── Qualitat ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v $(ARGS)

lint:
	$(RUFF) check .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
