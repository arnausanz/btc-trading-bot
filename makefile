# Makefile
.PHONY: help setup up down logs db-shell test lint clean

help:
	@echo "Comandas disponibles:"
	@echo "  make setup     - Instal·la dependències i aixeca serveis"
	@echo "  make up        - Aixeca els serveis Docker"
	@echo "  make down      - Para els serveis Docker"
	@echo "  make logs      - Mostra logs dels serveis"
	@echo "  make db-shell  - Obre una shell de PostgreSQL"
	@echo "  make test      - Executa els tests"
	@echo "  make lint      - Executa el linter"
	@echo "  make clean     - Elimina artefactes temporals"

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

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

validate-data:
	poetry run python scripts/validate_data.py

compare:
	poetry run python scripts/run_comparison.py

mlflow:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

optimize:
	poetry run python scripts/optimize_bots.py

train:
	poetry run python scripts/train_models.py

train-rl:
	poetry run python scripts/train_rl.py

demo:
	poetry run python scripts/run_demo.py

dashboard:
	poetry run streamlit run monitoring/dashboard.py