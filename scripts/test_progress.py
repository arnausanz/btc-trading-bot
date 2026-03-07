#!/usr/bin/env python3
# scripts/test_progress.py
"""
Script de prova del sistema de progrés tqdm.
Triga ~30 segons i exercita DOS tipus de feedback:
  1. Optimització clàssica (Optuna + backtest)
  2. Entrenament ML (RandomForest, CV)

Executa'l amb:
  python scripts/test_progress.py

Si tot va bé veuràs:
  - Barra de trials Optuna (s/trial, best_sharpe actualitzat)
  - Barra de ticks del backtest per cada trial (leave=False, desapareix)
  - Barra de folds CV (leave=False, desapareix)
  - Línies de resum amb tqdm.write
"""
import logging
import sys

sys.path.append(".")

# Suprimim logs de fons perquè el demo sigui net
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from tqdm import tqdm

# ── 1. Optimització clàssica: DCABot, 5 trials ────────────────────────────────
tqdm.write("=" * 55)
tqdm.write("  TEST PROGRÉS — optimització + entrenament ML")
tqdm.write("=" * 55)
tqdm.write("\n[1/2] Optimitzant DCABot (5 trials)...\n")

from bots.classical.dca_bot import DCABot
from core.backtesting.optimizer import BotOptimizer
from core.config import TRAIN_UNTIL

optimizer = BotOptimizer(
    bot_class=DCABot,
    base_config_path="config/bots/dca.yaml",
    param_space={
        "buy_every_n_ticks": {"type": "int",   "low": 6,   "high": 168},
        "buy_size":          {"type": "float", "low": 0.05, "high": 0.5},
    },
    n_trials=5,           # 5 en lloc de 30 per anar ràpid
    train_until=TRAIN_UNTIL,
)
optimizer.run()

# ── 2. Entrenament ML: RandomForest mínim ─────────────────────────────────────
tqdm.write("\n[2/2] Entrenant RandomForest (n_estimators=20, 5 folds CV)...\n")

from data.processing.dataset import DatasetBuilder
from bots.ml.random_forest import RandomForestModel

builder = DatasetBuilder(
    symbol="BTC/USDT",
    timeframes=["1h"],
    forward_window=24,
    threshold_pct=0.005,
    train_until=TRAIN_UNTIL,
)
X, y = builder.build()
tqdm.write(f"  Dataset: {X.shape[0]} files x {X.shape[1]} features\n")

model = RandomForestModel(n_estimators=20, max_depth=5)   # petit per anar ràpid
model.train(X, y)

tqdm.write("\n" + "=" * 55)
tqdm.write("  ✓ Test completat — el sistema de progrés funciona!")
tqdm.write("=" * 55 + "\n")
