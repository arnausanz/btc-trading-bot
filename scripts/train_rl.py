# scripts/train_rl.py
import logging
import sys

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from bots.rl.trainer import RLTrainer

if __name__ == "__main__":
    trainer = RLTrainer(config_path="config/training/rl_experiment_2.yaml")
    metrics = trainer.run()
    print(f"\nReturn: {metrics['val_return_pct']}% | Drawdown: {metrics['val_max_drawdown_pct']}% | Trades: {metrics['val_trades']}")