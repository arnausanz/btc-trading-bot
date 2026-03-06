# scripts/train_rl.py
import logging
import sys

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from bots.rl.trainer import RLTrainer

    CONFIGS = [
        "config/training/sac_experiment_1.yaml",
        "config/training/ppo_experiment_1.yaml",
    ]

    results = []
    for config_path in CONFIGS:
        logger.info(f"=== {config_path} ===")
        trainer = RLTrainer(config_path=config_path)
        metrics = trainer.run()
        results.append({"config": config_path, **metrics})

    logger.info("=== COMPARATIVA FINAL ===")
    logger.info(f"{'Config':<35} {'Return':>8} {'Drawdown':>10} {'Trades':>8}")
    logger.info("-" * 65)
    for r in results:
        logger.info(
            f"{r['config']:<35} "
            f"{r['val_return_pct']:>8.2f}% "
            f"{r['val_max_drawdown_pct']:>10.2f}% "
            f"{r['val_trades']:>8}"
        )