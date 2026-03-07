# scripts/train_rl.py
"""
Trains Reinforcement Learning agents (PPO, SAC).

NOTE: RL is conceptually different from supervised ML:
  - Does not learn from a static dataset but simulates episodes in an environment
  - Training time: hours (500k timesteps ≈ 30-90 min depending on hardware)
  - Recommended to run in background or overnight

Usage:
  python scripts/train_rl.py                     # train all (SAC + PPO)
  python scripts/train_rl.py --agents sac        # only SAC
  python scripts/train_rl.py --agents ppo        # only PPO
"""
import argparse
import logging
import os
import sys

sys.path.append(".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_best_config_path(agent_name: str) -> str:
    """
    Load optimized config if it exists, otherwise fall back to experiment_1.

    Args:
        agent_name: Name of the agent (e.g., 'sac', 'ppo')

    Returns:
        Path to the config file to use
    """
    optimized = f"config/training/{agent_name}_optimized.yaml"
    default = f"config/training/{agent_name}_experiment_1.yaml"

    if os.path.exists(optimized):
        logger.info(f"Found optimized config: {optimized}")
        return optimized

    logger.info(f"Using default config: {default}")
    return default

AVAILABLE_AGENTS = ["sac", "ppo"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents (PPO, SAC)")
    parser.add_argument(
        "--agents", nargs="+", default=AVAILABLE_AGENTS,
        choices=AVAILABLE_AGENTS,
        metavar="AGENT",
        help=f"Agents to train (default: all). Options: {AVAILABLE_AGENTS}",
    )
    args = parser.parse_args()

    from bots.rl.trainer import RLTrainer

    logger.info(f"Selected agents: {args.agents}")
    logger.info("NOTE: RL with 500k timesteps can take 30-90 min per agent.")

    results = []
    for agent_key in args.agents:
        config_path = get_best_config_path(agent_key)
        logger.info(f"=== Training {agent_key.upper()} ({config_path}) ===")
        trainer = RLTrainer(config_path=config_path)
        metrics = trainer.run()
        results.append({"agent": agent_key, "config": config_path, **metrics})
        logger.info(f"  {agent_key.upper()} → Return: {metrics['val_return_pct']:.2f}% | "
                    f"MaxDD: {metrics['val_max_drawdown_pct']:.2f}% | Trades: {metrics['val_trades']}")

    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"{'Agent':<8} {'Return%':>8} {'MaxDD%':>10} {'Trades':>8}")
    logger.info("-" * 40)
    for r in results:
        logger.info(f"{r['agent'].upper():<8} {r['val_return_pct']:>8.2f}% {r['val_max_drawdown_pct']:>10.2f}% {r['val_trades']:>8}")
