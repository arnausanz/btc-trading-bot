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


AVAILABLE_AGENTS = {
    "ppo": "config/models/ppo.yaml",
    "sac": "config/models/sac.yaml",
}


def get_best_config_path(agent_name: str) -> str:
    """
    Carrega el config optimitzat si existeix, sinó usa el YAML base.

    Args:
        agent_name: Nom de l'agent ('sac' | 'ppo')

    Returns:
        Path al fitxer de config a usar
    """
    optimized = f"config/models/{agent_name}_optimized.yaml"
    default = AVAILABLE_AGENTS[agent_name]

    if os.path.exists(optimized):
        logger.info(f"Found optimized config: {optimized}")
        return optimized

    logger.info(f"Using default config: {default}")
    return default

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents (PPO, SAC)")
    parser.add_argument(
        "--agents", nargs="+", default=list(AVAILABLE_AGENTS.keys()),
        choices=list(AVAILABLE_AGENTS.keys()),
        metavar="AGENT",
        help=f"Agents to train (default: all). Options: {AVAILABLE_AGENTS}",
    )
    args = parser.parse_args()

    from bots.rl.trainer import RLTrainer

    selected = args.agents
    logger.info(f"Selected agents: {selected}")
    logger.info("NOTE: RL with 500k timesteps can take 30-90 min per agent.")

    results = []
    for agent_key in selected:
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
