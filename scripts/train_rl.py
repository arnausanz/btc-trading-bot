# scripts/train_rl.py
"""
Trains Reinforcement Learning agents (PPO, SAC, TD3) in baseline, on-chain,
professional, and multi-timeframe variants.

NOTE: RL is conceptually different from supervised ML:
  - Does not learn from a static dataset but simulates episodes in an environment
  - Training time: hours (500k timesteps approx 30-90 min depending on hardware)
  - Recommended to run in background or overnight

Usage:
  python scripts/train_rl.py                                           # PPO + SAC baseline
  python scripts/train_rl.py --agents ppo_professional sac_professional
  python scripts/train_rl.py --agents td3_professional                 # C3: TD3 sentiment
  python scripts/train_rl.py --agents td3_multiframe                   # C3: TD3 multi-TF
  python scripts/train_rl.py --agents td3_professional --smoke         # smoke: 50k steps
  python scripts/train_rl.py --agents ppo_professional --smoke

Available agents:
  Baseline     : ppo, sac
  On-chain     : ppo_onchain, sac_onchain   (require external data in DB)
  Professional : ppo_professional, sac_professional
  C3 Advanced  : td3_professional, td3_multiframe
                 (require: download_data.py, download_fear_greed.py, update_futures.py)

PRE-REQUISITES for professional/TD3 variants:
  python scripts/download_data.py       (ensures 12h and 1h+4h candles in DB)
  python scripts/download_fear_greed.py
  python scripts/update_futures.py

--smoke flag: overrides total_timesteps to 50k for a quick env/reward sanity check.
"""
import argparse
import logging
import os
import sys
import copy
import yaml

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SMOKE_TIMESTEPS = 50_000


AVAILABLE_AGENTS = {
    # Baseline: technical only
    "ppo":              "config/models/ppo.yaml",
    "sac":              "config/models/sac.yaml",
    # On-chain: technical + Fear&Greed + funding rate + Open Interest + hash-rate
    "ppo_onchain":      "config/models/ppo_onchain.yaml",
    "sac_onchain":      "config/models/sac_onchain.yaml",
    # Professional: 12h swing + regime + on-chain + position state
    "ppo_professional": "config/models/ppo_professional.yaml",
    "sac_professional": "config/models/sac_professional.yaml",
    # C3 Advanced: TD3 variants
    "td3_professional": "config/models/td3_professional.yaml",
    "td3_multiframe":   "config/models/td3_multiframe.yaml",
}

# Agents excluded from default "train all" to avoid overwriting existing models
_DEFAULT_AGENTS = ["ppo", "sac"]


def get_best_config_path(agent_name: str) -> str:
    """
    Loads the optimized config if it exists, else uses the base YAML.
    """
    optimized = f"config/models/{agent_name}_optimized.yaml"
    default = AVAILABLE_AGENTS[agent_name]

    if os.path.exists(optimized):
        logger.info(f"Found optimized config: {optimized}")
        return optimized

    logger.info(f"Using default config: {default}")
    return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents (PPO, SAC, TD3)")
    parser.add_argument(
        "--agents", nargs="+", default=_DEFAULT_AGENTS,
        choices=list(AVAILABLE_AGENTS.keys()),
        metavar="AGENT",
        help=(
            f"Agents to train (default: {_DEFAULT_AGENTS}). "
            f"Options: {list(AVAILABLE_AGENTS.keys())}"
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            f"Smoke test mode: overrides total_timesteps to {_SMOKE_TIMESTEPS}. "
            "Use to verify that env and reward work before full training."
        ),
    )
    args = parser.parse_args()

    from bots.rl.trainer import RLTrainer

    selected = args.agents
    logger.info(f"Selected agents: {selected}")

    if args.smoke:
        logger.info(
            f"SMOKE MODE: total_timesteps overridden to {_SMOKE_TIMESTEPS}. "
            "For full training, remove --smoke."
        )
    else:
        logger.info("NOTE: RL with 500k timesteps can take 30-90 min per agent.")

    results = []
    for agent_key in selected:
        config_path = get_best_config_path(agent_key)
        logger.info(f"=== Training {agent_key.upper()} ({config_path}) ===")

        if args.smoke:
            import tempfile
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["training"]["model"]["total_timesteps"] = _SMOKE_TIMESTEPS
            cfg["training"]["experiment_name"] = cfg["training"]["experiment_name"] + "_smoke"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                yaml.dump(cfg, tmp)
                tmp_path = tmp.name
            trainer = RLTrainer(config_path=tmp_path)
            metrics = trainer.run()
            os.unlink(tmp_path)
        else:
            trainer = RLTrainer(config_path=config_path)
            metrics = trainer.run()

        results.append({"agent": agent_key, "config": config_path, **metrics})
        logger.info(
            f"  {agent_key.upper()} -> "
            f"Return: {metrics['val_return_pct']:.2f}% | "
            f"MaxDD: {metrics['val_max_drawdown_pct']:.2f}% | "
            f"Trades: {metrics['val_trades']}"
        )

    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"{'Agent':<18} {'Return%':>8} {'MaxDD%':>10} {'Trades':>8}")
    logger.info("-" * 50)
    for r in results:
        logger.info(
            f"{r['agent'].upper():<18} "
            f"{r['val_return_pct']:>8.2f}% "
            f"{r['val_max_drawdown_pct']:>10.2f}% "
            f"{r['val_trades']:>8}"
        )
