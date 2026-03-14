# bots/rl/callbacks.py
"""
Shared SB3 callbacks for all RL agents.

Centralizes ProgressCallback so it does not need to be copy-pasted
in every agent file (PPO / SAC / TD3 / ...).

Usage:
    from bots.rl.callbacks import ProgressCallback
"""
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """Prints a single-line progress bar during SB3 training."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        print(
            f"\r  {self.n_calls}/{self.total_timesteps} steps "
            f"({self.n_calls/self.total_timesteps*100:.1f}%)",
            end="",
            flush=True,
        )
        return True

    def _on_training_end(self) -> None:
        print()  # newline after training ends
