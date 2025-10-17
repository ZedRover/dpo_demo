"""DPO (Direct Preference Optimization) package."""

from .loss import dpo_loss
from .data import load_hh_rlhf_dataset, DPODataCollator, tokenize_batch
from .trainer import DPOTrainer
from .metrics import RewardKLTracker, compute_kl_divergence, compute_win_rate
from .plotting import (
    plot_reward_kl_frontier,
    plot_training_curves,
    plot_win_rate_by_temperature,
    create_paper_style_plots,
)

__all__ = [
    "dpo_loss",
    "load_hh_rlhf_dataset",
    "DPODataCollator",
    "tokenize_batch",
    "DPOTrainer",
    "RewardKLTracker",
    "compute_kl_divergence",
    "compute_win_rate",
    "plot_reward_kl_frontier",
    "plot_training_curves",
    "plot_win_rate_by_temperature",
    "create_paper_style_plots",
]
