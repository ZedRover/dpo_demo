"""Evaluation metrics for DPO training."""

import torch
import numpy as np
from typing import Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_kl_divergence(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
) -> float:
    """
    Compute KL divergence between policy and reference model.

    Args:
        policy_logps: Log probabilities from policy model (batch_size,)
        reference_logps: Log probabilities from reference model (batch_size,)

    Returns:
        kl: Average KL divergence
    """
    # KL(π || π_ref) = E[log π - log π_ref]
    kl = (policy_logps - reference_logps).mean().item()
    return kl


def compute_reward_accuracy(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    """
    Compute accuracy of reward model (how often chosen > rejected).

    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses

    Returns:
        accuracy: Fraction of examples where chosen > rejected
    """
    correct = (chosen_rewards > rejected_rewards).float()
    return correct.mean().item()


class RewardKLTracker:
    """
    Track reward and KL divergence over training for plotting.

    This is used to create the reward-KL frontier plot from Figure 2 (left) in the paper.
    """

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.kls = []
        self.chosen_rewards = []
        self.rejected_rewards = []

    def update(
        self,
        step: int,
        chosen_reward: float,
        rejected_reward: float,
        kl: float,
    ):
        """Add a new data point."""
        self.steps.append(step)
        # Average reward (as in the paper)
        avg_reward = (chosen_reward + rejected_reward) / 2
        self.rewards.append(avg_reward)
        self.kls.append(kl)
        self.chosen_rewards.append(chosen_reward)
        self.rejected_rewards.append(rejected_reward)

    def get_frontier_data(self) -> Dict[str, List[float]]:
        """Get data for plotting the reward-KL frontier."""
        return {
            'steps': self.steps,
            'rewards': self.rewards,
            'kls': self.kls,
            'chosen_rewards': self.chosen_rewards,
            'rejected_rewards': self.rejected_rewards,
        }

    def save(self, path: str):
        """Save tracking data to file."""
        import json
        data = self.get_frontier_data()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load tracking data from file."""
        import json
        tracker = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        tracker.steps = data['steps']
        tracker.rewards = data['rewards']
        tracker.kls = data['kls']
        tracker.chosen_rewards = data['chosen_rewards']
        tracker.rejected_rewards = data['rejected_rewards']
        return tracker


def compute_win_rate(
    model: PreTrainedModel,
    reference_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str],
    device: str = 'cpu',
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    Compute win rate: how often the model prefers chosen over rejected.

    This metric is used in Figure 2 (right) and Figure 3 in the paper.

    Args:
        model: The policy model to evaluate
        reference_model: The reference model
        tokenizer: Tokenizer
        prompts: List of prompts
        chosen_responses: List of chosen (preferred) responses
        rejected_responses: List of rejected (non-preferred) responses
        device: Device to run on
        beta: DPO beta parameter

    Returns:
        metrics: Dictionary containing win_rate and other statistics
    """
    model.eval()
    reference_model.eval()

    wins = []
    chosen_rewards = []
    rejected_rewards = []

    with torch.no_grad():
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            # Tokenize
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected

            chosen_inputs = tokenizer(chosen_text, return_tensors='pt').to(device)
            rejected_inputs = tokenizer(rejected_text, return_tensors='pt').to(device)

            # Get log probabilities from policy
            policy_chosen_out = model(**chosen_inputs)
            policy_rejected_out = model(**rejected_inputs)

            policy_chosen_logps = torch.nn.functional.log_softmax(
                policy_chosen_out.logits, dim=-1
            )
            policy_rejected_logps = torch.nn.functional.log_softmax(
                policy_rejected_out.logits, dim=-1
            )

            # Get log probabilities from reference
            ref_chosen_out = reference_model(**chosen_inputs)
            ref_rejected_out = reference_model(**rejected_inputs)

            ref_chosen_logps = torch.nn.functional.log_softmax(
                ref_chosen_out.logits, dim=-1
            )
            ref_rejected_logps = torch.nn.functional.log_softmax(
                ref_rejected_out.logits, dim=-1
            )

            # Compute implicit rewards: r(x,y) = β * log(π(y|x) / π_ref(y|x))
            policy_chosen_lp = policy_chosen_logps.mean().item()
            policy_rejected_lp = policy_rejected_logps.mean().item()
            ref_chosen_lp = ref_chosen_logps.mean().item()
            ref_rejected_lp = ref_rejected_logps.mean().item()

            chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
            rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)

            chosen_rewards.append(chosen_reward)
            rejected_rewards.append(rejected_reward)

            # Win if chosen reward > rejected reward
            wins.append(chosen_reward > rejected_reward)

    win_rate = np.mean(wins)

    return {
        'win_rate': win_rate,
        'mean_chosen_reward': np.mean(chosen_rewards),
        'mean_rejected_reward': np.mean(rejected_rewards),
        'mean_reward_margin': np.mean(chosen_rewards) - np.mean(rejected_rewards),
    }
