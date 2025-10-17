"""DPO Loss implementation based on the paper."""

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of chosen responses under the policy model, shape (batch_size,)
        policy_rejected_logps: Log probabilities of rejected responses under the policy model, shape (batch_size,)
        reference_chosen_logps: Log probabilities of chosen responses under the reference model, shape (batch_size,)
        reference_rejected_logps: Log probabilities of rejected responses under the reference model, shape (batch_size,)
        beta: Temperature parameter controlling the strength of the KL penalty

    Returns:
        loss: The DPO loss (scalar)
        metrics: Dictionary of metrics for logging
    """
    # Compute the log ratios for policy and reference models
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO loss: -E[log σ(β * (log π(y_w|x) / π_ref(y_w|x) - log π(y_l|x) / π_ref(y_l|x)))]
    logits = beta * (policy_logratios - reference_logratios)
    losses = -F.logsigmoid(logits)

    # Compute the mean loss
    loss = losses.mean()

    # Compute implicit rewards for logging
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    # Compute accuracy (how often the model prefers the chosen response)
    accuracy = (logits > 0).float().mean()

    metrics = {
        "loss": loss.detach(),
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        "accuracy": accuracy,
        "policy_chosen_logps": policy_chosen_logps.mean().detach(),
        "policy_rejected_logps": policy_rejected_logps.mean().detach(),
    }

    return loss, metrics
