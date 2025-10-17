"""DPO Trainer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from typing import Optional

from .loss import dpo_loss
from .metrics import RewardKLTracker, compute_kl_divergence


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization.

    This trainer implements the DPO algorithm from the paper:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        max_steps: int = 1000,
        warmup_steps: int = 150,
        eval_steps: int = 100,
        save_steps: int = 500,
        output_dir: str = "./outputs",
        use_wandb: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the DPO trainer.

        Args:
            model: The policy model to train
            ref_model: The reference model (frozen)
            tokenizer: Tokenizer for the models
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data (optional)
            beta: DPO beta parameter (KL penalty strength)
            learning_rate: Learning rate for optimizer
            max_steps: Maximum number of training steps
            warmup_steps: Number of warmup steps for learning rate scheduler
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            output_dir: Directory to save outputs
            use_wandb: Whether to use Weights & Biases for logging
            device: Device to train on
        """
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()  # Reference model is frozen
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.beta = beta
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.device = device

        # Optimizer (using RMSprop as in the paper)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler with linear warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

        self.global_step = 0

        # Metric tracking for paper-style plots
        self.reward_kl_tracker = RewardKLTracker()

    def compute_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of the labels under the model.

        Args:
            model: The model to use
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (-100 for ignored tokens)

        Returns:
            log_probs: Log probability of the sequence (summed over non-ignored tokens)
        """
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather the log probabilities of the labels
        # Shift labels to align with logits (next token prediction)
        labels = labels[:, 1:].clone()
        log_probs = log_probs[:, :-1, :]

        # Create mask for non-ignored tokens
        mask = (labels != -100).float()

        # Replace -100 with 0 for gathering (we'll mask them out anyway)
        labels_for_gather = labels.clone()
        labels_for_gather[labels == -100] = 0

        # Gather log probs for each token
        log_probs = torch.gather(
            log_probs, dim=2, index=labels_for_gather.unsqueeze(2)
        ).squeeze(2)

        # Mask out ignored tokens and sum
        log_probs = (log_probs * mask).sum(dim=1)

        return log_probs

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Perform a single training step.

        Args:
            batch: Batch of data containing chosen and rejected sequences

        Returns:
            loss: The loss value
            metrics: Dictionary of metrics
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Compute log probabilities for chosen responses
        policy_chosen_logps = self.compute_logprobs(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )

        # Compute log probabilities for rejected responses
        policy_rejected_logps = self.compute_logprobs(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

        # Compute reference model log probabilities (no gradients)
        with torch.no_grad():
            reference_chosen_logps = self.compute_logprobs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            reference_rejected_logps = self.compute_logprobs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.beta,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Add learning rate to metrics
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        # Compute KL divergence for tracking
        kl_chosen = compute_kl_divergence(policy_chosen_logps, reference_chosen_logps)
        kl_rejected = compute_kl_divergence(policy_rejected_logps, reference_rejected_logps)
        avg_kl = (kl_chosen + kl_rejected) / 2
        metrics["kl_divergence"] = avg_kl

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_metrics = {}
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            policy_chosen_logps = self.compute_logprobs(
                self.model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            policy_rejected_logps = self.compute_logprobs(
                self.model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            reference_chosen_logps = self.compute_logprobs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            reference_rejected_logps = self.compute_logprobs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            loss, metrics = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.beta,
            )

            total_loss += loss.item()
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value.item()

            num_batches += 1

        # Average metrics
        eval_metrics = {f"eval/{k}": v / num_batches for k, v in total_metrics.items()}
        eval_metrics["eval/loss"] = total_loss / num_batches

        return eval_metrics

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Beta: {self.beta}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")

        if self.use_wandb:
            wandb.init(project="dpo-training", config={
                "beta": self.beta,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "max_steps": self.max_steps,
                "model_name": self.model.config._name_or_path,
            })

        pbar = tqdm(total=self.max_steps, desc="Training")
        train_iterator = iter(self.train_dataloader)

        while self.global_step < self.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Restart the iterator if we run out of data
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            loss, metrics = self.train_step(batch)

            # Track metrics for plotting
            self.reward_kl_tracker.update(
                step=self.global_step,
                chosen_reward=metrics['chosen_rewards'].item() if torch.is_tensor(metrics['chosen_rewards']) else metrics['chosen_rewards'],
                rejected_reward=metrics['rejected_rewards'].item() if torch.is_tensor(metrics['rejected_rewards']) else metrics['rejected_rewards'],
                kl=metrics['kl_divergence'],
            )

            # Logging
            if self.global_step % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss'].item():.4f}",
                    "acc": f"{metrics['accuracy'].item():.3f}",
                    "margin": f"{metrics['reward_margin'].item():.3f}",
                    "kl": f"{metrics['kl_divergence']:.3f}",
                })

                if self.use_wandb:
                    wandb.log({f"train/{k}": v.item() if torch.is_tensor(v) else v
                              for k, v in metrics.items()}, step=self.global_step)

            # Evaluation
            if self.eval_steps > 0 and self.global_step % self.eval_steps == 0 and self.global_step > 0:
                eval_metrics = self.evaluate()
                print(f"\nStep {self.global_step} - Eval metrics: {eval_metrics}")

                if self.use_wandb:
                    wandb.log(eval_metrics, step=self.global_step)

            # Save checkpoint
            if self.save_steps > 0 and self.global_step % self.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint()

            self.global_step += 1
            pbar.update(1)

        pbar.close()

        # Final evaluation
        if self.eval_dataloader is not None:
            eval_metrics = self.evaluate()
            print(f"\nFinal evaluation metrics: {eval_metrics}")

            if self.use_wandb:
                wandb.log(eval_metrics, step=self.global_step)

        # Save final model
        self.save_checkpoint(final=True)

        # Save tracking data for plotting
        tracker_path = f"{self.output_dir}/reward_kl_tracker.json"
        self.reward_kl_tracker.save(tracker_path)
        print(f"\nSaved reward-KL tracking data to {tracker_path}")

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        if final:
            save_path = os.path.join(self.output_dir, "final_model")
        else:
            save_path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")

        print(f"\nSaving checkpoint to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
