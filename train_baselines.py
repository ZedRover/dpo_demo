"""Train baseline methods for comparison with DPO."""

import argparse
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo import load_hh_rlhf_dataset, DPODataCollator, tokenize_batch


def train_preferred_ft(
    model_name: str = "EleutherAI/pythia-410m",
    output_dir: str = "./outputs/preferred_ft",
    max_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    device: str = "cpu",
):
    """
    Train Preferred-FT baseline.

    This simply fine-tunes on the preferred (chosen) completions only,
    ignoring the rejected completions.
    """
    print("=" * 80)
    print("Training Preferred-FT Baseline")
    print("=" * 80)

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load dataset
    print("\nLoading dataset...")
    train_dataset = load_hh_rlhf_dataset(split="train", sanity_check=True)

    # Only keep the chosen completions for supervised training
    # Extract chosen text from the dataset
    chosen_data = []
    for example in train_dataset:
        chosen_data.append({
            "text": example["chosen"]  # Full conversation including prompt
        })

    print(f"Train dataset size: {len(chosen_data)}")

    # Tokenize the chosen completions
    def tokenize_chosen(examples):
        """Tokenize chosen completions for supervised learning."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )
        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Convert to HuggingFace dataset format
    from datasets import Dataset
    chosen_dataset = Dataset.from_list(chosen_data)
    chosen_dataset = chosen_dataset.map(
        tokenize_chosen,
        batched=True,
        remove_columns=chosen_dataset.column_names,
        desc="Tokenizing chosen completions",
    )

    # Create dataloader
    def collate_fn(examples):
        """Collate function for supervised training."""
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        labels = [torch.tensor(ex["labels"]) for ex in examples]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
        }

    train_dataloader = DataLoader(
        chosen_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining for {max_steps} steps...")
    model.train()

    global_step = 0
    train_iterator = iter(train_dataloader)

    from tqdm import tqdm
    pbar = tqdm(total=max_steps, desc="Training Preferred-FT")

    while global_step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        global_step += 1
        pbar.update(1)

    pbar.close()

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nPreferred-FT model saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train baseline methods")
    parser.add_argument(
        "--method",
        type=str,
        choices=["preferred_ft"],
        default="preferred_ft",
        help="Baseline method to train",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/preferred_ft",
        help="Output directory",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )

    args = parser.parse_args()

    if args.method == "preferred_ft":
        train_preferred_ft(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )


if __name__ == "__main__":
    main()
