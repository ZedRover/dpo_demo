"""Main training script for DPO."""

import argparse
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, DataConfig, TrainingConfig
from dpo import DPOTrainer, DPODataCollator, load_hh_rlhf_dataset, tokenize_batch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with DPO")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Pretrained model name (e.g., 'gpt2', 'facebook/opt-125m', 'EleutherAI/pythia-160m')",
    )
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")

    # Data arguments
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--sanity_check", action="store_true", help="Use small subset for testing")

    # Training arguments
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DPO Training Configuration")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Beta: {args.beta}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load models
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=args.load_in_8bit,
        device_map="auto" if args.load_in_8bit else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    print("Loading reference model (will be frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=args.load_in_8bit,
        device_map="auto" if args.load_in_8bit else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_hh_rlhf_dataset(split="train", sanity_check=args.sanity_check)
    eval_dataset = load_hh_rlhf_dataset(split="test", sanity_check=args.sanity_check)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenize_fn = partial(
        tokenize_batch,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )

    # Create data collator
    data_collator = DPODataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    print("\nTraining completed!")
    print(f"Model saved to: {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
