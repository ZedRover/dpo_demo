"""Configuration management for DPO training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name: str = field(
        default="gpt2",
        metadata={"help": "Pretrained model name or path (e.g., 'gpt2', 'facebook/opt-125m', 'EleutherAI/pythia-160m')"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision (requires bitsandbytes)"}
    )


@dataclass
class DataConfig:
    """Configuration for the data."""

    dataset_name: str = field(
        default="Anthropic/hh-rlhf",
        metadata={"help": "HuggingFace dataset name"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Maximum prompt length"}
    )
    sanity_check: bool = field(
        default=False,
        metadata={"help": "Use only a small subset of data for testing"}
    )


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # DPO specific
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta parameter (KL penalty strength)"}
    )

    # Optimizer
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate"}
    )
    warmup_steps: int = field(
        default=150,
        metadata={"help": "Number of warmup steps"}
    )

    # Training loop
    batch_size: int = field(
        default=4,
        metadata={"help": "Training batch size"}
    )
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Evaluation batch size"}
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "Maximum number of training steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluate every N steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps"}
    )

    # Logging
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use Weights & Biases for logging"}
    )
    wandb_project: str = field(
        default="dpo-training",
        metadata={"help": "Weights & Biases project name"}
    )

    # Output
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Output directory"}
    )

    # System
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of data loading workers"}
    )


@dataclass
class DPOConfig:
    """Complete DPO training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
