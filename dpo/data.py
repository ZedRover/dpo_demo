"""Data loading and preprocessing for DPO training."""

from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def load_hh_rlhf_dataset(split: str = "train", sanity_check: bool = False):
    """
    Load the Anthropic HH-RLHF dataset.

    Args:
        split: Dataset split to load ('train' or 'test')
        sanity_check: If True, only load a small subset for quick testing

    Returns:
        dataset: The loaded dataset
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)

    if sanity_check:
        dataset = dataset.select(range(min(100, len(dataset))))

    # The dataset has 'chosen' and 'rejected' fields
    # We need to extract the prompt and responses
    def extract_prompt_and_responses(example):
        # The chosen and rejected fields contain the full conversation
        # We need to split into prompt and response
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Find the last "Human:" to separate prompt from response
        # The format is: Human: ... Assistant: ...
        prompt_end = chosen.rfind("\n\nAssistant:")
        if prompt_end == -1:
            # Fallback: use the first Assistant response as split point
            prompt_end = chosen.find("Assistant:")

        prompt = chosen[:prompt_end] if prompt_end != -1 else ""
        chosen_response = chosen[prompt_end:] if prompt_end != -1 else chosen
        rejected_response = rejected[prompt_end:] if prompt_end != -1 else rejected

        return {
            "prompt": prompt.strip(),
            "chosen": chosen_response.strip(),
            "rejected": rejected_response.strip(),
        }

    dataset = dataset.map(extract_prompt_and_responses)

    return dataset


def tokenize_batch(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    max_prompt_length: int = 256,
) -> dict[str, Any]:
    """
    Tokenize a batch of prompts and responses.

    Args:
        batch: Batch containing 'prompt', 'chosen', and 'rejected' fields
        tokenizer: Tokenizer to use
        max_length: Maximum total sequence length
        max_prompt_length: Maximum prompt length

    Returns:
        Tokenized batch with input_ids, attention_mask, and labels
    """
    # Tokenize prompts
    prompt_tokens = tokenizer(
        batch["prompt"],
        max_length=max_prompt_length,
        truncation=True,
        add_special_tokens=True,
    )

    # Tokenize chosen and rejected responses
    chosen_tokens = tokenizer(
        batch["chosen"],
        max_length=max_length - max_prompt_length,
        truncation=True,
        add_special_tokens=False,
    )

    rejected_tokens = tokenizer(
        batch["rejected"],
        max_length=max_length - max_prompt_length,
        truncation=True,
        add_special_tokens=False,
    )

    # Combine prompt + chosen
    chosen_input_ids = []
    chosen_attention_mask = []
    chosen_labels = []

    for i in range(len(batch["prompt"])):
        # Concatenate prompt and chosen response
        input_ids = prompt_tokens["input_ids"][i] + chosen_tokens["input_ids"][i]
        attention_mask = prompt_tokens["attention_mask"][i] + chosen_tokens["attention_mask"][i]

        # Labels: -100 for prompt tokens (ignored in loss), actual tokens for response
        labels = [-100] * len(prompt_tokens["input_ids"][i]) + chosen_tokens["input_ids"][i]

        chosen_input_ids.append(input_ids)
        chosen_attention_mask.append(attention_mask)
        chosen_labels.append(labels)

    # Combine prompt + rejected
    rejected_input_ids = []
    rejected_attention_mask = []
    rejected_labels = []

    for i in range(len(batch["prompt"])):
        input_ids = prompt_tokens["input_ids"][i] + rejected_tokens["input_ids"][i]
        attention_mask = prompt_tokens["attention_mask"][i] + rejected_tokens["attention_mask"][i]
        labels = [-100] * len(prompt_tokens["input_ids"][i]) + rejected_tokens["input_ids"][i]

        rejected_input_ids.append(input_ids)
        rejected_attention_mask.append(attention_mask)
        rejected_labels.append(labels)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }


@dataclass
class DPODataCollator:
    """
    Data collator for DPO training.
    Pads the sequences to the same length in each batch.
    """

    tokenizer: PreTrainedTokenizer
    padding: str = "longest"
    max_length: int = 512

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of features."""
        batch_size = len(features)

        # Extract chosen and rejected sequences
        chosen_input_ids = [f["chosen_input_ids"] for f in features]
        chosen_attention_mask = [f["chosen_attention_mask"] for f in features]
        chosen_labels = [f["chosen_labels"] for f in features]

        rejected_input_ids = [f["rejected_input_ids"] for f in features]
        rejected_attention_mask = [f["rejected_attention_mask"] for f in features]
        rejected_labels = [f["rejected_labels"] for f in features]

        # Pad chosen sequences
        chosen_padded = self.tokenizer.pad(
            {
                "input_ids": chosen_input_ids,
                "attention_mask": chosen_attention_mask,
            },
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad labels for chosen
        max_label_length = max(len(l) for l in chosen_labels)
        chosen_labels_padded = [
            l + [-100] * (max_label_length - len(l)) for l in chosen_labels
        ]

        # Pad rejected sequences
        rejected_padded = self.tokenizer.pad(
            {
                "input_ids": rejected_input_ids,
                "attention_mask": rejected_attention_mask,
            },
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad labels for rejected
        max_label_length = max(len(l) for l in rejected_labels)
        rejected_labels_padded = [
            l + [-100] * (max_label_length - len(l)) for l in rejected_labels
        ]

        return {
            "chosen_input_ids": chosen_padded["input_ids"],
            "chosen_attention_mask": chosen_padded["attention_mask"],
            "chosen_labels": torch.tensor(chosen_labels_padded),
            "rejected_input_ids": rejected_padded["input_ids"],
            "rejected_attention_mask": rejected_padded["attention_mask"],
            "rejected_labels": torch.tensor(rejected_labels_padded),
        }
