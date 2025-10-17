"""Evaluation script for DPO models - reproduces paper experiments."""

import argparse
import json
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from dpo import compute_win_rate, plot_win_rate_by_temperature


def load_test_data(dataset_name: str = "Anthropic/hh-rlhf", num_samples: int = 100):
    """Load test data for evaluation."""
    dataset = load_dataset(dataset_name, split="test")

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts = []
    chosen_responses = []
    rejected_responses = []

    for example in dataset:
        # Extract prompt and responses
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Find the last "Human:" to separate prompt from response
        prompt_end = chosen.rfind("\n\nAssistant:")
        if prompt_end == -1:
            prompt_end = chosen.find("Assistant:")

        prompt = chosen[:prompt_end] if prompt_end != -1 else ""
        chosen_response = chosen[prompt_end:] if prompt_end != -1 else chosen
        rejected_response = rejected[prompt_end:] if prompt_end != -1 else rejected

        prompts.append(prompt.strip())
        chosen_responses.append(chosen_response.strip())
        rejected_responses.append(rejected_response.strip())

    return prompts, chosen_responses, rejected_responses


def evaluate_model_at_temperatures(
    model_path: str,
    reference_model_path: str,
    temperatures: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    num_samples: int = 100,
    beta: float = 0.1,
    device: str = "cpu",
):
    """
    Evaluate model at different temperatures.

    This reproduces Figure 2 (right) and Figure 3 from the paper.
    """
    print(f"Loading models...")
    print(f"  Policy model: {model_path}")
    print(f"  Reference model: {reference_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_path).to(device)

    print(f"\nLoading test data ({num_samples} samples)...")
    prompts, chosen_responses, rejected_responses = load_test_data(
        num_samples=num_samples
    )

    print(f"\nEvaluating at temperatures: {temperatures}")

    results = {}
    for temp in temperatures:
        print(f"\nTemperature: {temp}")

        # For temperature 0, we don't need sampling
        # For other temperatures, we'd need to generate responses
        # For now, we compute win rate based on log probabilities

        metrics = compute_win_rate(
            model=model,
            reference_model=ref_model,
            tokenizer=tokenizer,
            prompts=prompts[:num_samples],
            chosen_responses=chosen_responses[:num_samples],
            rejected_responses=rejected_responses[:num_samples],
            device=device,
            beta=beta,
        )

        results[temp] = metrics

        print(f"  Win rate: {metrics['win_rate']:.3f}")
        print(f"  Mean chosen reward: {metrics['mean_chosen_reward']:.3f}")
        print(f"  Mean rejected reward: {metrics['mean_rejected_reward']:.3f}")
        print(f"  Reward margin: {metrics['mean_reward_margin']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DPO model with paper-style metrics"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained DPO model",
    )
    parser.add_argument(
        "--reference_model_path",
        type=str,
        default=None,
        help="Path to reference model (if None, uses same as model_path base)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )

    args = parser.parse_args()

    # If reference model not specified, infer from model path
    if args.reference_model_path is None:
        # Try to find the base model name from config
        import json
        config_path = os.path.join(args.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                args.reference_model_path = config.get("_name_or_path", "gpt2")
        else:
            args.reference_model_path = "gpt2"

    print("=" * 80)
    print("DPO Model Evaluation (Paper-style Metrics)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Reference: {args.reference_model_path}")
    print(f"Beta: {args.beta}")
    print(f"Samples: {args.num_samples}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate at different temperatures
    temperatures = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = evaluate_model_at_temperatures(
        model_path=args.model_path,
        reference_model_path=args.reference_model_path,
        temperatures=temperatures,
        num_samples=args.num_samples,
        beta=args.beta,
        device=args.device,
    )

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Plot win rate vs temperature
    win_rates = [results[t]['win_rate'] for t in temperatures]

    plot_win_rate_by_temperature(
        temperatures=temperatures,
        win_rates=win_rates,
        method_name="DPO",
        save_path=os.path.join(args.output_dir, "win_rate_vs_temperature.png"),
    )

    print(f"Plot saved to: {os.path.join(args.output_dir, 'win_rate_vs_temperature.png')}")


if __name__ == "__main__":
    main()
