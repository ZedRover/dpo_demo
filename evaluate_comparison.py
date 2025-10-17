"""Evaluate and compare multiple methods - reproduces Figure 3 from DPO paper."""

import argparse
import json
import os
from typing import List, Dict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from dpo import compute_win_rate


def load_test_data(dataset_name: str = "Anthropic/hh-rlhf", num_samples: int = 100):
    """Load test data for evaluation."""
    dataset = load_dataset(dataset_name, split="test")

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts = []
    chosen_responses = []

    for example in dataset:
        # Extract prompt and chosen response
        chosen = example["chosen"]

        # Find the last "Assistant:" to separate prompt from response
        prompt_end = chosen.rfind("\n\nAssistant:")
        if prompt_end == -1:
            prompt_end = chosen.find("Assistant:")

        prompt = chosen[:prompt_end] if prompt_end != -1 else ""
        chosen_response = chosen[prompt_end:] if prompt_end != -1 else chosen

        prompts.append(prompt.strip())
        chosen_responses.append(chosen_response.strip())

    return prompts, chosen_responses


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float = 1.0,
    max_new_tokens: int = 256,
    num_samples: int = 1,
    device: str = "cpu",
) -> List[str]:
    """Generate responses from a model."""
    all_responses = []

    model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Generating (temp={temperature})"):
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate multiple samples if needed
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode responses
            responses = []
            for output in outputs:
                response = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                responses.append(response.strip())

            all_responses.append(responses if num_samples > 1 else responses[0])

    return all_responses


def best_of_n_sampling(
    model,
    ref_model,
    tokenizer,
    prompts: List[str],
    n: int = 128,
    temperature: float = 1.0,
    beta: float = 0.1,
    device: str = "cpu",
) -> List[str]:
    """
    Best of N sampling: generate N samples and return the one with highest reward.

    This is computationally expensive but provides a strong baseline.
    """
    print(f"\nBest of {n} sampling (this may take a while)...")

    best_responses = []

    model.eval()
    ref_model.eval()

    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Best of {n}"):
            # Generate N samples
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=min(n, 128),  # Limit for memory
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode all responses
            responses = []
            for output in outputs:
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response.strip())

            # Compute reward for each response using implicit DPO reward
            rewards = []
            for response in responses:
                full_text = prompt + " " + response
                inputs_full = tokenizer(full_text, return_tensors="pt").to(device)

                # Get log probabilities from policy and reference
                policy_output = model(**inputs_full)
                ref_output = ref_model(**inputs_full)

                policy_logps = torch.nn.functional.log_softmax(policy_output.logits, dim=-1)
                ref_logps = torch.nn.functional.log_softmax(ref_output.logits, dim=-1)

                # Average log probability as a simple reward proxy
                policy_lp = policy_logps.mean().item()
                ref_lp = ref_logps.mean().item()

                # Implicit DPO reward
                reward = beta * (policy_lp - ref_lp)
                rewards.append(reward)

            # Select best response
            best_idx = np.argmax(rewards)
            best_responses.append(responses[best_idx])

    return best_responses


def evaluate_all_methods(
    methods: Dict[str, Dict],
    test_prompts: List[str],
    test_chosen: List[str],
    temperatures: List[float],
    num_samples: int = 100,
    beta: float = 0.1,
    device: str = "cpu",
) -> Dict:
    """
    Evaluate all methods at different temperatures.

    Args:
        methods: Dictionary mapping method name to dict with 'model_path' and optional 'ref_path'
        test_prompts: Test prompts
        test_chosen: Chosen (reference) responses to compare against
        temperatures: List of temperatures to evaluate
        num_samples: Number of test samples
        beta: DPO beta parameter
        device: Device to use

    Returns:
        results: Dictionary mapping method -> temperature -> metrics
    """
    results = {}

    for method_name, method_config in methods.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {method_name}")
        print(f"{'=' * 80}")

        model_path = method_config["model_path"]
        ref_path = method_config.get("ref_path", model_path)

        # Load models
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        ref_model = AutoModelForCausalLM.from_pretrained(ref_path).to(device)

        results[method_name] = {}

        # Handle Best of N specially
        if method_name.startswith("Best of"):
            n = int(method_name.split()[2])
            for temp in temperatures:
                print(f"\nTemperature: {temp}")

                # Generate with Best of N
                generated = best_of_n_sampling(
                    model, ref_model, tokenizer,
                    test_prompts[:num_samples],
                    n=n,
                    temperature=temp,
                    beta=beta,
                    device=device,
                )

                # Compute win rate against chosen responses
                # Simple comparison: count how often generated is preferred
                # In practice, you'd use GPT-4 or human evaluation
                # For now, use a simple heuristic (length + diversity)
                win_count = 0
                for gen, ref in zip(generated, test_chosen[:num_samples]):
                    # Simple heuristic: prefer longer, more diverse responses
                    gen_score = len(set(gen.split())) / max(len(gen.split()), 1)
                    ref_score = len(set(ref.split())) / max(len(ref.split()), 1)
                    if gen_score > ref_score:
                        win_count += 1

                win_rate = win_count / len(generated)

                results[method_name][temp] = {
                    "win_rate": win_rate,
                    "num_samples": len(generated),
                }

                print(f"  Win rate: {win_rate:.3f}")

        else:
            # Standard evaluation for other methods
            for temp in temperatures:
                print(f"\nTemperature: {temp}")

                # Generate responses
                generated = generate_responses(
                    model, tokenizer,
                    test_prompts[:num_samples],
                    temperature=temp,
                    device=device,
                )

                # Compute win rate (using simple heuristic for demo)
                win_count = 0
                for gen, ref in zip(generated, test_chosen[:num_samples]):
                    gen_score = len(set(gen.split())) / max(len(gen.split()), 1)
                    ref_score = len(set(ref.split())) / max(len(ref.split()), 1)
                    if gen_score > ref_score:
                        win_count += 1

                win_rate = win_count / len(generated)

                results[method_name][temp] = {
                    "win_rate": win_rate,
                    "num_samples": len(generated),
                }

                print(f"  Win rate: {win_rate:.3f}")

    return results


def plot_comparison(
    results: Dict,
    temperatures: List[float],
    save_path: str = "comparison_plot.png",
):
    """
    Create Figure 3 style comparison plot.

    Shows win rate vs temperature for multiple methods.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Define colors for each method (matching paper style)
    colors = {
        "DPO": "#FFA500",  # Orange
        "Preferred-FT": "#FF1493",  # Pink
        "Best of 128": "#FFD700",  # Gold
        "Pythia-2.8B": "#008080",  # Teal
    }

    for method_name, method_results in results.items():
        temps = []
        win_rates = []
        for temp in sorted(method_results.keys()):
            temps.append(temp)
            win_rates.append(method_results[temp]["win_rate"])

        color = colors.get(method_name, None)
        plt.plot(temps, win_rates, 'o-', label=method_name,
                linewidth=2, markersize=8, color=color)

    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel("Sampling temperature", fontsize=14)
    plt.ylabel("Win rate", fontsize=14)
    plt.title("Anthropic-HH Dialogue Win Rate vs Chosen", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple methods (reproduce Figure 3)"
    )
    parser.add_argument(
        "--dpo_path",
        type=str,
        default="./outputs/local_test/final_model",
        help="Path to DPO model",
    )
    parser.add_argument(
        "--preferred_ft_path",
        type=str,
        default="./outputs/preferred_ft",
        help="Path to Preferred-FT model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model name (for reference)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of test samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_prompts, test_chosen = load_test_data(num_samples=args.num_samples)

    # Define methods to compare
    methods = {
        "DPO": {
            "model_path": args.dpo_path,
            "ref_path": args.base_model,
        },
        "Preferred-FT": {
            "model_path": args.preferred_ft_path,
            "ref_path": args.base_model,
        },
        "Best of 128": {
            "model_path": args.preferred_ft_path,  # Use Preferred-FT as base
            "ref_path": args.base_model,
        },
        "Pythia-2.8B": {
            "model_path": args.base_model,
            "ref_path": args.base_model,
        },
    }

    # Temperatures to evaluate (matching paper)
    temperatures = [0.25, 0.50, 0.75, 1.00]

    # Evaluate all methods
    results = evaluate_all_methods(
        methods=methods,
        test_prompts=test_prompts,
        test_chosen=test_chosen,
        temperatures=temperatures,
        num_samples=args.num_samples,
        device=args.device,
    )

    # Save results
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create comparison plot
    plot_path = os.path.join(args.output_dir, "figure3_comparison.png")
    plot_comparison(results, temperatures, save_path=plot_path)


if __name__ == "__main__":
    main()
