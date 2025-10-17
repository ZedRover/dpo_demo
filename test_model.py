"""Test the trained DPO model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model(model_path: str = "./outputs/local_test/final_model"):
    """Test the trained model with a simple prompt."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Test prompt
    prompt = "Human: How do I make pizza?\n\nAssistant:"

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./outputs/local_test/final_model")
    args = parser.parse_args()

    test_model(args.model_path)
