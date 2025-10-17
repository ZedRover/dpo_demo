# Method Comparison Guide

This guide explains how to reproduce Figure 3 from the DPO paper, which compares multiple methods:

## Methods Compared

1. **DPO (Direct Preference Optimization)** - Our main method
2. **Preferred-FT** - Supervised fine-tuning on only the preferred completions
3. **Best of N** - Sample N completions and select the best according to a reward model
4. **Base Model** - The original pretrained model (e.g., Pythia-2.8B)

## Quick Start

To run the full comparison:

```bash
bash examples/run_comparison.sh
```

This script will:
1. Train the Preferred-FT baseline (if not already trained)
2. Evaluate all methods at different temperatures
3. Generate a comparison plot similar to Figure 3

## Individual Steps

### Step 1: Train Preferred-FT Baseline

```bash
uv run python train_baselines.py \
    --method preferred_ft \
    --model_name EleutherAI/pythia-410m \
    --output_dir ./outputs/preferred_ft \
    --max_steps 1000 \
    --batch_size 4 \
    --learning_rate 1e-5
```

**What this does:** Trains a model using supervised learning on only the preferred (chosen) completions from the preference dataset. This is a simple baseline that ignores the rejected completions.

### Step 2: Run Comparison Evaluation

```bash
uv run python evaluate_comparison.py \
    --dpo_path ./outputs/local_test/final_model \
    --preferred_ft_path ./outputs/preferred_ft \
    --base_model EleutherAI/pythia-410m \
    --num_samples 100 \
    --output_dir ./comparison_results
```

**What this does:**
- Loads all trained models (DPO, Preferred-FT, Base)
- Evaluates each at multiple temperatures (0.25, 0.5, 0.75, 1.0)
- Implements Best of N sampling (generates N samples and picks the best)
- Computes win rates against the chosen responses in the test set
- Generates comparison plots

## Understanding the Methods

### DPO (Your Trained Model)
- Uses preference pairs directly without explicit reward model
- Optimizes implicit reward: r(x,y) = β log(π(y|x) / π_ref(y|x))
- More stable and efficient than PPO-based RLHF

### Preferred-FT
- Simple supervised baseline
- Only trains on chosen completions, ignoring rejected ones
- Fast to train but doesn't explicitly optimize for preferences

### Best of N
- Sample N completions from the policy
- Score each using the (implicit) reward model
- Return the highest-scoring one
- Very strong but computationally expensive at inference time
- In paper, they use N=128

### Base Model
- The original pretrained model without any preference training
- Serves as a baseline to show improvement

## Output

The comparison script generates:

1. **comparison_results.json** - Raw win rate data for all methods
2. **figure3_comparison.png** - Visual comparison plot showing:
   - X-axis: Sampling temperature
   - Y-axis: Win rate vs chosen responses
   - Multiple lines for different methods

## Expected Results

Based on the paper (Figure 3):

- **DPO** should show strong, stable performance across temperatures
- **Preferred-FT** typically performs worse than DPO but better than base
- **Best of 128** can match or exceed DPO but requires 128× more computation
- **Base model** should be the weakest

## Customization

### Using Different Models

```bash
# Use a larger model
bash examples/run_comparison.sh
# Edit the BASE_MODEL variable to "EleutherAI/pythia-2.8b"
```

### Adjusting Sample Count

```bash
# Evaluate on more samples (slower but more reliable)
uv run python evaluate_comparison.py \
    --num_samples 500 \
    --output_dir ./comparison_results
```

### Changing Best of N

Edit `evaluate_comparison.py` to change N:

```python
"Best of 128": {  # Change 128 to your desired N
    "model_path": args.preferred_ft_path,
    "ref_path": args.base_model,
},
```

## Notes on Evaluation

**Important:** The current implementation uses a simple heuristic for computing win rates (vocabulary diversity). For production use, you should:

1. **Use GPT-4 evaluation** (as in the paper)
   - See `evaluate.py` for GPT-4 evaluation examples
   - Requires OpenAI API access

2. **Human evaluation**
   - Most accurate but expensive
   - See paper Appendix D.3 for human study details

3. **Learned reward model**
   - Train an explicit reward model on preferences
   - Use it to score all completions

## Computational Requirements

- **Preferred-FT training**: ~5-10 minutes on CPU
- **Comparison evaluation**: ~20-30 minutes on CPU with 100 samples
- **Best of N**: Scales linearly with N (128× slower than regular sampling)

For faster evaluation, use GPU:
```bash
uv run python evaluate_comparison.py --device cuda
```

## Troubleshooting

### Out of Memory
- Reduce `--num_samples`
- Reduce batch size
- Use smaller base model

### Models Not Found
Make sure you've run the DPO training first:
```bash
bash examples/local_test.sh
```

### Slow Best of N
- Reduce N (e.g., Best of 16 instead of 128)
- Use smaller test set
- Enable GPU

## Paper Reference

This comparison reproduces:
- **Figure 3** (left): "Anthropic-HH Dialogue Win Rate vs Chosen"
- Shows DPO outperforms Preferred-FT and matches Best of 128 efficiency

Original paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
