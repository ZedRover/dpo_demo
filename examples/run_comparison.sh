#!/bin/bash

# Script to reproduce Figure 3 comparison from DPO paper
# Compares DPO, Preferred-FT, Best of 128, and base model

echo "=========================================="
echo "Reproducing Figure 3: Method Comparison"
echo "=========================================="

# Configuration
BASE_MODEL="EleutherAI/pythia-410m"
DPO_MODEL="./outputs/local_test/final_model"
PREFERRED_FT_MODEL="./outputs/preferred_ft"
NUM_SAMPLES=50
MAX_STEPS=100

# Step 1: Train Preferred-FT baseline if it doesn't exist
if [ ! -d "$PREFERRED_FT_MODEL" ]; then
    echo ""
    echo "Step 1: Training Preferred-FT baseline..."
    echo "=========================================="
    uv run python train_baselines.py \
        --method preferred_ft \
        --model_name $BASE_MODEL \
        --output_dir $PREFERRED_FT_MODEL \
        --max_steps $MAX_STEPS \
        --batch_size 2 \
        --learning_rate 1e-5 \
        --device cpu
else
    echo ""
    echo "Step 1: Preferred-FT model already exists, skipping training"
fi

# Step 2: Run comparison evaluation
echo ""
echo "Step 2: Evaluating all methods..."
echo "=========================================="
uv run python evaluate_comparison.py \
    --dpo_path $DPO_MODEL \
    --preferred_ft_path $PREFERRED_FT_MODEL \
    --base_model $BASE_MODEL \
    --num_samples $NUM_SAMPLES \
    --output_dir ./comparison_results \
    --device cpu

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: ./comparison_results/"
echo "Plot saved to: ./comparison_results/figure3_comparison.png"
echo "=========================================="
