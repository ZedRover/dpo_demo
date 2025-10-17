#!/bin/bash
# Example script for local testing with a small model

echo "Running DPO training with a small model for local testing..."

uv run python main.py \
    --model_name "gpt2" \
    --beta 0.1 \
    --learning_rate 1e-6 \
    --batch_size 2 \
    --max_steps 50 \
    --eval_steps 25 \
    --save_steps 50 \
    --output_dir "./outputs/local_test" \
    --sanity_check \
    --num_workers 0

echo "Local test completed! Check outputs/local_test for results."
