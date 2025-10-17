#!/bin/bash
# Example script for training a small model (suitable for macOS or single GPU)

echo "Training DPO with a small model..."

# You can choose from these small models:
# - gpt2 (124M parameters)
# - facebook/opt-125m (125M parameters)
# - EleutherAI/pythia-160m (160M parameters)
# - EleutherAI/pythia-410m (410M parameters)

MODEL_NAME="gpt2"

uv run python main.py \
    --model_name "$MODEL_NAME" \
    --beta 0.1 \
    --learning_rate 1e-6 \
    --batch_size 4 \
    --max_steps 1000 \
    --eval_steps 100 \
    --save_steps 500 \
    --output_dir "./outputs/${MODEL_NAME}_dpo" \
    --num_workers 4

echo "Training completed! Model saved to outputs/${MODEL_NAME}_dpo/final_model"
