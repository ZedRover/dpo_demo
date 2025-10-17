#!/bin/bash
#SBATCH --job-name=dpo_qwen2.5_8b
#SBATCH --output=logs/dpo_qwen_%j.out
#SBATCH --error=logs/dpo_qwen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                    # 1 GPU (A100/V100 recommended)
#SBATCH --time=48:00:00                 # 48 hours for full training
#SBATCH --mem=64G                       # 64GB RAM for 8B model
#SBATCH --partition=gpu

# Qwen2.5-8B specific configuration
# Requires:
# - GPU with >=24GB VRAM (A100-40GB/80GB, V100-32GB, or A6000)
# - 64GB+ system RAM
# - CUDA 11.8+ / 12.1+

# Print job information
echo "=========================================="
echo "DPO Training with Qwen2.5-8B"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load required modules (adjust for your cluster)
# Example for common HPC setups:
# module load cuda/12.1
# module load python/3.10
# module load gcc/11.2.0

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export CUDA_LAUNCH_BLOCKING=0        # Async CUDA ops for speed

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Model configuration for Qwen2.5-8B
MODEL_NAME="Qwen/Qwen2.5-8B"

# Training hyperparameters optimized for 8B model
BETA=0.1                    # DPO beta parameter
LEARNING_RATE=5e-7          # Lower LR for larger model
BATCH_SIZE=4                # Adjust based on GPU memory
MAX_STEPS=10000             # Full training run
EVAL_STEPS=500              # Evaluate every 500 steps
SAVE_STEPS=1000             # Save checkpoint every 1000 steps
WARMUP_STEPS=150            # Warmup steps
MAX_LENGTH=512              # Sequence length
MAX_PROMPT_LENGTH=256       # Prompt length

# Output configuration
OUTPUT_DIR="./outputs/qwen2.5_8b_${SLURM_JOB_ID}"

# Optional: Use gradient checkpointing to save memory
GRADIENT_CHECKPOINTING=true

# Optional: Use 8-bit quantization (requires bitsandbytes)
LOAD_IN_8BIT=false  # Set to true if GPU memory < 40GB

echo "Training Configuration:"
echo "----------------------------------------"
echo "Model: $MODEL_NAME"
echo "Beta: $BETA"
echo "Learning rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE"
echo "Max steps: $MAX_STEPS"
echo "Eval steps: $EVAL_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Max length: $MAX_LENGTH"
echo "Output directory: $OUTPUT_DIR"
echo "Load in 8-bit: $LOAD_IN_8BIT"
echo "=========================================="

# Build command
CMD="uv run python main.py \
    --model_name $MODEL_NAME \
    --beta $BETA \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --max_length $MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --output_dir $OUTPUT_DIR \
    --num_workers 4"

# Add optional flags
if [ "$LOAD_IN_8BIT" = true ]; then
    CMD="$CMD --load_in_8bit"
fi

# Add W&B logging if available
# Uncomment and set your W&B API key
# export WANDB_API_KEY="your_wandb_api_key"
# CMD="$CMD --use_wandb"

echo "Running command:"
echo "$CMD"
echo "=========================================="

# Run training
eval $CMD

# Training status
TRAIN_STATUS=$?

echo "=========================================="
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $TRAIN_STATUS"
fi
echo "End time: $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Optional: Print training summary
if [ -f "$OUTPUT_DIR/reward_kl_tracker.json" ]; then
    echo ""
    echo "Training metrics saved:"
    ls -lh "$OUTPUT_DIR/reward_kl_tracker.json"
fi

exit $TRAIN_STATUS
