#!/bin/bash
#SBATCH --job-name=dpo_training
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (modify based on your cluster setup)
# module load cuda/11.8
# module load python/3.10

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print GPU information
nvidia-smi

# Model and training configuration
MODEL_NAME=${MODEL_NAME:-"gpt2"}  # Default to gpt2, can be overridden
BETA=${BETA:-0.1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-5000}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/${SLURM_JOB_ID}"}

echo "=========================================="
echo "Training Configuration:"
echo "Model: $MODEL_NAME"
echo "Beta: $BETA"
echo "Learning rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE"
echo "Max steps: $MAX_STEPS"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run training
uv run python main.py \
    --model_name "$MODEL_NAME" \
    --beta "$BETA" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --eval_steps 500 \
    --save_steps 1000 \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb \
    --num_workers 4

echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
