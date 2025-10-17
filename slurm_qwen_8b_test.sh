#!/bin/bash
#SBATCH --job-name=dpo_qwen_test
#SBATCH --output=logs/dpo_qwen_test_%j.out
#SBATCH --error=logs/dpo_qwen_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00                 # 2 hours for quick test
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Quick test script for Qwen2.5-8B
# Tests model loading and runs 50 training steps

echo "=========================================="
echo "Quick Test: DPO with Qwen2.5-8B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

mkdir -p logs

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Print GPU info
nvidia-smi

# Quick test configuration
MODEL_NAME="Qwen/Qwen2.5-8B"
OUTPUT_DIR="./outputs/qwen_test_${SLURM_JOB_ID}"

echo ""
echo "Test Configuration:"
echo "Model: $MODEL_NAME"
echo "Steps: 50 (quick test)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run quick test
uv run python main.py \
    --model_name "$MODEL_NAME" \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --batch_size 2 \
    --max_steps 50 \
    --eval_steps 25 \
    --save_steps 50 \
    --output_dir "$OUTPUT_DIR" \
    --sanity_check \
    --num_workers 4

STATUS=$?

echo "=========================================="
if [ $STATUS -eq 0 ]; then
    echo "✓ Test passed! Ready for full training."
    echo "Run full training with: sbatch slurm_qwen_8b.sh"
else
    echo "✗ Test failed! Check errors above."
fi
echo "End: $(date)"
echo "=========================================="

exit $STATUS
