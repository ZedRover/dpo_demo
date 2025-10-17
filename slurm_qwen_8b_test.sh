#!/bin/bash
#SBATCH --job-name=dpo_qwen_test
#SBATCH --output=logs/dpo_qwen_test_%j.out
#SBATCH --error=logs/dpo_qwen_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=h100:1
#SBATCH --time=01:00:00                 # 2 hours for quick test
#SBATCH --mem=64G

# Quick test script for Qwen3-8B
# Tests model loading and runs 50 training steps
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Quick Test: DPO with Qwen3-8B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

module load cuda/12.2
module load python/3.12.4

source .venv/bin/activate

mkdir -p logs

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Print GPU info
nvidia-smi

# Quick test configuration
MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_DIR="./outputs/qwen_test_${SLURM_JOB_ID}"

echo ""
echo "Test Configuration:"
echo "Model: $MODEL_NAME"
echo "Steps: 50 (quick test)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run quick test
python main.py \
    --model_name "$MODEL_NAME" \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --batch_size 1 \
    --max_steps 50 \
    --eval_steps 25 \
    --save_steps 50 \
    --warmup_steps 10 \
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
