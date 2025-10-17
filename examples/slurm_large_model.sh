#!/bin/bash
# Example script for submitting a large model training job to SLURM

# Set environment variables for the job
export MODEL_NAME="meta-llama/Llama-2-7b-hf"  # Or any large model you want to use
export BETA=0.1
export LEARNING_RATE=1e-6
export BATCH_SIZE=8
export MAX_STEPS=10000
export OUTPUT_DIR="./outputs/llama2_7b_dpo"

# Submit the job
sbatch slurm_job.sh

echo "SLURM job submitted for model: $MODEL_NAME"
echo "Monitor the job with: squeue -u \$USER"
echo "Check logs in: logs/"
