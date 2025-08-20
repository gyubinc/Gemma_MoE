#!/bin/bash
# Code Domain Training Script
set -e

echo "💻 Starting Code Domain Training..."
echo "================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=3
cd "$(dirname "$0")/.."

# Code domain training
python run_domain_training.py \
    --domain code \
    --gpu_id 3 \
    --experiment_name code_training \
    --config config.yaml

echo "✅ Code domain training completed!"
