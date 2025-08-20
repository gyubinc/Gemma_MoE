#!/bin/bash
# Math Domain Training Script
set -e

echo "🔢 Starting Math Domain Training..."
echo "================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=3
cd "$(dirname "$0")/.."

# Math domain training
python run_domain_training.py \
    --domain math \
    --gpu_id 3 \
    --experiment_name math_training \
    --config config.yaml

echo "✅ Math domain training completed!"
