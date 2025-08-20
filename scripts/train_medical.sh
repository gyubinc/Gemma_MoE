#!/bin/bash
# Medical Domain Training Script
set -e

echo "🏥 Starting Medical Domain Training..."
echo "=================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=3
cd "$(dirname "$0")/.."

# Medical domain training
python run_domain_training.py \
    --domain medical \
    --gpu_id 3 \
    --experiment_name medical_training \
    --config config.yaml

echo "✅ Medical domain training completed!"
