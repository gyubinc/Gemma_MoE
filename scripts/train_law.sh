#!/bin/bash
# Law Domain Training Script
set -e

echo "⚖️ Starting Law Domain Training..."
echo "==============================="

# Environment setup
export CUDA_VISIBLE_DEVICES=3
cd "$(dirname "$0")/.."

# Law domain training
python run_domain_training.py \
    --domain law \
    --gpu_id 3 \
    --experiment_name law_training \
    --config config.yaml

echo "✅ Law domain training completed!"
