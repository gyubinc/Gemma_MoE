#!/bin/bash
# Single Script for All Domain Training
# 하나의 스크립트로 4개 도메인 순차 훈련

set -e

echo "🚀 Starting All Domain Training in Single Script"
echo "==============================================="

# Setup environment
export CUDA_VISIBLE_DEVICES=3
cd /data/disk5/internship_disk/gyubin/Qwen_MoE

echo "📋 Training Order: Medical → Law → Math → Code"
echo "⚠️  Sequential execution to prevent memory issues"
echo ""

# Function to train single domain
train_domain() {
    local domain=$1
    echo "🎯 Starting ${domain} domain training..."
    echo "======================================"
    
    python run_domain_training.py \
        --domain ${domain} \
        --gpu_id 3 \
        --experiment_name ${domain}_training \
        --config config.yaml
    
    # Check if training was successful
    if [ -d "domain_models/${domain}/final_adapter" ]; then
        echo "✅ ${domain} training completed successfully!"
        echo "📁 Model saved at: domain_models/${domain}/final_adapter"
    else
        echo "❌ ${domain} training failed - no model found"
        exit 1
    fi
    
    echo ""
}

# Main execution
echo "🔍 Environment check..."
echo "✅ Using current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

echo "✅ Starting domain training sequence..."
echo ""

# Train each domain sequentially
train_domain "medical"
train_domain "law" 
train_domain "math"
train_domain "code"

# Final summary
echo "🎉 All Domain Training Completed!"
echo "================================="
echo ""
echo "📊 Training Results:"
for domain in medical law math code; do
    if [ -d "domain_models/${domain}/final_adapter" ]; then
        echo "  ✅ ${domain}: Successfully trained"
    else
        echo "  ❌ ${domain}: Training failed"
    fi
done

echo ""
echo "📁 Next steps:"
echo "  1. Check evaluation results in experiments/ directories"
echo "  2. Run MoE router training: python train_moe_router.py"
echo "  3. Test the complete MoE model"
