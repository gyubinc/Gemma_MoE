#!/bin/bash
# Qwen-MoE Training Pipeline for A6000 46GB VRAM
# Sequential domain training with MoE router training

set -e
echo "🚀 Starting Qwen-MoE Training Pipeline"
echo "====================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0
CONFIG_FILE="../config.yaml"
EXPERIMENT_NAME="qwen_moe_training"

# Change to project root directory
cd "$(dirname "$0")/.."

# Validate environment
echo "🔍 Validating environment..."
python -c "
from utils import validate_environment, check_data_availability
import sys

if not validate_environment():
    print('❌ Environment validation failed')
    sys.exit(1)

availability = check_data_availability(['medical', 'law', 'math', 'code'])
print('📊 Data availability:', availability)

for domain, available in availability.items():
    if not available:
        print(f'❌ Data not available for {domain} domain')
        sys.exit(1)

print('✅ Environment validation passed')
"

# 1단계: 도메인별 LoRA 훈련
echo ""
echo "🎯 Stage 1: Domain-specific LoRA Training"
echo "==========================================="

DOMAINS=("medical" "law" "math" "code")
TRAINED_DOMAINS=()

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "🔧 Training ${domain} domain..."
    echo "----------------------------------------"
    
    # Check if domain model already exists
    if [ -d "./domain_models/${domain}/final_adapter" ]; then
        echo "⚠️  ${domain} model already exists, skipping..."
        TRAINED_DOMAINS+=("${domain}")
        continue
    fi
    
    # Train domain model
    if python train_domain_models.py \
        --config $CONFIG_FILE \
        --domain $domain \
        --gpu_id 0 \
        --experiment_name $EXPERIMENT_NAME; then
        
        echo "✅ ${domain} training completed!"
        TRAINED_DOMAINS+=("${domain}")
    else
        echo "❌ ${domain} training failed!"
        echo "🛑 Pipeline stopped due to training failure"
        exit 1
    fi
    
    # Clear GPU memory between domains
    echo "🧹 Clearing GPU memory..."
    python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
print('✅ GPU memory cleared')
"
    
    echo "⏳ Waiting 10 seconds before next domain..."
    sleep 10
done

echo ""
echo "📊 Training Summary:"
echo "==================="
echo "Successfully trained domains: ${TRAINED_DOMAINS[*]}"
echo "Total domains trained: ${#TRAINED_DOMAINS[@]}"

# 2단계: MoE Router 훈련
echo ""
echo "🎯 Stage 2: MoE Router Training"
echo "==============================="

if [ ${#TRAINED_DOMAINS[@]} -ge 2 ]; then
    echo "🔀 Preparing MoE Router training..."
    echo "📁 Available domain adapters:"
    
    for domain in "${TRAINED_DOMAINS[@]}"; do
        adapter_path="./domain_models/${domain}/final_adapter"
        if [ -d "$adapter_path" ]; then
            echo "  ✅ ${domain}: $adapter_path"
        else
            echo "  ❌ ${domain}: Not found"
        fi
    done
    
    echo ""
    echo "🚀 Starting MoE Router training..."
    
    # Train MoE router
    if python train_moe_router.py \
        --config $CONFIG_FILE \
        --gpu_id 0 \
        --experiment_name "${EXPERIMENT_NAME}_moe"; then
        
        echo "✅ MoE Router training completed!"
        echo "📁 MoE model saved in experiments/${EXPERIMENT_NAME}_moe_*/moe_model/final_moe_model/"
    else
        echo "❌ MoE Router training failed!"
        echo "🛑 Pipeline stopped due to MoE training failure"
        exit 1
    fi
    
else
    echo "❌ Need at least 2 trained domains for MoE training"
    echo "📊 Only ${#TRAINED_DOMAINS[@]} domains trained"
fi

echo ""
echo "🎉 Full Pipeline Completed!"
echo "=========================="
echo "📁 Results:"
echo "  - Domain models: ./domain_models/"
echo "  - MoE model: ./experiments/${EXPERIMENT_NAME}_moe_*/moe_model/final_moe_model/"
echo "  - Experiments: ./experiments/"
echo "  - Logs: ./experiments/${EXPERIMENT_NAME}_*/"
echo ""
echo "🎯 Next steps:"
echo "  1. Test individual domain models"
echo "  2. Evaluate MoE model performance"
echo "  3. Deploy MoE model for inference"
echo ""
echo "✅ Ready to use the trained MoE model!"


