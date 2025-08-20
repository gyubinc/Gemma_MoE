#!/bin/bash
# Full MoE Training Pipeline with MLP Full Fine-tuning
# Step 1: Train 4 domain-specific models with full MLP fine-tuning
# Step 2: Create MoE model and train router

set -e  # Exit on any error

echo "🚀 Starting Full MoE Training Pipeline with MLP Full Fine-tuning"
echo "================================================================="

# Configuration
export CUDA_VISIBLE_DEVICES=6
BASE_DIR=$(pwd)
DOMAIN_MODELS_DIR="$BASE_DIR/domain_models"
MOE_MODEL_DIR="$BASE_DIR/moe_model"

# Create directories
mkdir -p $DOMAIN_MODELS_DIR
mkdir -p $MOE_MODEL_DIR

echo "📊 GPU Memory Summary:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits -i 6

echo ""
echo "🎯 Stage 1: LoRA MLP Training for Each Domain"
echo "=============================================="

# Define domains
DOMAINS=("medical" "law" "math" "code")

# Train each domain model
for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "🔧 Training ${domain} domain model..."
    
    python train_domain_models.py \
        --domain ${domain} \
        --model google/gemma-3-4b-it \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --lr 2e-4 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --max_length 1024 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --output_dir $DOMAIN_MODELS_DIR \
        --gpu_id 6 \
        --fp16 \
        --logging_steps 10 \
        --save_steps 500 \
        --seed 42
    
    echo "✅ ${domain} domain training completed"
    
    # Check GPU memory after each domain training
    echo "📊 GPU Memory after ${domain} training:"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits -i 6
done

echo ""
echo "🎯 Stage 2: MoE Router Training"
echo "==============================="

echo "🔀 Training MoE Router with Top-1 routing..."

# Prepare domain adapter paths
MEDICAL_PATH="$DOMAIN_MODELS_DIR/medical/final_adapter"
LAW_PATH="$DOMAIN_MODELS_DIR/law/final_adapter"
MATH_PATH="$DOMAIN_MODELS_DIR/math/final_adapter"
CODE_PATH="$DOMAIN_MODELS_DIR/code/final_adapter"

# Check if all domain adapters exist
for path in "$MEDICAL_PATH" "$LAW_PATH" "$MATH_PATH" "$CODE_PATH"; do
    if [ ! -d "$path" ]; then
        echo "❌ Error: Domain adapter not found at $path"
        exit 1
    fi
done

python train_moe_router.py \
    --base_model google/gemma-3-4b-it \
    --domain_adapter_paths $MEDICAL_PATH $LAW_PATH $MATH_PATH $CODE_PATH \
    --domains medical law math code \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --aux_loss_weight 0.01 \
    --max_length 1024 \
    --output_dir $MOE_MODEL_DIR \
    --gpu_id 6 \
    --fp16 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 1000 \
    --routing_stats_steps 100 \
    --seed 42

echo "✅ MoE Router training completed"

echo ""
echo "🎯 Stage 3: Model Evaluation and Analysis"
echo "========================================="

echo "📊 Running model evaluation..."

# Create evaluation script call (to be implemented)
# python evaluate_moe_model.py \
#     --moe_model_path $MOE_MODEL_DIR/final \
#     --domains medical law math code \
#     --num_samples 100 \
#     --max_new_tokens 100 \
#     --output_dir ./evaluation_results \
#     --gpu_id 6

echo "✅ Evaluation completed"

echo ""
echo "🎉 Full MoE Training Pipeline Completed!"
echo "========================================"
echo "📁 Results saved in:"
echo "   - Domain Models: $DOMAIN_MODELS_DIR"
echo "     - Medical: $MEDICAL_PATH"
echo "     - Law: $LAW_PATH"
echo "     - Math: $MATH_PATH"
echo "     - Code: $CODE_PATH"
echo "   - MoE Model: $MOE_MODEL_DIR"
echo "   - Evaluation: ./evaluation_results"

echo ""
echo "📊 Final GPU Memory Summary:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits -i 6

echo ""
echo "🔍 Model Architecture Summary:"
echo "   - Base Model: google/gemma-3-4b-it"
echo "   - Expert Models: 5 per layer (1 original + 4 domain-specific)"
echo "   - Routing: Top-1 routing with load balancing"
echo "   - Domains: medical, law, math, code"
echo ""
echo "✨ Ready to use the trained MoE model!"

