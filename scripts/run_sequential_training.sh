#!/bin/bash
# Sequential Domain Training Pipeline
# 순차적으로 하나씩 실행하여 메모리 문제 방지

set -e

echo "🚀 Starting Sequential Domain Training Pipeline"
echo "=============================================="

# Configuration
PROJECT_DIR="/data/disk5/internship_disk/gyubin/Qwen_MoE"
CONDA_ENV="gyubin"
DOMAINS=("medical" "law" "math" "code")

# Change to project directory
cd "${PROJECT_DIR}"

# Setup environment
export CUDA_VISIBLE_DEVICES=3

echo "📋 Training Order: Medical → Law → Math → Code"
echo "⚠️  Each domain trains completely before starting the next"
echo ""

# Function to run single domain training
run_single_domain() {
    local domain=$1
    local session_name="train_${domain}"
    
    echo "🎯 Starting ${domain} domain training..."
    echo "======================================="
    
    # Kill any existing session for this domain
    tmux kill-session -t "${session_name}" 2>/dev/null || true
    
    # Create new tmux session
    tmux new-session -d -s "${session_name}" -c "${PROJECT_DIR}"
    
    # Setup environment and run training
    tmux send-keys -t "${session_name}" "conda deactivate && conda activate ${CONDA_ENV}" Enter
    tmux send-keys -t "${session_name}" "export CUDA_VISIBLE_DEVICES=3" Enter
    tmux send-keys -t "${session_name}" "python run_domain_training.py --domain ${domain} --gpu_id 3 --experiment_name ${domain}_training" Enter
    
    echo "✅ ${domain} training started in tmux session: ${session_name}"
    echo "📺 Monitor with: tmux attach-session -t ${session_name}"
    
    # Wait for training to complete
    echo "⏳ Waiting for ${domain} training to complete..."
    while tmux has-session -t "${session_name}" 2>/dev/null; do
        sleep 30
        echo "⏳ ${domain} training still in progress..."
    done
    
    # Check if training was successful
    if [ -d "domain_models/${domain}/final_adapter" ]; then
        echo "✅ ${domain} training completed successfully!"
        echo "📁 Model saved at: domain_models/${domain}/final_adapter"
    else
        echo "❌ ${domain} training may have failed - no model found"
        echo "🔍 Check logs in experiments/ directory"
    fi
    
    echo ""
}

# Main execution
main() {
    echo "🔍 Environment check..."
    
    # Check conda environment
    if ! conda info --envs | grep -q "${CONDA_ENV}"; then
        echo "❌ Conda environment '${CONDA_ENV}' not found"
        exit 1
    fi
    
    # Check GPU
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "❌ NVIDIA GPU not available"
        exit 1
    fi
    
    echo "✅ Environment check passed"
    echo ""
    
    # Run each domain sequentially
    for domain in "${DOMAINS[@]}"; do
        run_single_domain "${domain}"
        
        # Small break between domains
        echo "⏱️  Brief pause before next domain..."
        sleep 10
    done
    
    # Final summary
    echo "🎉 All Domain Training Completed!"
    echo "================================="
    echo ""
    echo "📊 Training Results:"
    for domain in "${DOMAINS[@]}"; do
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
}

# Run the main function
main "$@"
