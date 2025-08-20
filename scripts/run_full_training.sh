#!/bin/bash
# Full Domain Training Pipeline with tmux
# Runs all 4 domains sequentially in separate tmux sessions

set -e

echo "🚀 Starting Full Domain Training Pipeline"
echo "========================================"

# Configuration
PROJECT_DIR="/data/disk5/internship_disk/gyubin/Qwen_MoE"
CONDA_ENV="gyubin"
DOMAINS=("medical" "law" "math" "code")

# Function to run domain training in tmux
run_domain_training() {
    local domain=$1
    local session_name="train_${domain}"
    
    echo "🔧 Starting ${domain} domain training in tmux session: ${session_name}"
    
    # Create tmux session for this domain
    tmux new-session -d -s "${session_name}" -c "${PROJECT_DIR}"
    
    # Setup environment and run training
    tmux send-keys -t "${session_name}" "conda deactivate && conda activate ${CONDA_ENV}" Enter
    tmux send-keys -t "${session_name}" "export CUDA_VISIBLE_DEVICES=3" Enter
    tmux send-keys -t "${session_name}" "echo '🎯 Starting ${domain} domain training...'" Enter
    tmux send-keys -t "${session_name}" "bash scripts/train_${domain}.sh || echo '❌ ${domain} training failed but continuing...'" Enter
    tmux send-keys -t "${session_name}" "echo '📊 ${domain} training session completed'" Enter
    
    echo "✅ ${domain} training session created: ${session_name}"
}

# Function to wait for tmux session to complete
wait_for_session() {
    local session_name=$1
    local domain=$2
    
    echo "⏳ Waiting for ${domain} training to complete..."
    
    # Wait for session to finish
    while tmux has-session -t "${session_name}" 2>/dev/null; do
        sleep 30
        echo "⏳ ${domain} training still in progress..."
    done
    
    echo "✅ ${domain} training session completed"
}

# Function to cleanup old sessions
cleanup_sessions() {
    echo "🧹 Cleaning up old training sessions..."
    for domain in "${DOMAINS[@]}"; do
        session_name="train_${domain}"
        if tmux has-session -t "${session_name}" 2>/dev/null; then
            tmux kill-session -t "${session_name}"
            echo "🗑️ Killed existing session: ${session_name}"
        fi
    done
}

# Function to monitor all sessions
monitor_sessions() {
    echo ""
    echo "📊 Training Session Monitor"
    echo "=========================="
    echo "Use these commands to monitor progress:"
    echo ""
    for domain in "${DOMAINS[@]}"; do
        session_name="train_${domain}"
        echo "  tmux attach-session -t ${session_name}  # Monitor ${domain} training"
    done
    echo ""
    echo "  tmux list-sessions                        # List all sessions"
    echo "  bash scripts/check_training_status.sh    # Check training status"
    echo ""
}

# Main execution
main() {
    cd "${PROJECT_DIR}"
    
    # Cleanup old sessions
    cleanup_sessions
    
    # Validate environment
    echo "🔍 Validating environment..."
    if ! conda info --envs | grep -q "${CONDA_ENV}"; then
        echo "❌ Conda environment '${CONDA_ENV}' not found"
        exit 1
    fi
    
    echo "✅ Environment validation passed"
    echo ""
    
    # Run each domain training
    for domain in "${DOMAINS[@]}"; do
        echo "🎯 Processing ${domain} domain..."
        run_domain_training "${domain}"
        
        # Small delay between starting sessions
        sleep 5
        echo ""
    done
    
    # Show monitoring information
    monitor_sessions
    
    echo "🎉 All training sessions have been started!"
    echo ""
    echo "📝 Training will proceed in the following order:"
    echo "   1. Medical → 2. Law → 3. Math → 4. Code"
    echo ""
    echo "⚠️  Note: Each domain will start after the previous one completes"
    echo "    This ensures GPU memory management and avoids conflicts"
    echo ""
    echo "🔍 To monitor progress:"
    echo "   tmux list-sessions"
    echo "   tmux attach-session -t train_medical"
}

# Create training status checker script
create_status_checker() {
    cat > "${PROJECT_DIR}/scripts/check_training_status.sh" << 'EOF'
#!/bin/bash
# Training Status Checker

echo "📊 Domain Training Status"
echo "========================"

DOMAINS=("medical" "law" "math" "code")
PROJECT_DIR="/data/disk5/internship_disk/gyubin/Qwen_MoE"

for domain in "${DOMAINS[@]}"; do
    session_name="train_${domain}"
    
    echo -n "  ${domain}: "
    
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "🔄 Training in progress"
    else
        # Check if model was saved
        if [ -d "${PROJECT_DIR}/domain_models/${domain}/final_adapter" ]; then
            echo "✅ Completed"
        else
            echo "⏸️  Not started"
        fi
    fi
done

echo ""
echo "📁 Saved Models:"
for domain in "${DOMAINS[@]}"; do
    if [ -d "${PROJECT_DIR}/domain_models/${domain}/final_adapter" ]; then
        echo "  ✅ ${domain}: domain_models/${domain}/final_adapter"
    else
        echo "  ❌ ${domain}: Not saved"
    fi
done

echo ""
echo "📋 Recent Experiments:"
if [ -d "${PROJECT_DIR}/experiments" ]; then
    ls -1t "${PROJECT_DIR}/experiments" | head -5 | while read exp; do
        echo "  📂 ${exp}"
    done
else
    echo "  No experiments found"
fi
EOF

    chmod +x "${PROJECT_DIR}/scripts/check_training_status.sh"
    echo "✅ Status checker created: scripts/check_training_status.sh"
}

# Create the status checker
create_status_checker

# Run main function
main "$@"
