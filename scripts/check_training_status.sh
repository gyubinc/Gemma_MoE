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
