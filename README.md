# Qwen-MoE: Domain-Specific Mixture of Experts

A comprehensive OOP-designed implementation of domain-specific LoRA adapters for Qwen-3-4B-Instruct model with MoE (Mixture of Experts) architecture, optimized for A6000 46GB VRAM.

## 🎯 Project Overview

This project implements a complete MoE (Mixture of Experts) pipeline with object-oriented design:

1. **Domain-Specific Training**: Train LoRA adapters for 4 different domains using OOP classes
2. **Automatic Evaluation**: Each training includes comprehensive model evaluation  
3. **MoE Architecture**: Combine adapters into a single MoE model with router
4. **Sequential Pipeline**: Memory-optimized training to prevent GPU memory issues

### 🏥 Supported Domains:
- **Medical**: MedMCQA medical question answering (15k samples)
- **Law**: LegalBench case_hold legal case analysis (45k samples)
- **Math**: GSM8K mathematical problem solving (7.5k samples)
- **Code**: CodeXGLUE code generation (252k samples)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/gyubinc/Gemma_MoE.git
cd Qwen_MoE

# Activate conda environment
conda deactivate && conda activate gyubin

# Set GPU
export CUDA_VISIBLE_DEVICES=3
```

### 2. One-Command Training (Recommended)

```bash
# Run all domains sequentially (Medical → Law → Math → Code)
bash run_all_domains.sh
```

### 3. Individual Domain Training

```bash
# Train specific domain
python run_domain_training.py --domain medical --gpu_id 3 --experiment_name medical_training
python run_domain_training.py --domain law --gpu_id 3 --experiment_name law_training
python run_domain_training.py --domain math --gpu_id 3 --experiment_name math_training
python run_domain_training.py --domain code --gpu_id 3 --experiment_name code_training
```

### 4. tmux Background Training

```bash
# Run in tmux for long-running training
tmux new-session -d -s all_domains_training
tmux send-keys -t all_domains_training "cd /path/to/Qwen_MoE && conda activate gyubin && bash run_all_domains.sh" Enter

# Monitor progress
tmux attach-session -t all_domains_training
```

## 📁 Project Structure

```
Qwen_MoE/
├── 📋 Main Scripts
│   ├── run_all_domains.sh           # Single script for all domain training
│   ├── run_domain_training.py       # Unified training pipeline
│   ├── domain_trainer.py            # OOP training class
│   └── domain_evaluator.py          # OOP evaluation class
│
├── ⚙️ Core Components  
│   ├── config.yaml                  # Centralized configuration
│   ├── dataset.py                   # Domain data loaders with teacher forcing
│   ├── utils.py                     # Utilities (GPU memory, evaluation, etc.)
│   ├── moe_architecture.py          # MoE model implementation
│   └── train_moe_router.py          # Router training
│
├── 📊 Data & Results
│   ├── data/                        # Domain datasets
│   │   ├── medical/                 # MedMCQA data
│   │   ├── law/                     # LegalBench data  
│   │   ├── math/                    # GSM8K data
│   │   └── code/                    # CodeXGLUE data
│   ├── domain_models/               # Trained LoRA adapters
│   └── experiments/                 # Training logs and evaluation results
│
└── 🛠️ Scripts & Tools
    └── scripts/
        ├── train_medical.sh         # Individual domain scripts
        ├── train_law.sh
        ├── train_math.sh
        ├── train_code.sh
        └── check_training_status.sh # Training status checker
```

## 🔧 Configuration

All training parameters are managed in `config.yaml`:

```yaml
# Model configuration
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"
  torch_dtype: "float16"

# LoRA configuration  
lora:
  r: 8                               # Low rank for fast training
  alpha: 16
  target_modules: ["gate_proj", "up_proj", "down_proj"]

# Training configuration
training:
  per_device_batch_size: 32          # Optimized for A6000
  gradient_accumulation_steps: 1
  learning_rate: 5e-4
  num_epochs: 1
  max_length: 256                    # Optimized sequence length

# Domain-specific settings
domain_configs:
  medical:
    max_samples: 15000               # Limited for faster training
    eval_samples: 1000
  law:
    max_samples: null                # Use full dataset
    eval_samples: 1000
```

## 🎯 OOP Design

### DomainTrainer Class
```python
from domain_trainer import DomainTrainer

trainer = DomainTrainer(config, domain="medical", experiment_dir="experiments/")
adapter_path = trainer.train()
```

### DomainEvaluator Class  
```python
from domain_evaluator import DomainEvaluator

evaluator = DomainEvaluator(config, domain="medical", experiment_dir="experiments/")
results = evaluator.evaluate(adapter_path)
```

### Unified Pipeline
```python
from run_domain_training import DomainPipeline

pipeline = DomainPipeline("config.yaml", "medical", "medical_exp", gpu_id=3)
results = pipeline.run_full_pipeline()
```

## 📊 Training Pipeline

### Automatic Train → Evaluate Flow

1. **Setup**: Load model, tokenizer, LoRA configuration
2. **Training**: Train domain-specific LoRA adapter with teacher forcing
3. **Saving**: Save trained adapter to `domain_models/{domain}/final_adapter/`
4. **Evaluation**: Automatic evaluation on test dataset
5. **Results**: Save evaluation results to `experiments/{experiment_name}/`

### Memory Management

- **Sequential Training**: One domain at a time to prevent memory issues
- **GPU Memory Monitoring**: Automatic memory tracking and cleanup
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Mixed Precision**: FP16 training for optimal memory usage

## 📈 Monitoring & Results

### Check Training Status
```bash
# Check current training progress
bash scripts/check_training_status.sh

# View tmux session
tmux list-sessions
tmux attach-session -t all_domains_training
```

### Results Structure
```
experiments/medical_training_YYYYMMDD_HHMMSS/
├── config.yaml                     # Training configuration
├── medical_pipeline.log            # Training logs
└── medical_evaluation_results.json # Evaluation results

domain_models/medical/final_adapter/
├── adapter_config.json             # LoRA configuration
├── adapter_model.safetensors        # Trained weights
└── tokenizer files...               # Tokenizer files
```

### Evaluation Metrics

Each domain evaluation includes:
- **Accuracy**: Overall prediction accuracy
- **Sample Predictions**: First 10 predictions for inspection
- **Detailed Results**: Per-sample comparison with ground truth

## 🔄 MoE Router Training

After all domains are trained:

```bash
# Train MoE router
python train_moe_router.py --gpu_id 3 --experiment_name moe_router_training
```

## ⚡ Performance Optimizations

- **Optimized Batch Size**: 32 for A6000 46GB VRAM
- **Low-Rank LoRA**: r=8, alpha=16 for fast training
- **Short Sequences**: max_length=256 based on data analysis
- **Single Epoch**: Sufficient for domain adaptation
- **Teacher Forcing**: Only train on assistant responses

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce `per_device_batch_size` in config.yaml
2. **conda activate Error**: Run `conda init bash && source ~/.bashrc`
3. **Data Not Found**: Check data files in `data/{domain}/` directories

### Memory Monitoring
```bash
# Check GPU memory
nvidia-smi

# Monitor in training logs
grep "GPU" experiments/*/logs
```

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

**🎯 Ready for production: Sequential domain training with automatic evaluation and MoE router integration!**