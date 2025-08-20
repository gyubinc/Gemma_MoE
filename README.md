# Qwen-MoE: Domain-Specific Mixture of Experts

A comprehensive implementation of domain-specific LoRA adapters for Qwen-3-4B-Instruct model with MoE (Mixture of Experts) architecture, optimized for A6000 46GB VRAM.

## 🎯 Project Overview

This project implements a complete MoE (Mixture of Experts) pipeline:

1. **Domain-Specific Training**: Train LoRA adapters for 4 different domains
2. **MoE Architecture**: Combine adapters into a single MoE model with router
3. **Router Training**: Train the router on combined dataset from all domains

### Supported Domains:
- **Medical**: MedMCQA medical question answering
- **Law**: LegalBench case_hold legal case analysis  
- **Math**: GSM8K mathematical problem solving
- **Code**: CodeXGLUE code generation

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Qwen_MoE

# Create conda environment
conda create -n qwen_moe python=3.10
conda activate qwen_moe

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download all datasets
cd scripts
python download_data.py
```

### 3. Environment Validation

```bash
# Test environment
python utils.py

# Test dataset loading
cd tests
python test_dataset.py
```

### 4. Complete Pipeline Training

```bash
# Run complete pipeline (domain training + MoE router training)
cd scripts
bash run_pipeline.sh
```

### 5. Individual Training

```bash
# Train individual domains
python train_domain_models.py --domain medical --gpu_id 0
python train_domain_models.py --domain law --gpu_id 0
python train_domain_models.py --domain math --gpu_id 0
python train_domain_models.py --domain code --gpu_id 0

# Train MoE router (after domain training)
python train_moe_router.py --gpu_id 0
```

## 📁 Project Structure

```
Qwen_MoE/
├── dataset.py              # Dataset loading and processing
├── train_domain_models.py  # Domain-specific LoRA training
├── train_moe_router.py     # MoE router training
├── moe_architecture.py     # MoE model implementation
├── utils.py               # Utility functions
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
├── README.md             # Project documentation
├── scripts/              # Utility scripts
│   ├── run_pipeline.sh   # Complete training pipeline
│   └── download_data.py  # Dataset download script
├── tests/                # Test files
│   └── test_dataset.py   # Dataset loading tests
├── docs/                 # Documentation
│   └── SETUP.md         # Setup guide
├── data/                 # Downloaded datasets
│   ├── medical/         # MedMCQA data
│   ├── law/             # LegalBench data
│   ├── math/            # GSM8K data
│   └── code/            # CodeXGLUE data
├── domain_models/        # Trained domain adapters
└── experiments/          # Training logs and results
```

## ⚙️ Configuration

The `config.yaml` file contains all training parameters optimized for A6000 46GB VRAM:

```yaml
# Domain training settings
training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_epochs: 3

# MoE settings
moe:
  num_experts: 4
  router_type: "top2"
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 1e-4
  num_epochs: 2
```

## 🏗️ MoE Architecture

### 1. **Domain-Specific Experts**
- Each domain (medical, law, math, code) has its own LoRA adapter
- Adapters are frozen during router training
- Expert capacity limits tokens per expert

### 2. **Router Network**
- Neural network that selects appropriate experts
- Top-2 routing: selects 2 experts per token
- Load balancing loss encourages uniform expert usage

### 3. **MoE Forward Pass**
1. Get base model hidden states
2. Router selects experts for each token
3. Apply selected experts to tokens
4. Combine expert outputs
5. Final forward pass with combined states

## 🎯 Supported Domains

### Medical Domain (MedMCQA)
- **Dataset**: MedMCQA medical question answering
- **Format**: Multiple choice questions with explanations
- **Size**: ~182K training samples
- **Task**: Medical knowledge assessment

### Law Domain (LegalBench)
- **Dataset**: LegalBench case_hold legal case analysis
- **Format**: Legal case scenarios with multiple holdings
- **Size**: ~45K training samples  
- **Task**: Legal reasoning and case analysis

### Math Domain (GSM8K)
- **Dataset**: GSM8K mathematical problem solving
- **Format**: Word problems with step-by-step solutions
- **Size**: ~7.5K training samples
- **Task**: Mathematical reasoning and problem solving

### Code Domain (CodeXGLUE)
- **Dataset**: CodeXGLUE text-to-code generation
- **Format**: Natural language to Python code
- **Size**: ~252K training samples
- **Task**: Code generation from natural language

## 🔧 Memory Optimization

The project is specifically optimized for A6000 46GB VRAM:

- **Gradient Checkpointing**: Enabled for memory efficiency
- **Mixed Precision**: FP16 training for reduced memory usage
- **Batch Size**: Optimized batch sizes for domain and MoE training
- **Memory Management**: Automatic GPU memory cleanup between stages
- **Expert Capacity**: Limits tokens per expert to prevent OOM

## 📊 Training Process

### Stage 1: Domain-Specific Training
1. **Environment Validation**: Check GPU, packages, and data availability
2. **Sequential Training**: Train domains one by one to avoid memory conflicts
3. **Memory Cleanup**: Clear GPU memory between domain training
4. **Checkpoint Management**: Save only the best and latest checkpoints

### Stage 2: MoE Router Training
1. **Adapter Loading**: Load all trained domain adapters
2. **Dataset Combination**: Combine datasets from all domains
3. **Router Training**: Train router network on combined dataset
4. **Load Balancing**: Optimize expert utilization
5. **Final Model**: Save complete MoE model

## 🚨 Error Handling

The project includes comprehensive error handling:

- **Environment Validation**: Check all dependencies and hardware
- **Data Availability**: Verify dataset files exist before training
- **Memory Management**: Handle OOM errors gracefully
- **Training Recovery**: Resume from checkpoints if training fails
- **Logging**: Detailed error logs for debugging

## 📈 Monitoring

Training progress can be monitored through:

- **Console Output**: Real-time training metrics
- **Log Files**: Detailed logs in `experiments/` directory
- **GPU Memory**: Memory usage tracking at each stage
- **Expert Utilization**: Router statistics and expert usage
- **Checkpoints**: Automatic model saving and evaluation

## 🎯 Usage Examples

### Complete Pipeline
```bash
# Run entire pipeline
cd scripts
bash run_pipeline.sh
```

### Individual Components
```bash
# Test MoE architecture
python moe_architecture.py

# Train specific domain
python train_domain_models.py --domain medical --gpu_id 0

# Train MoE router
python train_moe_router.py --gpu_id 0
```

### Environment Testing
```bash
# Test utility functions
python utils.py

# Test dataset loading
cd tests
python test_dataset.py
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.yaml`
   - Enable gradient checkpointing
   - Clear GPU memory between stages

2. **Dataset Not Found**
   - Run `python scripts/download_data.py` first
   - Check data directory structure
   - Verify file permissions

3. **Package Import Errors**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.10+ recommended)
   - Verify conda environment activation

### Performance Tips

- Use `gradient_checkpointing: true` for memory efficiency
- Monitor expert utilization during MoE training
- Use FP16 for faster training on A6000
- Clear memory between domain and MoE training

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions and support, please open an issue on the repository.
