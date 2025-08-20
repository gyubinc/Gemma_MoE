# Qwen-MoE Setup Guide

## 🚀 Quick Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n qwen_moe python=3.10
conda activate qwen_moe

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Download

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

## 📁 Project Structure

```
Qwen_MoE/
├── dataset.py              # Dataset loading and processing
├── train_domain_models.py  # Domain-specific LoRA training
├── utils.py               # Utility functions
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
├── README.md             # Project documentation
├── scripts/              # Utility scripts
│   ├── run_pipeline.sh   # Full training pipeline
│   └── download_data.py  # Dataset download script
├── tests/                # Test files
│   └── test_dataset.py   # Dataset loading tests
├── docs/                 # Documentation
│   └── SETUP.md         # This file
├── data/                 # Downloaded datasets
│   ├── medical/         # MedMCQA data
│   ├── law/             # LegalBench data
│   ├── math/            # GSM8K data
│   └── code/            # CodeXGLUE data
├── domain_models/        # Trained domain adapters
└── experiments/          # Training logs and results
```

## 🔧 Configuration

The project uses `config.yaml` for all settings. Key configurations:

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **LoRA**: r=64, alpha=128
- **Training**: batch_size=4, gradient_accumulation=8
- **Memory**: Optimized for A6000 46GB VRAM

## 🎯 Usage

### Single Domain Training

```bash
# Train medical domain
python train_domain_models.py --domain medical --gpu_id 0

# Train law domain
python train_domain_models.py --domain law --gpu_id 0

# Train math domain
python train_domain_models.py --domain math --gpu_id 0

# Train code domain
python train_domain_models.py --domain code --gpu_id 0
```

### Full Pipeline

```bash
# Run complete pipeline
cd scripts
bash run_pipeline.sh
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.yaml`
   - Enable gradient checkpointing
   - Clear GPU memory between domains

2. **Dataset Not Found**
   - Run `python scripts/download_data.py`
   - Check data directory structure

3. **Package Import Errors**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.10+ recommended)

### Performance Tips

- Use gradient checkpointing for memory efficiency
- Monitor GPU memory usage during training
- Use FP16 for faster training on A6000
- Clear memory between domain training

## 📊 Monitoring

- **Console Output**: Real-time training metrics
- **Log Files**: Detailed logs in `experiments/` directory
- **GPU Memory**: Memory usage tracking at each stage
- **Checkpoints**: Automatic model saving and evaluation
