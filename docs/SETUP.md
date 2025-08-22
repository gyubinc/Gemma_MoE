# Qwen-MoE v2.0 Setup Guide

## 🚀 Quick Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n gyubin python=3.10
conda activate gyubin

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Download

```bash
# Download all datasets
python download_mathqa.py  # Math dataset
# Other datasets should be manually downloaded to data/ directory
```

### 3. Environment Validation

```bash
# Test environment
python scripts/analyze.py --check-only

# Test dataset loading
python scripts/analyze.py --stats-only
```

## 📁 Project Structure (v2.0)

```
Qwen_MoE/
├── src/                          # 핵심 소스 코드
│   ├── core/                     # 핵심 기능
│   │   ├── trainer.py            # 통합 훈련기 (도메인별 설정)
│   │   ├── evaluator.py          # 통합 평가기 (도메인별 설정)
│   │   ├── dataset.py            # 통합 데이터셋 (도메인별 설정)
│   │   └── model.py              # 모델 관리 (베이스/어댑터)
│   ├── configs/                  # 설정 관리
│   │   └── domains.py            # 도메인별 설정 중앙 관리
│   └── utils/                    # 유틸리티
│       ├── memory.py             # GPU 메모리 관리
│       └── data_utils.py         # 데이터 유틸리티
├── scripts/                      # 실행 스크립트
│   ├── train.py                  # 훈련 실행
│   ├── evaluate.py               # 평가 실행
│   ├── analyze.py                # 데이터 분석
│   └── run_all.py                # 전체 도메인 순차 훈련
├── configs/                      # 설정 파일
├── data/                         # 데이터
├── experiments/                  # 실험 결과
└── docs/                         # 문서
    ├── SETUP.md                  # 이 파일
    ├── API.md                    # API 문서
    └── EXAMPLES.md               # 사용 예제
```

## 🔧 Configuration

The project uses centralized configuration in `src/configs/domains.py`. Key configurations:

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **LoRA**: r=64, alpha=128, target_modules=["gate_proj", "up_proj", "down_proj"]
- **Training**: batch_size=4, gradient_accumulation=8, epochs=3
- **Memory**: Optimized for A6000 46GB VRAM

## 🎯 Usage

### Data Analysis

```bash
# 데이터 가용성 확인
python scripts/analyze.py --check-only

# 데이터 샘플 분석
python scripts/analyze.py --samples-only

# 도메인별 통계 확인
python scripts/analyze.py --stats-only

# 전체 분석 리포트 생성
python scripts/analyze.py
```

### Single Domain Training

```bash
# Medical 도메인 훈련
python scripts/train.py --domain medical --max-samples 1000

# Law 도메인 훈련
python scripts/train.py --domain law --max-samples 1000

# Math 도메인 훈련
python scripts/train.py --domain math --max-samples 1000
```

### Full Pipeline

```bash
# 모든 도메인 순차 훈련
python scripts/run_all.py --max-samples 1000

# 특정 도메인만 훈련
python scripts/run_all.py --domains medical law --max-samples 1000

# 기존 모델 건너뛰기
python scripts/run_all.py --skip-existing
```

### Evaluation

```bash
# 베이스 모델 평가
python scripts/evaluate.py

# 특정 도메인 평가
python scripts/evaluate.py --domain medical

# LoRA 어댑터와 함께 평가
python scripts/evaluate.py --adapter-path domain_models/medical/final_adapter

# 100개 샘플로 빠른 평가
python scripts/evaluate.py --max-samples 100
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce max_samples: `--max-samples 500`
   - Clear GPU memory between domains (automatic)
   - Check GPU memory: `python -c "from src.utils import print_gpu_memory_summary; print_gpu_memory_summary()"`

2. **Dataset Not Found**
   - Run `python scripts/analyze.py --check-only`
   - Check data directory structure
   - Download missing datasets

3. **Package Import Errors**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.10+ recommended)
   - Activate conda environment: `conda activate gyubin`

4. **Module Import Errors**
   - Ensure you're in the project root directory
   - Check Python path setup in scripts

### Performance Tips

- Use gradient checkpointing (enabled by default)
- Monitor GPU memory usage during training
- Use FP16 for faster training on A6000
- Clear memory between domain training (automatic)

## 📊 Monitoring

- **Console Output**: Real-time training metrics
- **Log Files**: Detailed logs in training.log
- **GPU Memory**: Memory usage tracking at each stage
- **Checkpoints**: Automatic model saving and evaluation
- **Results**: JSON files in domain_models/ and experiments/

## 🔧 Advanced Usage

### Custom Domain Configuration

```python
# src/configs/domains.py에 새 도메인 추가
"custom": DomainConfig(
    name="custom",
    data_path="data/custom",
    train_file="custom_train.json",
    validation_file="custom_validation.json",
    test_file="custom_test.json",
    instruction_template="Custom instruction: {question}\n\nAnswer:",
    response_format="text",
    evaluation_type="exact_match",
    max_length=512
)
```

### Training Configuration

```python
from src.core import UnifiedTrainer

trainer = UnifiedTrainer('medical')
trainer.setup_training(max_samples=5000)  # 더 많은 샘플 사용

# LoRA 설정 변경
trainer._setup_lora()  # 내부에서 LoRA 설정 수정 가능
```

## 📈 Expected Performance

- **Medical (MedMCQA)**: ~75% accuracy (baseline)
- **Law (CaseHOLD)**: ~30% accuracy (baseline)
- **Math (GSM8K)**: ~25% accuracy (baseline)
- **Training Time**: ~2-4 hours per domain (1000 samples)
- **Memory Usage**: ~20-30GB VRAM per domain
