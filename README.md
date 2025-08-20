# Gemma LoRA-based MoE

## 🎯 목표
Gemma-3-4B 모델을 기반으로 한 **LoRA 기반 Mixture of Experts (MoE)** 시스템 구축

## 🏗️ 아키텍처

### MoE 구조
각 Transformer 레이어마다:
- **Expert 0**: Original MLP (freeze)
- **Expert 1**: Original MLP + Medical LoRA
- **Expert 2**: Original MLP + Law LoRA  
- **Expert 3**: Original MLP + Math LoRA
- **Expert 4**: Original MLP + Code LoRA
- **Router**: Top-1 routing (학습 대상)

### 장점
- **메모리 효율성**: 전체 파라미터의 0.48%만 훈련
- **확장성**: 새로운 도메인 추가 용이
- **성능**: 도메인별 전문화된 응답

## 📁 프로젝트 구조

```
gemma-moe/
├── train_domain_models.py    # 도메인별 LoRA 훈련
├── train_moe_router.py       # MoE 라우터 훈련  
├── moe_architecture.py       # MoE 아키텍처 구현
├── dataset.py               # 데이터셋 로더
├── utils.py                 # 유틸리티 함수들
├── run_full_moe_pipeline.sh # 전체 파이프라인
├── requirements.txt         # 의존성
├── data/                    # 훈련 데이터
│   ├── medical/
│   ├── law/
│   ├── math/
│   └── code/
└── domain_models/           # 훈련된 LoRA 어댑터들
    ├── medical/
    ├── law/
    ├── math/
    └── code/
```

## 🚀 사용법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 다운로드
```bash
python download_data.py
```

### 3. 전체 파이프라인 실행
```bash
./run_full_moe_pipeline.sh
```

### 4. 개별 도메인 훈련
```bash
python train_domain_models.py \
    --domain medical \
    --model google/gemma-3-4b-it \
    --epochs 3 \
    --batch_size 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --fp16
```

### 5. MoE 라우터 훈련
```bash
python train_moe_router.py \
    --base_model google/gemma-3-4b-it \
    --domain_adapter_paths ./domain_models/medical/final_adapter ./domain_models/law/final_adapter ./domain_models/math/final_adapter ./domain_models/code/final_adapter \
    --domains medical law math code
```

## ⚙️ 주요 파라미터

### LoRA 설정
- **lora_r**: 16 (LoRA rank)
- **lora_alpha**: 32 (LoRA scaling factor)
- **lora_dropout**: 0.1
- **target_modules**: gate_proj, up_proj, down_proj

### 훈련 설정
- **batch_size**: 4
- **learning_rate**: 2e-4
- **max_length**: 1024
- **fp16**: True

## 📊 메모리 요구사항
- **GPU**: NVIDIA RTX A6000 (48GB) 권장
- **모델 로딩**: ~8GB
- **LoRA 훈련**: ~12GB 총 사용
- **훈련 파라미터**: 20.9M / 4.3B (0.48%)

## 🎯 도메인
1. **Medical**: 의료 QA (MedMCQA)
2. **Law**: 법률 QA (LegalBench)
3. **Math**: 수학 문제 (GSM8K)
4. **Code**: 코딩 문제 (CodeXGLUE)

## 🔍 특징
- **Parameter Efficient**: LoRA 기반으로 메모리 효율적
- **Modular Design**: 도메인별 독립적 훈련
- **Top-1 Routing**: 빠르고 효율적인 전문가 선택
- **Load Balancing**: 전문가 간 균등한 사용률 유도