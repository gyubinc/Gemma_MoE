# Qwen-MoE v2.0: Domain-Specific Mixture of Experts

Qwen-3-4B-Instruct 모델 기반 도메인별 LoRA 어댑터 훈련 및 MoE 아키텍처 구현 프로젝트

## 🎯 프로젝트 개요

Qwen-3-4B-Instruct 모델 기반 4개 도메인(Medical, Law, Math, Code)에 대한 LoRA 어댑터를 훈련하고, 이를 MoE(Mixture of Experts) 아키텍처로 통합하는 파이프라인

### 🎯 주요 특징
- **3개 핵심 파일**: 간단하고 효율적인 구조
- **도메인별 최적화**: 각 도메인에 특화된 설정
- **메모리 효율성**: LoRA를 통한 파라미터 효율적 훈련
- **확장 가능성**: 새로운 도메인 쉽게 추가 가능

## 📁 핵심 파일 구조

```
Qwen_MoE/
├── train_domain_lora.py      # 1️⃣ 도메인별 LoRA 훈련
├── evaluate_model.py         # 2️⃣ 모델 평가
├── build_moe_model.py        # 3️⃣ MoE 모델 생성
├── src/                      # 핵심 소스 코드
├── docs/                     # 문서
├── data/                     # 데이터
├── domain_models/            # 훈련된 도메인 모델
├── moe_model/                # 생성된 MoE 모델
└── internal/                 # 내부 파일들 (숨김)
```

## 🚀 사용법

### 1️⃣ 도메인별 LoRA 훈련

```bash
# Medical 도메인 훈련
python train_domain_lora.py --domain medical --max-samples 1000

# Law 도메인 훈련
python train_domain_lora.py --domain law --max-samples 1000

# Math 도메인 훈련
python train_domain_lora.py --domain math --max-samples 1000

# Code 도메인 훈련
python train_domain_lora.py --domain code --max-samples 1000
```

**옵션:**
- `--domain`: 훈련할 도메인 (medical, law, math, code)
- `--max-samples`: 최대 샘플 수 (기본값: 전체)
- `--output-dir`: 출력 디렉토리 (기본값: domain_models)
- `--epochs`: 훈련 에포크 수 (기본값: 3)
- `--batch-size`: 배치 크기 (기본값: 4)
- `--learning-rate`: 학습률 (기본값: 2e-4)

### 2️⃣ 모델 평가

```bash
# 베이스 모델 평가
python evaluate_model.py --model-type base

# LoRA 어댑터 평가
python evaluate_model.py --model-type lora --model-path domain_models/medical/final_adapter

# MoE 모델 평가
python evaluate_model.py --model-type moe --model-path moe_model

# 특정 도메인만 평가
python evaluate_model.py --model-type base --domain medical

# 빠른 평가 (100개 샘플)
python evaluate_model.py --model-type base --max-samples 100
```

**옵션:**
- `--model-type`: 모델 타입 (base, lora, moe)
- `--model-path`: 모델/어댑터 경로 (lora/moe 타입에 필요)
- `--domain`: 평가할 도메인 (기본값: 모든 도메인)
- `--max-samples`: 최대 샘플 수 (기본값: 1000)
- `--output`: 결과 파일 (기본값: evaluation_results.json)

### 3️⃣ MoE 모델 생성

```bash
# MoE 모델 생성
python build_moe_model.py --adapters-dir domain_models --output-dir moe_model

# MoE 모델 검증만
python build_moe_model.py --output-dir moe_model --validate-only
```

**옵션:**
- `--adapters-dir`: 어댑터 디렉토리 (기본값: domain_models)
- `--output-dir`: 출력 디렉토리 (기본값: moe_model)
- `--base-model`: 베이스 모델 (기본값: Qwen/Qwen3-4B-Instruct-2507)
- `--validate-only`: 기존 모델 검증만 수행

## 📊 지원하는 도메인

| 도메인 | 데이터셋 | 평가 유형 | 선택지 수 | Train | Test |
|--------|----------|-----------|-----------|-------|------|
| **Medical** | MedMCQA | Multiple Choice | 4개 | 182,822 | 6,150 |
| **Law** | CaseHOLD | Multiple Choice | 5개 | 45,000 | 3,600 |
| **Math** | MathQA | Multiple Choice | 5개 | 29,837 | 2,985 |
| **Code** | CyberMetric | Exact Match | - | 준비 중 | 준비 중 |

### 📋 데이터셋 상세 정보

#### 🏥 Medical (MedMCQA)
- **데이터셋**: MedMCQA (Medical Multiple Choice Question Answering)
- **형식**: 의학 지식 기반 객관식 문제
- **선택지**: 4개 (A, B, C, D)
- **평가**: 정확한 선택지 선택

#### ⚖️ Law (CaseHOLD)
- **데이터셋**: CaseHOLD (Case Holdings On Legal Decisions)
- **형식**: 법률 판례 분석 및 적절한 판결 선택
- **선택지**: 5개 (A, B, C, D, E)
- **평가**: 올바른 법률 판결 선택

#### 🔢 Math (MathQA)
- **데이터셋**: MathQA (Mathematical Question Answering)
- **형식**: 수학 문제 해결
- **선택지**: 5개 (A, B, C, D, E)
- **평가**: 정확한 수치 답안 선택

#### 💻 Code (CyberMetric)
- **데이터셋**: CyberMetric (코딩 문제 데이터셋)
- **형식**: 프로그래밍 문제 해결
- **평가**: 정확한 코드 생성

## 🔄 전체 워크플로우

### Step 1: 도메인별 훈련
```bash
# 각 도메인을 순차적으로 훈련 (전체 데이터 사용)
python train_domain_lora.py --domain medical
python train_domain_lora.py --domain law
python train_domain_lora.py --domain math

# 또는 빠른 테스트용 (1000 샘플)
python train_domain_lora.py --domain medical --max-samples 1000
python train_domain_lora.py --domain law --max-samples 1000
python train_domain_lora.py --domain math --max-samples 1000
```

### Step 2: 베이스 모델 평가
```bash
# 베이스 모델 성능 확인
python evaluate_model.py --model-type base
```

### Step 3: LoRA 어댑터 평가
```bash
# 각 도메인 어댑터 성능 확인
python evaluate_model.py --model-type lora --model-path domain_models/medical/final_adapter
python evaluate_model.py --model-type lora --model-path domain_models/law/final_adapter
python evaluate_model.py --model-type lora --model-path domain_models/math/final_adapter
```

### Step 4: MoE 모델 생성
```bash
# 모든 어댑터를 합쳐서 MoE 모델 생성
python build_moe_model.py --adapters-dir domain_models --output-dir moe_model
```

### Step 5: MoE 모델 평가
```bash
# MoE 모델 성능 확인
python evaluate_model.py --model-type moe --model-path moe_model
```

## ⚡ 성능 최적화

- **A6000 최적화**: 46GB VRAM에 최적화된 설정
- **LoRA 설정**: r=64, alpha=128, target_modules=["gate_proj", "up_proj", "down_proj"]
- **메모리 관리**: 자동 GPU 메모리 정리
- **순차 훈련**: 도메인 간 메모리 충돌 방지

## 📈 성능 결과

### 도메인별 성능 비교

| 도메인 | Before Training | After Training | MoE Model |
|--------|----------------|----------------|-----------|
| **Base** | TBD | - | - |
| **Medical** | TBD | TBD | TBD |
| **Law** | TBD | TBD | TBD |
| **Math** | TBD | TBD | TBD |
| **Code** | TBD | TBD | TBD |

### 예상 성능 (Baseline)

- **Medical (MedMCQA)**: ~75% accuracy (baseline)
- **Law (CaseHOLD)**: ~30% accuracy (baseline)
- **Math (MathQA)**: ~25% accuracy (baseline)
- **훈련 시간**: ~2-4시간/도메인 (1000 샘플)
- **메모리 사용량**: ~20-30GB VRAM/도메인

### 📊 데이터셋 통계

| 도메인 | Train Size | Test Size | Total |
|--------|------------|-----------|-------|
| **Medical** | 182,822 | 6,150 | 188,972 |
| **Law** | 45,000 | 3,600 | 48,600 |
| **Math** | 29,837 | 2,985 | 32,822 |
| **Total** | 257,659 | 12,735 | 270,394 |

## 🛠️ 환경 설정

```bash
# Conda 환경 활성화
conda deactivate && conda activate gyubin

# GPU 설정 (3번 GPU 사용)
export CUDA_VISIBLE_DEVICES=3

# 프로젝트 디렉토리 이동
cd /data/disk5/internship_disk/gyubin/Qwen_MoE

# 의존성 설치 (필요시)
pip install -r requirements.txt
```

## 📝 결과 구조

```
domain_models/
├── medical/
│   ├── final_adapter/           # 훈련된 LoRA 어댑터
│   └── medical_training_summary.json
├── law/
├── math/
└── code/

moe_model/
├── moe_config.json             # MoE 설정
├── domain_mapping.json         # 도메인 매핑
├── router_config.json          # 라우터 설정
├── model_info.json             # 모델 정보
└── build_summary.json          # 빌드 요약
```

## 🚨 문제 해결

### 일반적인 문제들
1. **CUDA 메모리 오류**: `--max-samples` 줄이기
2. **데이터 없음**: 데이터 디렉토리 확인
3. **Import 오류**: `conda activate gyubin` 확인

### 메모리 모니터링
```bash
# GPU 메모리 확인
nvidia-smi

# 실시간 모니터링
watch -n 1 nvidia-smi
```

## 📚 추가 문서

- **설치 가이드**: [docs/SETUP.md](docs/SETUP.md)
- **API 문서**: [docs/API.md](docs/API.md)
- **사용 예제**: [docs/EXAMPLES.md](docs/EXAMPLES.md)

---

**🎯 핵심 파일 3개로 간단하고 효율적인 Qwen-MoE 파이프라인!**
