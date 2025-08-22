# Qwen-MoE v2.0 Usage Examples

## 🚀 Quick Start Examples

### 1. 데이터 분석

```bash
# 데이터 가용성 확인
python scripts/analyze.py --check-only

# Medical 도메인 샘플 분석
python scripts/analyze.py --samples-only --domain medical

# Medical 도메인 통계 확인
python scripts/analyze.py --stats-only --domain medical

# 전체 분석 리포트 생성
python scripts/analyze.py --output analysis_report.json
```

### 2. 단일 도메인 훈련

```bash
# Medical 도메인 훈련 (1000개 샘플)
python scripts/train.py --domain medical --max-samples 1000

# Law 도메인 훈련 (500개 샘플)
python scripts/train.py --domain law --max-samples 500

# Math 도메인 훈련 (전체 데이터)
python scripts/train.py --domain math
```

### 3. 전체 도메인 순차 훈련

```bash
# 모든 도메인 순차 훈련
python scripts/run_all.py --max-samples 1000

# Medical과 Law만 훈련
python scripts/run_all.py --domains medical law --max-samples 1000

# 기존 모델 건너뛰기
python scripts/run_all.py --skip-existing
```

### 4. 평가

```bash
# 베이스 모델 평가
python scripts/evaluate.py

# Medical 도메인만 평가
python scripts/evaluate.py --domain medical

# LoRA 어댑터와 함께 평가
python scripts/evaluate.py --adapter-path domain_models/medical/final_adapter

# 빠른 평가 (100개 샘플)
python scripts/evaluate.py --max-samples 100
```

## 🔧 Python API Examples

### 1. 도메인 관리

```python
from src.configs import domain_manager

# 사용 가능한 도메인 확인
domains = domain_manager.get_available_domains()
print(f"Available domains: {domains}")

# Medical 도메인 설정 가져오기
medical_config = domain_manager.get_domain('medical')
print(f"Medical max_length: {medical_config.max_length}")

# 데이터 가용성 확인
availability = domain_manager.check_data_availability()
for domain, available in availability.items():
    print(f"{domain}: {'✅' if available else '❌'}")

# 프롬프트 포맷팅
prompt = domain_manager.format_prompt(
    'medical',
    question="What is the main symptom of diabetes?",
    options=["High blood pressure", "High blood sugar", "Low blood sugar", "None"]
)
print(prompt)
```

### 2. 데이터셋 생성

```python
from transformers import AutoTokenizer
from src.core import UnifiedDataset

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
tokenizer.pad_token = tokenizer.eos_token

# Medical 도메인 데이터셋 생성
medical_dataset = UnifiedDataset(
    tokenizer=tokenizer,
    domain='medical',
    split='train',
    max_samples=100,
    max_length=512
)

print(f"Dataset size: {len(medical_dataset)}")

# 샘플 확인
sample = medical_dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Input shape: {sample['input_ids'].shape}")
```

### 3. 훈련

```python
from src.core import UnifiedTrainer

# 훈련기 생성
trainer = UnifiedTrainer(
    domain='medical',
    output_dir='domain_models',
    device='cuda:0'
)

# 훈련 설정
trainer.setup_training(max_samples=1000)

# 훈련 실행
results = trainer.train()
print(f"Training completed: {results}")

# 정리
trainer.cleanup()
```

### 4. 평가

```python
from src.core import UnifiedEvaluator

# 평가기 생성
evaluator = UnifiedEvaluator(device='cuda:0')

# 베이스 모델 로딩
evaluator.load_model()

# Medical 도메인 평가
results = evaluator.evaluate_domain('medical', max_samples=100)
print(f"Medical accuracy: {results['accuracy']:.4f}")

# 정리
evaluator.cleanup()
```

### 5. 모델 관리

```python
from src.core import model_manager

# 베이스 모델 로딩
model, tokenizer = model_manager.load_base_model()

# 텍스트 생성
prompt = "Answer the following medical question:\n\nWhat is diabetes?\n\nA. A heart disease\nB. A blood sugar disorder\nC. A lung disease\nD. A kidney disease\n\nAnswer:"
response = model_manager.generate(prompt, max_new_tokens=20)
print(f"Response: {response}")

# 메모리 정리
model_manager.clear_memory()
```

## 📊 Advanced Examples

### 1. 커스텀 도메인 추가

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

# 사용
from src.configs import domain_manager
custom_config = domain_manager.get_domain('custom')
```

### 2. 훈련 설정 커스터마이징

```python
from src.core import UnifiedTrainer

trainer = UnifiedTrainer('medical')

# 훈련 설정
trainer.setup_training(max_samples=5000)

# LoRA 설정 변경 (내부에서)
trainer._setup_lora()  # LoRA 설정 수정 가능

# 훈련 실행
results = trainer.train()
```

### 3. 배치 처리

```python
from src.core import train_domain, evaluate_domain
import concurrent.futures

# 여러 도메인 병렬 처리
domains = ['medical', 'law']
results = {}

def train_and_evaluate(domain):
    # 훈련
    train_results = train_domain(domain, max_samples=1000)
    
    # 평가
    eval_results = evaluate_domain(domain, max_samples=100)
    
    return {
        'domain': domain,
        'train': train_results,
        'evaluate': eval_results
    }

# 병렬 실행
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(train_and_evaluate, domain): domain for domain in domains}
    
    for future in concurrent.futures.as_completed(futures):
        domain = futures[future]
        try:
            results[domain] = future.result()
        except Exception as e:
            print(f"Error processing {domain}: {e}")
```

### 4. 메모리 모니터링

```python
from src.utils import print_gpu_memory_summary, clear_gpu_memory

# 훈련 전 메모리 확인
print_gpu_memory_summary("Before training")

# 훈련 실행
trainer = UnifiedTrainer('medical')
trainer.setup_training(max_samples=1000)
results = trainer.train()

# 훈련 후 메모리 확인
print_gpu_memory_summary("After training")

# 메모리 정리
clear_gpu_memory()
trainer.cleanup()
```

### 5. 결과 분석

```python
import json
import matplotlib.pyplot as plt

# 결과 로딩
with open('domain_models/training_summary.json', 'r') as f:
    summary = json.load(f)

# 도메인별 정확도 시각화
domains = []
accuracies = []

for domain, result in summary['results'].items():
    if result['status'] == 'completed':
        domains.append(domain)
        accuracies.append(result['results']['accuracy'])

plt.figure(figsize=(10, 6))
plt.bar(domains, accuracies)
plt.title('Domain-wise Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
```

## 🚨 Error Handling Examples

### 1. 데이터 없음 처리

```python
from src.utils import check_data_availability

# 데이터 가용성 확인
availability = check_data_availability(['medical', 'law', 'math'])

missing_domains = [domain for domain, available in availability.items() if not available]

if missing_domains:
    print(f"Missing data for domains: {missing_domains}")
    print("Please download the required datasets first.")
    exit(1)
```

### 2. 메모리 부족 처리

```python
from src.core import train_domain

try:
    results = train_domain('medical', max_samples=1000)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU memory insufficient. Trying with fewer samples...")
        results = train_domain('medical', max_samples=500)
    else:
        raise e
```

### 3. 모델 로딩 실패 처리

```python
from src.core import model_manager

try:
    model, tokenizer = model_manager.load_base_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Please check your internet connection and model path.")
    exit(1)
```

## 📈 Performance Optimization Examples

### 1. 배치 크기 최적화

```python
from src.utils import get_optimal_batch_size

# GPU 메모리에 따른 최적 배치 크기 계산
batch_sizes = get_optimal_batch_size(model_size_gb=4.0, gpu_memory_gb=46.0)
print(f"Optimal batch sizes: {batch_sizes}")
```

### 2. 메모리 효율적 훈련

```python
from src.core import UnifiedTrainer
from src.utils import clear_gpu_memory

# 도메인별 순차 훈련 (메모리 효율적)
domains = ['medical', 'law', 'math']

for domain in domains:
    print(f"Training {domain} domain...")
    
    trainer = UnifiedTrainer(domain)
    trainer.setup_training(max_samples=1000)
    results = trainer.train()
    trainer.cleanup()
    
    # 메모리 정리
    clear_gpu_memory()
    
    print(f"{domain} training completed: {results['train_loss']:.4f}")
```

### 3. 로깅 설정

```python
import logging

# 상세한 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detailed_training.log'),
        logging.StreamHandler()
    ]
)

# 훈련 실행
from src.core import train_domain
results = train_domain('medical', max_samples=1000)
```

이러한 예제들을 참고하여 Qwen-MoE v2.0을 효과적으로 사용하세요!
