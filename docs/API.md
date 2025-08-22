# Qwen-MoE v2.0 API Documentation

## 📚 Core Components

### DomainManager

도메인별 설정을 중앙에서 관리하는 클래스입니다.

```python
from src.configs import domain_manager

# 사용 가능한 도메인 확인
domains = domain_manager.get_available_domains()
# ['medical', 'law', 'math', 'code']

# 도메인 설정 가져오기
config = domain_manager.get_domain('medical')

# 데이터 가용성 확인
availability = domain_manager.check_data_availability('medical')

# 프롬프트 포맷팅
prompt = domain_manager.format_prompt('medical', question="...", options=["A", "B", "C", "D"])

# 응답 포맷팅
response = domain_manager.format_response('medical', correct_option=0)
```

#### Methods

- `get_domain(domain_name: str) -> DomainConfig`: 도메인 설정 반환
- `get_available_domains() -> List[str]`: 사용 가능한 도메인 목록
- `check_data_availability(domain_name: str = None) -> Dict[str, bool]`: 데이터 가용성 확인
- `get_domain_stats(domain_name: str) -> Dict[str, Any]`: 도메인 통계 정보
- `format_prompt(domain_name: str, **kwargs) -> str`: 프롬프트 포맷팅
- `format_response(domain_name: str, **kwargs) -> str`: 응답 포맷팅

### UnifiedDataset

모든 도메인을 지원하는 통합 데이터셋 클래스입니다.

```python
from src.core import UnifiedDataset

# 데이터셋 생성
dataset = UnifiedDataset(
    tokenizer=tokenizer,
    domain='medical',
    split='train',
    max_samples=1000,
    max_length=512
)

# 데이터셋 크기
print(len(dataset))

# 샘플 가져오기
sample = dataset[0]
# {'input_ids': tensor(...), 'attention_mask': tensor(...), 'labels': tensor(...)}
```

#### Parameters

- `tokenizer`: HuggingFace 토크나이저
- `domain`: 도메인 이름 ('medical', 'law', 'math', 'code')
- `split`: 데이터 분할 ('train', 'validation', 'test')
- `max_samples`: 최대 샘플 수 (None이면 전체 사용)
- `max_length`: 최대 시퀀스 길이

### UnifiedTrainer

모든 도메인을 지원하는 통합 훈련기 클래스입니다.

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

# 정리
trainer.cleanup()
```

#### Methods

- `setup_training(max_samples: int = None)`: 훈련 설정
- `train() -> Dict[str, Any]`: 훈련 실행
- `evaluate(adapter_path: str = None) -> Dict[str, Any]`: 평가 실행
- `cleanup()`: 리소스 정리

### UnifiedEvaluator

모든 도메인을 지원하는 통합 평가기 클래스입니다.

```python
from src.core import UnifiedEvaluator

# 평가기 생성
evaluator = UnifiedEvaluator(device='cuda:0')

# 모델 로딩
evaluator.load_model(adapter_path='domain_models/medical/final_adapter')

# 단일 도메인 평가
results = evaluator.evaluate_domain('medical', max_samples=1000)

# 모든 도메인 평가
results = evaluator.evaluate_all_domains()

# 정리
evaluator.cleanup()
```

#### Methods

- `load_model(adapter_path: str = None)`: 모델 로딩
- `evaluate_domain(domain: str, max_samples: int = 1000, split: str = 'test') -> Dict[str, Any]`: 단일 도메인 평가
- `evaluate_all_domains(domains: List[str] = None, max_samples: int = 1000) -> Dict[str, Any]`: 모든 도메인 평가
- `cleanup()`: 리소스 정리

### ModelManager

모델 로딩과 관리를 담당하는 클래스입니다.

```python
from src.core import model_manager

# 베이스 모델 로딩
model, tokenizer = model_manager.load_base_model()

# 어댑터와 함께 모델 로딩
model, tokenizer = model_manager.load_model_with_adapter('path/to/adapter')

# 텍스트 생성
text = model_manager.generate("Hello, world!", max_new_tokens=50)

# 어댑터 저장
model_manager.save_adapter('path/to/save')

# 메모리 정리
model_manager.clear_memory()
```

#### Methods

- `load_base_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]`: 베이스 모델 로딩
- `load_model_with_adapter(adapter_path: str) -> Tuple[PeftModel, AutoTokenizer]`: 어댑터와 함께 모델 로딩
- `generate(prompt: str, max_new_tokens: int = 50, **kwargs) -> str`: 텍스트 생성
- `save_adapter(adapter_path: str)`: 어댑터 저장
- `clear_memory()`: 메모리 정리

## 🔧 Utility Functions

### Memory Management

```python
from src.utils import (
    print_gpu_memory_summary,
    print_system_memory_summary,
    clear_gpu_memory,
    get_optimal_batch_size
)

# GPU 메모리 요약 출력
print_gpu_memory_summary("After model loading")

# 시스템 메모리 요약 출력
print_system_memory_summary()

# GPU 메모리 정리
clear_gpu_memory()

# 최적 배치 크기 계산
batch_sizes = get_optimal_batch_size(model_size_gb=4.0, gpu_memory_gb=46.0)
```

### Data Utilities

```python
from src.utils import (
    check_data_availability,
    analyze_dataset_samples,
    get_domain_statistics,
    validate_environment
)

# 데이터 가용성 확인
availability = check_data_availability(['medical', 'law'])

# 데이터셋 샘플 분석
analysis = analyze_dataset_samples(['medical'], max_samples=3)

# 도메인 통계 가져오기
stats = get_domain_statistics('medical')

# 환경 검증
is_valid = validate_environment()
```

## 🚀 Convenience Functions

### Training

```python
from src.core import train_domain

# 도메인 훈련 (편의 함수)
results = train_domain(
    domain='medical',
    max_samples=1000,
    output_dir='domain_models'
)
```

### Evaluation

```python
from src.core import evaluate_domain, evaluate_all_domains

# 단일 도메인 평가 (편의 함수)
results = evaluate_domain(
    domain='medical',
    adapter_path='domain_models/medical/final_adapter',
    max_samples=1000
)

# 모든 도메인 평가 (편의 함수)
results = evaluate_all_domains(
    adapter_path=None,  # 베이스 모델 사용
    max_samples=1000
)
```

### Dataset Creation

```python
from src.core import create_datasets

# 도메인별 데이터셋 생성
datasets = create_datasets(
    domain='medical',
    tokenizer=tokenizer,
    max_samples=1000
)

# datasets = {
#     'train': UnifiedDataset(...),
#     'validation': UnifiedDataset(...) or 'test': UnifiedDataset(...)
# }
```

## 📊 Data Structures

### DomainConfig

```python
@dataclass
class DomainConfig:
    name: str                    # 도메인 이름
    data_path: str              # 데이터 경로
    train_file: str             # 훈련 파일명
    validation_file: str        # 검증 파일명
    test_file: str              # 테스트 파일명
    instruction_template: str   # 지시사항 템플릿
    response_format: str        # 응답 형식
    evaluation_type: str        # 평가 유형
    max_length: int = 512       # 최대 길이
    num_choices: int = 4        # 선택지 수
```

### Training Results

```python
{
    "domain": "medical",
    "train_loss": 0.1234,
    "train_runtime": 3600.0,
    "train_samples_per_second": 0.5,
    "adapter_path": "domain_models/medical/final_adapter"
}
```

### Evaluation Results

```python
{
    "domain": "medical",
    "accuracy": 0.7567,
    "total_samples": 1000,
    "correct_predictions": 757,
    "predictions": ["A", "B", "C", ...],
    "references": ["A", "B", "C", ...],
    "evaluation_type": "multiple_choice"
}
```

## 🔍 Error Handling

모든 함수는 적절한 예외 처리를 포함합니다:

```python
try:
    results = train_domain('medical', max_samples=1000)
except FileNotFoundError as e:
    print(f"데이터 파일을 찾을 수 없습니다: {e}")
except RuntimeError as e:
    print(f"CUDA 메모리 부족: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
```

## 📝 Best Practices

1. **메모리 관리**: 훈련 후 항상 `cleanup()` 호출
2. **도메인 검증**: 훈련 전 `check_data_availability()` 확인
3. **환경 검증**: 시작 전 `validate_environment()` 실행
4. **로깅**: 적절한 로깅 설정으로 진행 상황 추적
5. **예외 처리**: 모든 API 호출에 try-catch 블록 사용
