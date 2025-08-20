#!/usr/bin/env python3
"""
데이터셋 자동 다운로드 스크립트
GitHub 파일 크기 제한으로 인해 데이터를 별도로 다운로드합니다.
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_medical_data():
    """MedMCQA 데이터셋 다운로드"""
    print("📚 Downloading medical dataset (MedMCQA)...")
    
    os.makedirs("../data/medical", exist_ok=True)
    
    # Train set
    train_dataset = load_dataset("medmcqa", split="train")
    with open("../data/medical/medmcqa_train.json", "w") as f:
        json.dump([dict(item) for item in tqdm(train_dataset, desc="Processing train")], f)
    
    # Validation set
    val_dataset = load_dataset("medmcqa", split="validation")
    with open("../data/medical/medmcqa_validation.json", "w") as f:
        json.dump([dict(item) for item in tqdm(val_dataset, desc="Processing validation")], f)
    
    # Test set
    test_dataset = load_dataset("medmcqa", split="test")
    with open("../data/medical/medmcqa_test.json", "w") as f:
        json.dump([dict(item) for item in tqdm(test_dataset, desc="Processing test")], f)
    
    # Summary
    summary = {
        "dataset": "medmcqa",
        "splits": {
            "train": len(train_dataset),
            "validation": len(val_dataset), 
            "test": len(test_dataset)
        }
    }
    with open("../data/medical/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Medical dataset downloaded: {summary['splits']}")

def download_math_data():
    """GSM8K 데이터셋 다운로드"""
    print("📚 Downloading math dataset (GSM8K)...")
    
    os.makedirs("../data/math", exist_ok=True)
    
    # Train set
    train_dataset = load_dataset("gsm8k", "main", split="train")
    with open("../data/math/gsm8k_train.json", "w") as f:
        json.dump([dict(item) for item in tqdm(train_dataset, desc="Processing train")], f)
    
    # Test set
    test_dataset = load_dataset("gsm8k", "main", split="test")
    with open("../data/math/gsm8k_test.json", "w") as f:
        json.dump([dict(item) for item in tqdm(test_dataset, desc="Processing test")], f)
    
    # Summary
    summary = {
        "dataset": "gsm8k",
        "splits": {
            "train": len(train_dataset),
            "test": len(test_dataset)
        }
    }
    with open("../data/math/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Math dataset downloaded: {summary['splits']}")

def download_law_data():
    """LegalBench case_hold 데이터셋 다운로드"""
    print("📚 Downloading law dataset (LegalBench case_hold)...")
    
    os.makedirs("../data/law", exist_ok=True)
    
    # Train set
    train_dataset = load_dataset("nguha/legalbench", "case_hold", split="train")
    with open("../data/law/case_hold_train.json", "w") as f:
        json.dump([dict(item) for item in tqdm(train_dataset, desc="Processing train")], f)
    
    # Validation set
    val_dataset = load_dataset("nguha/legalbench", "case_hold", split="validation")
    with open("../data/law/case_hold_validation.json", "w") as f:
        json.dump([dict(item) for item in tqdm(val_dataset, desc="Processing validation")], f)
    
    # Test set
    test_dataset = load_dataset("nguha/legalbench", "case_hold", split="test")
    with open("../data/law/case_hold_test.json", "w") as f:
        json.dump([dict(item) for item in tqdm(test_dataset, desc="Processing test")], f)
    
    # Summary
    summary = {
        "dataset": "legalbench_case_hold",
        "splits": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset)
        }
    }
    with open("../data/law/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Law dataset downloaded: {summary['splits']}")

def download_code_data():
    """CodeXGLUE 데이터셋 다운로드"""
    print("📚 Downloading code dataset (CodeXGLUE)...")
    
    os.makedirs("../data/code", exist_ok=True)
    
    # Train set
    train_dataset = load_dataset("microsoft/CodeXGLUE", "text-to-code-generation-codexglue_text_to_code-python-en", split="train")
    with open("../data/code/codexglue_train.json", "w") as f:
        json.dump([dict(item) for item in tqdm(train_dataset, desc="Processing train")], f)
    
    # Validation set
    val_dataset = load_dataset("microsoft/CodeXGLUE", "text-to-code-generation-codexglue_text_to_code-python-en", split="validation")
    with open("../data/code/codexglue_validation.json", "w") as f:
        json.dump([dict(item) for item in tqdm(val_dataset, desc="Processing validation")], f)
    
    # Test set
    test_dataset = load_dataset("microsoft/CodeXGLUE", "text-to-code-generation-codexglue_text_to_code-python-en", split="test")
    with open("../data/code/codexglue_test.json", "w") as f:
        json.dump([dict(item) for item in tqdm(test_dataset, desc="Processing test")], f)
    
    # Summary
    summary = {
        "dataset": "codexglue_text_to_code",
        "splits": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset)
        }
    }
    with open("../data/code/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Code dataset downloaded: {summary['splits']}")

def create_overall_summary():
    """전체 데이터셋 요약 생성"""
    summaries = {}
    
    for domain in ["medical", "law", "math", "code"]:
        summary_path = f"../data/{domain}/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summaries[domain] = json.load(f)
    
    overall = {
        "total_datasets": len(summaries),
        "domains": list(summaries.keys()),
        "dataset_info": summaries
    }
    
    with open("../data/overall_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    
    print(f"📊 Overall summary created: {len(summaries)} domains")

def main():
    """메인 다운로드 함수"""
    print("🚀 Starting dataset download...")
    
    # 데이터 폴더 생성
    os.makedirs("../data", exist_ok=True)
    
    # 각 도메인 데이터 다운로드
    download_medical_data()
    download_math_data()
    download_law_data()
    download_code_data()
    
    # 전체 요약 생성
    create_overall_summary()
    
    print("✅ All datasets downloaded successfully!")
    print("📁 Data structure:")
    print("../data/")
    print("├── medical/ (MedMCQA)")
    print("├── law/ (LegalBench)")
    print("├── math/ (GSM8K)")
    print("└── code/ (CodeXGLUE)")

if __name__ == "__main__":
    main()
