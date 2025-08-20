#!/usr/bin/env python3
"""
Utility functions for Qwen-MoE project
Optimized for A6000 46GB VRAM
"""

import os
import yaml
import logging
import torch
import gc
from typing import Dict, Any, Optional, List
import psutil
import GPUtil
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def print_gpu_memory_summary(stage: str = ""):
    """Print detailed GPU memory usage summary"""
    if not torch.cuda.is_available():
        print(f"[GPU] {stage}: CUDA not available")
        return
    
    try:
        # Initialize CUDA if needed
        if torch.cuda.device_count() == 0:
            torch.cuda.init()
        
        # Get GPU info using PyTorch
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            free = total - reserved
            
            print(f"[GPU{i}] {stage}: mem: alloc={allocated:.2f} GiB, "
                  f"reserved={reserved:.2f} GiB, max_alloc={max_allocated:.2f} GiB, "
                  f"free={free:.2f} GiB/ total={total:.2f} GiB")
    
    except Exception as e:
        print(f"[GPU] {stage}: Error getting memory info - {e}")
        # Fallback to basic info
        try:
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"[GPU{i}] {stage}: Total memory: {total:.2f} GiB")
        except Exception as e2:
            print(f"[GPU] {stage}: Fallback also failed - {e2}")

def print_system_memory_summary():
    """Print system memory usage"""
    memory = psutil.virtual_memory()
    print(f"[SYSTEM] Memory: used={memory.used/1024**3:.2f} GiB, "
          f"available={memory.available/1024**3:.2f} GiB, "
          f"total={memory.total/1024**3:.2f} GiB")

def clear_gpu_memory():
    """Clear GPU memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_optimal_batch_size(model_size_gb: float, gpu_memory_gb: float = 46.0) -> Dict[str, int]:
    """Calculate optimal batch sizes for A6000 46GB VRAM"""
    # Conservative memory allocation (leave 20% for overhead)
    available_memory = gpu_memory_gb * 0.8
    
    # Base memory per sample (rough estimate)
    base_memory_per_sample = model_size_gb * 0.1  # Conservative estimate
    
    # Calculate batch sizes
    max_batch_size = int(available_memory / base_memory_per_sample)
    
    # Conservative batch sizes for different scenarios
    return {
        "per_device_batch_size": min(4, max_batch_size),
        "gradient_accumulation_steps": max(8, max_batch_size // 4),
        "eval_batch_size": min(2, max_batch_size // 2)
    }

def create_optimized_config() -> Dict[str, Any]:
    """Create optimized configuration for A6000 46GB VRAM"""
    return {
        "model": {
            "name": "Qwen/Qwen3-4B-Instruct-2507",
            "torch_dtype": "bfloat16"
        },
        "lora": {
            "r": 64,
            "alpha": 128,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "dropout": 0.1,
            "bias": "none"
        },
        "training": {
            "num_epochs": 3,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "fp16": True,
            "max_grad_norm": 1.0,
            "max_length": 512,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 2
        },
        "system": {
            "output_dir": "./domain_models",
            "seed": 42,
            "gradient_checkpointing": True
        }
    }

def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def check_data_availability(domains: list) -> Dict[str, bool]:
    """Check if data files are available for each domain"""
    availability = {}
    
    for domain in domains:
        data_path = f"data/{domain}"
        if domain == "medical":
            required_files = {
                "train": f"{data_path}/medmcqa_train.json",
                "validation": f"{data_path}/medmcqa_validation.json",
                "test": f"{data_path}/medmcqa_test.json"
            }
        elif domain == "law":
            required_files = {
                "train": f"{data_path}/case_hold_train.json",
                "validation": f"{data_path}/case_hold_validation.json",
                "test": f"{data_path}/case_hold_test.json"
            }
        elif domain == "math":
            required_files = {
                "train": f"{data_path}/gsm8k_train.json",
                "test": f"{data_path}/gsm8k_test.json"
            }
        elif domain == "code":
            required_files = {
                "train": f"{data_path}/codexglue_train.json",
                "validation": f"{data_path}/codexglue_validation.json",
                "test": f"{data_path}/codexglue_test.json"
            }
        else:
            required_files = {
                "train": f"{data_path}/{domain}_train.json",
                "validation": f"{data_path}/{domain}_validation.json",
                "test": f"{data_path}/{domain}_test.json"
            }
        
        availability[domain] = all(os.path.exists(f) for f in required_files.values())
    
    return availability

def validate_environment():
    """Validate the training environment"""
    issues = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        issues.append("CUDA is not available")
    else:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            issues.append("No GPU devices found")
        else:
            print(f"Found {gpu_count} GPU device(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check required packages
    required_packages = [
        "transformers", "peft", "accelerate", "datasets", 
        "torch", "numpy", "yaml"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Package '{package}' is not installed")
    
    # Check data directories
    if not os.path.exists("data"):
        issues.append("Data directory not found")
    
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Environment validation passed")
        return True

def create_experiment_dir(experiment_name: str) -> str:
    """Create experiment directory with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def log_experiment_config(config: Dict[str, Any], experiment_dir: str):
    """Log experiment configuration"""
    config_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_path)
    print(f"📁 Experiment config saved to: {config_path}")

def cleanup_old_checkpoints(output_dir: str, keep_last: int = 2):
    """Clean up old checkpoints to save disk space"""
    if not os.path.exists(output_dir):
        return
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            checkpoints.append(item)
    
    if len(checkpoints) > keep_last:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        for checkpoint in checkpoints[:-keep_last]:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            import shutil
            shutil.rmtree(checkpoint_path)
            print(f"🗑️ Removed old checkpoint: {checkpoint}")

if __name__ == "__main__":
    # Test utility functions
    setup_logging()
    print("🧪 Testing utility functions...")
    
    # Test environment validation
    validate_environment()
    
    # Test config creation
    config = create_optimized_config()
    print("✅ Config created successfully")
    
    # Test GPU memory summary
    print_gpu_memory_summary("Test")
    
    # Test data availability
    availability = check_data_availability(["medical", "law", "math", "code"])
    print(f"📊 Data availability: {availability}")

# Medical domain evaluation functions
def evaluate_medical_accuracy(predictions: List[str], references: List[str]) -> float:
    """Evaluate accuracy for medical domain (MedMCQA)"""
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Clean predictions and references
        pred_clean = clean_medical_prediction(pred)
        ref_clean = ref.strip().upper()
        
        if pred_clean == ref_clean:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def clean_medical_prediction(prediction: str) -> str:
    """Clean medical prediction to extract answer choice"""
    # Remove common prefixes and suffixes
    pred = prediction.strip()
    
    # Extract answer choice (A, B, C, D)
    # Look for patterns like "Answer: A", "The answer is B", etc.
    patterns = [
        r'answer[:\s]*([ABCD])',
        r'([ABCD])[\.\)]',
        r'option[:\s]*([ABCD])',
        r'choice[:\s]*([ABCD])',
        r'([ABCD])\s*$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no pattern found, try to extract single letter
    letters = re.findall(r'\b[ABCD]\b', pred.upper())
    if letters:
        return letters[0]
    
    return pred.upper()

def load_model_for_evaluation(model_path: str, adapter_path: Optional[str] = None, device: str = "cuda:0"):
    """Load model for evaluation"""
    logger = logging.getLogger(__name__)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer

def evaluate_medical_model(model, tokenizer, test_dataset, max_samples: int = 1000, device: str = "cuda:0"):
    """Evaluate medical model on test dataset"""
    logger = logging.getLogger(__name__)
    
    model.eval()
    predictions = []
    references = []
    
    # Limit samples for evaluation
    eval_samples = min(len(test_dataset), max_samples)
    logger.info(f"Evaluating on {eval_samples} samples")
    
    with torch.no_grad():
        for i in range(eval_samples):
            if i % 100 == 0:
                logger.info(f"Evaluating sample {i}/{eval_samples}")
            
            # Get sample
            sample = test_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            
            # Find the position where assistant response should start
            assistant_start = None
            for j in range(len(input_ids[0]) - 2):
                if (input_ids[0][j] == tokenizer.encode("<|im_start|>")[0] and 
                    input_ids[0][j+1] == tokenizer.encode("assistant")[0] and
                    input_ids[0][j+2] == tokenizer.encode("\n")[0]):
                    assistant_start = j + 3
                    break
            
            if assistant_start is None:
                continue
            
            # Generate response
            try:
                outputs = model.generate(
                    input_ids=input_ids[:, :assistant_start],
                    attention_mask=attention_mask[:, :assistant_start],
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.encode("<|im_end|>")[0]
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(outputs[0][assistant_start:], skip_special_tokens=True)
                predictions.append(generated_text)
                
                # Get reference answer
                ref_text = tokenizer.decode(sample["input_ids"][assistant_start:], skip_special_tokens=True)
                references.append(ref_text)
                
            except Exception as e:
                logger.warning(f"Error generating for sample {i}: {e}")
                continue
    
    # Calculate accuracy
    accuracy = evaluate_medical_accuracy(predictions, references)
    
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "references": references,
        "num_samples": len(predictions)
    }

def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Evaluation results saved to: {output_path}")

def evaluate_domain_model(model, tokenizer, test_dataset, domain: str, max_samples: int = 1000, device: str = "cuda:0") -> Dict[str, Any]:
    """Evaluate model on any domain dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"📊 Evaluating {domain} domain model...")
    
    model.eval()
    predictions = []
    references = []
    
    # Limit samples for evaluation
    eval_samples = min(max_samples, len(test_dataset))
    logger.info(f"📊 Evaluating on {eval_samples} samples")
    
    with torch.no_grad():
        for i in range(eval_samples):
            if i % 100 == 0:
                logger.info(f"📊 Processing sample {i}/{eval_samples}")
            
            sample = test_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            
            # Generate prediction
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode prediction and reference
            prediction = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            reference = tokenizer.decode(sample["labels"][sample["labels"] != -100], skip_special_tokens=True)
            
            predictions.append(prediction.strip())
            references.append(reference.strip())
    
    # Calculate accuracy (simple exact match for now)
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        if pred.strip().lower() == ref.strip().lower():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "domain": domain,
        "accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "predictions": predictions[:10],  # Save first 10 for inspection
        "references": references[:10]
    }
    
    logger.info(f"📊 {domain.title()} domain accuracy: {accuracy:.4f} ({correct}/{total})")
    return results
