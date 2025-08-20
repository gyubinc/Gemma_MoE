#!/usr/bin/env python3
"""
Domain-specific Full MLP Fine-tuning Script
각 도메인에 대해 Gemma 모델의 모든 MLP 레이어를 full fine-tuning
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from tqdm import tqdm
import logging
from datetime import datetime

from dataset import create_domain_datasets
from utils import (
    setup_logging, 
    print_gpu_memory_summary, 
    calculate_model_parameters,
    save_config
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def apply_lora_to_mlp(model, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """MLP 레이어에만 LoRA 적용"""
    
    # LoRA configuration for MLP layers only
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "gate_proj",  # MLP gate projection
            "up_proj",    # MLP up projection  
            "down_proj"   # MLP down projection
        ],
        bias="none",
        inference_mode=False,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Enable training mode and ensure gradients are enabled
    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"🔓 Trainable LoRA parameters: {trainable_params:,}")
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Print some trainable parameter names for debugging
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"🔍 Sample trainable parameters: {trainable_param_names[:5]}")
    
    return model, trainable_params


def create_domain_trainer(model, tokenizer, train_dataset, args):
    """도메인별 trainer 생성"""
    
    # Custom data collator to handle our format
    def data_collator(features):
        batch = {}
        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features])
        batch['labels'] = torch.tensor([f['labels'] for f in features])
        return batch
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler_type,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,  # FP16 비활성화로 gradient scaler 문제 해결
        gradient_checkpointing=True,  # 메모리 절약을 위한 gradient checkpointing
        dataloader_drop_last=True,
        dataloader_num_workers=2,  # 메모리 절약을 위해 worker 수 감소
        optim="adamw_8bit",  # 8bit optimizer 사용으로 메모리 절약
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # wandb 비활성화
        seed=args.seed,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer


def train_domain_model(args):
    """특정 도메인에 대해 MLP 레이어 full fine-tuning"""
    
    logger.info(f"🚀 Starting {args.domain} domain MLP fine-tuning")
    print_gpu_memory_summary("Initial", args.gpu_id)
    
    # Set device
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    
    # Load tokenizer
    logger.info("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("🤖 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    
    print_gpu_memory_summary("Model loaded", args.gpu_id)
    
    # Apply LoRA to MLP layers
    logger.info("🔧 Applying LoRA to MLP layers...")
    model, trainable_params = apply_lora_to_mlp(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Load domain dataset
    logger.info(f"📚 Loading {args.domain} dataset...")
    datasets = create_domain_datasets(tokenizer, args.max_length, 'train')
    train_dataset = datasets[args.domain]
    
    logger.info(f"📊 Dataset size: {len(train_dataset)} samples")
    print_gpu_memory_summary("Dataset loaded", args.gpu_id)
    
    # Create output directory
    domain_output_dir = os.path.join(args.output_dir, args.domain)
    os.makedirs(domain_output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'domain': args.domain,
        'model': args.model,
        'trainable_parameters': trainable_params,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_length': args.max_length,
        'timestamp': datetime.now().isoformat()
    }
    save_config(config, os.path.join(domain_output_dir, 'training_config.json'))
    
    # Create trainer
    trainer_args = argparse.Namespace(**vars(args))
    trainer_args.output_dir = domain_output_dir
    trainer = create_domain_trainer(model, tokenizer, train_dataset, trainer_args)
    
    print_gpu_memory_summary("Before training", args.gpu_id)
    
    # Start training
    logger.info(f"🎯 Starting training for {args.domain} domain...")
    try:
        trainer.train()
        logger.info(f"✅ {args.domain} domain training completed!")
        
        # Save final LoRA adapter
        final_model_path = os.path.join(domain_output_dir, "final_adapter")
        model.save_pretrained(final_model_path)
        logger.info(f"💾 LoRA adapter saved to {final_model_path}")
        
        print_gpu_memory_summary("Training completed", args.gpu_id)
        
        return final_model_path
        
    except Exception as e:
        logger.error(f"❌ Training failed for {args.domain}: {str(e)}")
        raise e


def parse_args():
    parser = argparse.ArgumentParser(description="Full MLP fine-tuning for specific domain")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                       help="Base model name")
    parser.add_argument("--domain", type=str, required=True,
                       choices=["medical", "law", "math", "code"],
                       help="Domain to fine-tune for")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # System arguments
    parser.add_argument("--output_dir", type=str, default="./domain_models",
                       help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=6,
                       help="GPU ID to use")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint interval")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train domain model
    model_path = train_domain_model(args)
    
    logger.info(f"🎉 {args.domain} domain fine-tuning completed!")
    logger.info(f"📁 Model saved at: {model_path}")


if __name__ == "__main__":
    main()
