#!/usr/bin/env python3
"""
Domain-specific LoRA training script for Qwen-MoE
Optimized for A6000 46GB VRAM
"""

import argparse
import os
import sys
import yaml
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, Any, Tuple

from dataset import create_domain_datasets
from utils import (
    setup_logging, print_gpu_memory_summary, load_config, 
    clear_gpu_memory, validate_environment, check_data_availability,
    create_experiment_dir, log_experiment_config, cleanup_old_checkpoints,
    evaluate_domain_model, load_model_for_evaluation, save_evaluation_results
)

logger = logging.getLogger(__name__)

def create_lora_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Create LoRA model with optimized settings for A6000"""
    logger.info("🤖 Loading base model...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['torch_dtype']),
        device_map=config['system']['device_map'],
        trust_remote_code=config['model'].get('trust_remote_code', True),
        max_memory={0: f"{config['system']['max_memory_mb']}MB"}
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config['system'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_domain_model(domain: str, config: Dict[str, Any], experiment_dir: str) -> str:
    """Train domain-specific LoRA model"""
    logger.info(f"🎯 Starting {domain} domain training...")
    
    # Create output directory
    output_dir = os.path.join(config['system']['output_dir'], domain)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear GPU memory before training
    clear_gpu_memory()
    print_gpu_memory_summary(f"Before {domain} training")
    
    try:
        # Create model and tokenizer
        model, tokenizer = create_lora_model(config)
        
        # Load dataset
        logger.info(f"📚 Loading {domain} dataset...")
        max_samples = config.get('domain_configs', {}).get(domain, {}).get('max_samples', None)
        train_dataset, eval_dataset = create_domain_datasets(
            domain=domain,
            tokenizer=tokenizer,
            max_length=int(config['training']['max_length']),
            max_samples=max_samples
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(config['training']['num_epochs']),
            per_device_train_batch_size=int(config['training']['per_device_batch_size']),
            per_device_eval_batch_size=int(config['training']['per_device_batch_size']),
            gradient_accumulation_steps=int(config['training']['gradient_accumulation_steps']),
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            warmup_ratio=float(config['training']['warmup_ratio']),
            lr_scheduler_type=config['training']['lr_scheduler_type'],
            fp16=bool(config['training']['fp16']),
            bf16=bool(config['training'].get('bf16', False)),
            max_grad_norm=float(config['training']['max_grad_norm']),
            logging_steps=int(config['training']['logging_steps']),
            save_steps=int(config['training']['save_steps']),
            save_total_limit=int(config['training']['save_total_limit']),
            eval_steps=int(config['training']['eval_steps']),
            load_best_model_at_end=False,
            report_to=config['training']['report_to'],
            dataloader_num_workers=int(config['training']['dataloader_num_workers']),
            remove_unused_columns=bool(config['training']['remove_unused_columns']),
            seed=int(config['system']['seed']),
            gradient_checkpointing=bool(config['system'].get('gradient_checkpointing', True)),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Training
        logger.info(f"🚀 Starting {domain} training...")
        trainer.train()
        
        # Save final model
        final_adapter_path = os.path.join(output_dir, "final_adapter")
        model.save_pretrained(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        
        # Evaluate trained model (for all domains)
        logger.info(f"🔍 Evaluating trained {domain} model...")
        evaluation_results = evaluate_trained_model(config, domain, final_adapter_path, experiment_dir)
        logger.info(f"📊 {domain.title()} evaluation accuracy: {evaluation_results['accuracy']:.4f}")
        
        # Cleanup old checkpoints
        if config['system'].get('cleanup_checkpoints', True):
            cleanup_old_checkpoints(output_dir, config['system'].get('keep_last_checkpoints', 2))
        
        logger.info(f"✅ {domain} training completed!")
        logger.info(f"📁 Model saved to: {final_adapter_path}")
        
        return final_adapter_path
        
    except Exception as e:
        logger.error(f"❌ Error during {domain} training: {e}")
        raise
    finally:
        # Cleanup
        clear_gpu_memory()
        print_gpu_memory_summary(f"After {domain} training")

def evaluate_trained_model(config: Dict[str, Any], domain: str, adapter_path: str, experiment_dir: str) -> Dict[str, Any]:
    """Evaluate trained model on test dataset"""
    logger.info("🔍 Loading trained model for evaluation...")
    
    # Load trained model
    model, tokenizer = load_model_for_evaluation(
        config['model']['name'],
        adapter_path=adapter_path,
        device="cuda:0"
    )
    
    # Load test dataset
    logger.info("📚 Loading test dataset...")
    _, test_dataset = create_domain_datasets(
        domain=domain,
        tokenizer=tokenizer,
        max_length=int(config['training']['max_length']),
        max_samples=1000  # Limit evaluation samples
    )
    
    # Evaluate all domains with the same method
    logger.info("📊 Running evaluation...")
    evaluation_results = evaluate_domain_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        domain=domain,
        max_samples=1000,
        device="cuda:0"
    )
    
    # Save evaluation results
    eval_path = os.path.join(experiment_dir, "evaluation_results.json")
    save_evaluation_results(evaluation_results, eval_path)
    
    logger.info(f"📊 Evaluation accuracy: {evaluation_results['accuracy']:.4f}")
    
    # Cleanup
    del model, tokenizer
    clear_gpu_memory()
    
    return evaluation_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Domain-specific LoRA training for Qwen-MoE")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--domain", type=str, required=True, 
                       choices=["medical", "law", "math", "code"], 
                       help="Domain to train")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--experiment_name", type=str, default="domain_training", help="Experiment name")
    
    args = parser.parse_args()
    
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Setup logging
    experiment_dir = create_experiment_dir(args.experiment_name)
    log_file = os.path.join(experiment_dir, f"{args.domain}_training.log")
    setup_logging(log_file=log_file)
    
    logger.info(f"🚀 Starting {args.domain} domain training")
    logger.info(f"📁 Experiment directory: {experiment_dir}")
    
    try:
        # Environment validation
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            sys.exit(1)
        
        # Load config
        config = load_config(args.config)
        log_experiment_config(config, experiment_dir)
        
        # Check data availability
        availability = check_data_availability([args.domain])
        if not availability.get(args.domain, False):
            logger.error(f"❌ Data not available for {args.domain} domain")
            sys.exit(1)
        
        # Train domain model
        adapter_path = train_domain_model(args.domain, config, experiment_dir)
        
        logger.info(f"🎉 {args.domain} domain training completed successfully!")
        logger.info(f"📍 Adapter saved at: {adapter_path}")
        
        # Final memory summary
        print_gpu_memory_summary("Final")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()