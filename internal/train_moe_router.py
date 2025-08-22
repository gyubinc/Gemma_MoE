#!/usr/bin/env python3
"""
MoE Router Training Script
Trains the router network on combined dataset from all domains
"""

import argparse
import os
import sys
import yaml
import torch
import gc
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import logging
from typing import Dict, Any, List
from torch.utils.data import ConcatDataset, DataLoader

from moe_architecture import create_moe_model, load_domain_adapters
from dataset import create_domain_datasets
from utils import (
    setup_logging, print_gpu_memory_summary, load_config, 
    clear_gpu_memory, validate_environment, create_experiment_dir, 
    log_experiment_config
)

logger = logging.getLogger(__name__)

class MoEDataset(torch.utils.data.Dataset):
    """Combined dataset for MoE training"""
    
    def __init__(self, domain_datasets: Dict[str, Any], tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        
        # Combine samples from all domains
        for domain, (train_ds, eval_ds) in domain_datasets.items():
            logger.info(f"Adding {domain} domain: {len(train_ds)} train, {len(eval_ds)} eval samples")
            
            # Add train samples
            for i in range(len(train_ds)):
                sample = train_ds[i]
                sample['domain'] = domain
                self.samples.append(sample)
            
            # Add eval samples
            for i in range(len(eval_ds)):
                sample = eval_ds[i]
                sample['domain'] = domain
                self.samples.append(sample)
        
        logger.info(f"Total combined samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def create_combined_dataset(config: Dict[str, Any], tokenizer) -> MoEDataset:
    """Create combined dataset from all domains"""
    logger.info("📚 Creating combined dataset from all domains...")
    
    domain_datasets = {}
    domains = config.get('domains', ['medical', 'law', 'math', 'code'])
    
    for domain in domains:
        try:
            train_ds, eval_ds = create_domain_datasets(
                domain=domain,
                tokenizer=tokenizer,
                max_length=config['training']['max_length']
            )
            domain_datasets[domain] = (train_ds, eval_ds)
            logger.info(f"✅ Loaded {domain} dataset")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {domain} dataset: {e}")
            continue
    
    if len(domain_datasets) < 2:
        raise ValueError("Need at least 2 domain datasets for MoE training")
    
    return MoEDataset(domain_datasets, tokenizer)

def train_moe_router(config: Dict[str, Any], experiment_dir: str) -> str:
    """Train MoE router on combined dataset"""
    logger.info("🎯 Starting MoE router training...")
    
    # Clear GPU memory
    clear_gpu_memory()
    print_gpu_memory_summary("Before MoE training")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['name'],
            trust_remote_code=config['model'].get('trust_remote_code', True)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load domain adapters
        domain_adapters = load_domain_adapters(config['system']['output_dir'])
        
        if len(domain_adapters) < 2:
            raise ValueError(f"Need at least 2 domain adapters, found {len(domain_adapters)}")
        
        # Create MoE model
        moe_model = create_moe_model(
            base_model_name=config['model']['name'],
            domain_adapters=domain_adapters,
            num_experts=config['moe']['num_experts'],
            router_type=config['moe']['router_type']
        )
        
        # Create combined dataset
        combined_dataset = create_combined_dataset(config, tokenizer)
        
        # Split dataset
        total_size = len(combined_dataset)
        train_size = int(0.8 * total_size)
        eval_size = total_size - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(
            combined_dataset, [train_size, eval_size]
        )
        
        logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Training arguments
        output_dir = os.path.join(experiment_dir, "moe_model")
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config['moe']['num_epochs'],
            per_device_train_batch_size=config['moe']['batch_size'],
            per_device_eval_batch_size=config['moe']['batch_size'],
            gradient_accumulation_steps=config['moe']['gradient_accumulation_steps'],
            learning_rate=config['moe']['learning_rate'],
            weight_decay=config['moe']['weight_decay'],
            warmup_ratio=config['moe']['warmup_ratio'],
            lr_scheduler_type=config['moe']['lr_scheduler_type'],
            fp16=config['training']['fp16'],
            bf16=config['training'].get('bf16', False),
            max_grad_norm=config['training']['max_grad_norm'],
            logging_steps=config['moe']['logging_steps'],
            save_steps=config['moe']['save_steps'],
            save_total_limit=config['moe']['save_total_limit'],
            evaluation_strategy=config['moe']['evaluation_strategy'],
            eval_steps=config['moe']['eval_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=config['training']['report_to'],
            dataloader_num_workers=config['training']['dataloader_num_workers'],
            remove_unused_columns=config['training']['remove_unused_columns'],
            seed=config['system']['seed'],
            gradient_checkpointing=config['system'].get('gradient_checkpointing', True),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Custom training step to handle router loss
        class MoETrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                outputs = model(**inputs, return_router_loss=True)
                loss = outputs.get('total_loss', outputs['loss'])
                return (loss, outputs) if return_outputs else loss
        
        # Trainer
        trainer = MoETrainer(
            model=moe_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Training
        logger.info("🚀 Starting MoE router training...")
        trainer.train()
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_moe_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"✅ MoE router training completed!")
        logger.info(f"📁 Model saved to: {final_model_path}")
        
        return final_model_path
        
    except Exception as e:
        logger.error(f"❌ Error during MoE training: {e}")
        raise
    finally:
        # Cleanup
        clear_gpu_memory()
        print_gpu_memory_summary("After MoE training")

def main():
    """Main MoE router training function"""
    parser = argparse.ArgumentParser(description="MoE Router Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--experiment_name", type=str, default="moe_router_training", help="Experiment name")
    
    args = parser.parse_args()
    
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Setup logging
    experiment_dir = create_experiment_dir(args.experiment_name)
    log_file = os.path.join(experiment_dir, "moe_training.log")
    setup_logging(log_file=log_file)
    
    logger.info(f"🚀 Starting MoE router training")
    logger.info(f"📁 Experiment directory: {experiment_dir}")
    
    try:
        # Environment validation
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            sys.exit(1)
        
        # Load config
        config = load_config(args.config)
        log_experiment_config(config, experiment_dir)
        
        # Check domain adapters
        domain_adapters = load_domain_adapters(config['system']['output_dir'])
        if len(domain_adapters) < 2:
            logger.error(f"❌ Need at least 2 domain adapters, found {len(domain_adapters)}")
            sys.exit(1)
        
        logger.info(f"✅ Found {len(domain_adapters)} domain adapters: {list(domain_adapters.keys())}")
        
        # Train MoE router
        model_path = train_moe_router(config, experiment_dir)
        
        logger.info(f"🎉 MoE router training completed successfully!")
        logger.info(f"📍 Model saved at: {model_path}")
        
        # Final memory summary
        print_gpu_memory_summary("Final")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
