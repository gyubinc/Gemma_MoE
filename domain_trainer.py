#!/usr/bin/env python3
"""
Domain-specific LoRA Trainer Class
OOP design for modular training
"""

import os
import torch
import gc
import logging
from typing import Dict, Any, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

from dataset import create_domain_datasets
from utils import (
    setup_logging, print_gpu_memory_summary, clear_gpu_memory,
    cleanup_old_checkpoints
)

logger = logging.getLogger(__name__)

class DomainTrainer:
    """Domain-specific LoRA Trainer"""
    
    def __init__(self, config: Dict[str, Any], domain: str, experiment_dir: str):
        self.config = config
        self.domain = domain
        self.experiment_dir = experiment_dir
        self.output_dir = os.path.join(config['system']['output_dir'], domain)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_model(self) -> Tuple[Any, Any]:
        """Setup model and tokenizer with LoRA"""
        logger.info("🤖 Loading base model...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=self.config['model'].get('trust_remote_code', True)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=getattr(torch, self.config['model']['torch_dtype']),
            device_map=self.config['system']['device_map'],
            trust_remote_code=self.config['model'].get('trust_remote_code', True),
            max_memory={0: f"{self.config['system']['max_memory_mb']}MB"}
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['system'].get('gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def setup_datasets(self):
        """Setup training and evaluation datasets"""
        logger.info(f"📚 Loading {self.domain} dataset...")
        max_samples = self.config.get('domain_configs', {}).get(self.domain, {}).get('max_samples', None)
        
        self.train_dataset, self.eval_dataset = create_domain_datasets(
            domain=self.domain,
            tokenizer=self.tokenizer,
            max_length=int(self.config['training']['max_length']),
            max_samples=max_samples
        )
        
        logger.info(f"📊 {self.domain.upper()} Dataset:")
        logger.info(f"  Train: {len(self.train_dataset):,} samples")
        logger.info(f"  Eval: {len(self.eval_dataset):,} samples")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=int(self.config['training']['num_epochs']),
            per_device_train_batch_size=int(self.config['training']['per_device_batch_size']),
            per_device_eval_batch_size=int(self.config['training']['per_device_batch_size']),
            gradient_accumulation_steps=int(self.config['training']['gradient_accumulation_steps']),
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            warmup_ratio=float(self.config['training']['warmup_ratio']),
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            fp16=bool(self.config['training']['fp16']),
            bf16=bool(self.config['training'].get('bf16', False)),
            max_grad_norm=float(self.config['training']['max_grad_norm']),
            logging_steps=int(self.config['training']['logging_steps']),
            save_steps=int(self.config['training']['save_steps']),
            save_total_limit=int(self.config['training']['save_total_limit']),
            eval_steps=int(self.config['training']['eval_steps']),
            load_best_model_at_end=False,
            report_to=self.config['training']['report_to'],
            dataloader_num_workers=int(self.config['training']['dataloader_num_workers']),
            remove_unused_columns=bool(self.config['training']['remove_unused_columns']),
            seed=int(self.config['system']['seed']),
            gradient_checkpointing=bool(self.config['system'].get('gradient_checkpointing', True)),
        )
        
        return training_args
    
    def train(self) -> str:
        """Execute training process"""
        logger.info(f"🎯 Starting {self.domain} domain training...")
        
        # Clear GPU memory before training
        clear_gpu_memory()
        print_gpu_memory_summary(f"Before {self.domain} training")
        
        try:
            # Setup components
            self.setup_model()
            self.setup_datasets()
            training_args = self.setup_training_args()
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
            )
            
            # Training
            logger.info(f"🚀 Starting {self.domain} training...")
            trainer.train()
            
            # Save final model
            final_adapter_path = os.path.join(self.output_dir, "final_adapter")
            self.model.save_pretrained(final_adapter_path)
            self.tokenizer.save_pretrained(final_adapter_path)
            
            # Cleanup old checkpoints
            if self.config['system'].get('cleanup_checkpoints', True):
                cleanup_old_checkpoints(self.output_dir, self.config['system'].get('keep_last_checkpoints', 2))
            
            logger.info(f"✅ {self.domain} training completed!")
            logger.info(f"📁 Model saved to: {final_adapter_path}")
            
            return final_adapter_path
            
        except Exception as e:
            logger.error(f"❌ Error during {self.domain} training: {e}")
            raise
        finally:
            # Cleanup
            clear_gpu_memory()
            print_gpu_memory_summary(f"After {self.domain} training")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            del self.train_dataset
        if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
            del self.eval_dataset
        
        clear_gpu_memory()
        gc.collect()
