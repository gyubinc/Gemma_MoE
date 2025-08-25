#!/usr/bin/env python3
"""
Unified trainer for Qwen-MoE project
Supports training for all domains with centralized configuration
"""

import torch
import os
import logging
import json
from typing import Dict, Any, Optional
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from ..configs.domains import domain_manager
from .dataset import UnifiedDataset, create_datasets
from .model import model_manager
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """Unified trainer supporting all domains"""
    
    def __init__(self, domain: str, output_dir: str = "domain_models", device: str = None):
        self.domain = domain
        self.output_dir = output_dir
        self.device = device or self._get_device_from_config()
        
        # Get domain configuration
        self.domain_config = domain_manager.get_domain(domain)
        
        # Setup output directory
        self.domain_output_dir = os.path.join(output_dir, domain)
        os.makedirs(self.domain_output_dir, exist_ok=True)
        
        # Training components
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
    
    def _get_device_from_config(self) -> str:
        """Get device from configuration"""
        config = get_config()
        return config.get_cuda_device()
    
    def setup_training(self, max_samples: int = None):
        """Setup training components"""
        logger.info(f"Setting up training for {self.domain} domain")
        
        # Load model
        self.model, self.tokenizer = model_manager.load_base_model()
        
        # Load datasets
        datasets = create_datasets(self.domain, self.tokenizer, max_samples)
        self.train_dataset = datasets['train']
        self.eval_dataset = datasets['test']  # validation → test로 변경
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup trainer
        self._setup_trainer()
        
        logger.info(f"Training setup completed for {self.domain} domain")
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        config = get_config()
        lora_config_dict = config.get_lora_config()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config_dict.get('r', 64),
            lora_alpha=lora_config_dict.get('alpha', 128),
            lora_dropout=lora_config_dict.get('dropout', 0.1),
            target_modules=lora_config_dict.get('target_modules', ["gate_proj", "up_proj", "down_proj"]),
            bias=lora_config_dict.get('bias', "none")
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()  # Set to training mode
        
        # Ensure LoRA parameters require gradients
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        
        self.model.print_trainable_parameters()
    
    def _setup_trainer(self):
        """Setup trainer with domain-specific configuration"""
        config = get_config()
        training_config = config.get_training_config()
        
        training_args = TrainingArguments(
            output_dir=self.domain_output_dir,
            num_train_epochs=training_config.get('epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 16),
            learning_rate=float(training_config.get('learning_rate', 2e-4)),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            lr_scheduler_type=training_config.get('lr_scheduler_type', "cosine"),
            fp16=training_config.get('fp16', True),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            logging_steps=training_config.get('logging_steps', 5),
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=training_config.get('save_total_limit', 2),
            eval_strategy=training_config.get('eval_strategy', "steps"),
            eval_steps=training_config.get('eval_steps', 500),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Model will be moved to device by trainer automatically
    
    def train(self) -> Dict[str, Any]:
        """Train the model"""
        logger.info(f"Starting training for {self.domain} domain")
        
        try:
            # Train
            train_result = self.trainer.train()
            
            # Save model
            final_adapter_path = os.path.join(self.domain_output_dir, "final_adapter")
            self.trainer.save_model(final_adapter_path)
            
            # Save training results
            results = {
                "domain": self.domain,
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "adapter_path": final_adapter_path
            }
            
            # Save results
            results_path = os.path.join(self.domain_output_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training completed for {self.domain} domain")
            logger.info(f"Final adapter saved to: {final_adapter_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed for {self.domain} domain: {e}")
            raise
    
    def evaluate(self, adapter_path: str = None) -> Dict[str, Any]:
        """Evaluate the trained model"""
        logger.info(f"Evaluating {self.domain} domain model")
        
        try:
            # Load model with adapter if provided
            if adapter_path:
                model, tokenizer = model_manager.load_model_with_adapter(adapter_path)
            else:
                model, tokenizer = self.model, self.tokenizer
            
            # Evaluate
            eval_results = self.trainer.evaluate()
            
            # Save evaluation results
            eval_path = os.path.join(self.domain_output_dir, "evaluation_results.json")
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info(f"Evaluation completed for {self.domain} domain")
            return eval_results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {self.domain} domain: {e}")
            raise
    
    def cleanup(self):
        """Cleanup training resources"""
        if self.trainer:
            del self.trainer
            self.trainer = None
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Cleanup completed for {self.domain} domain")

def train_domain(domain: str, max_samples: int = None, output_dir: str = "domain_models") -> Dict[str, Any]:
    """Convenience function to train a domain"""
    trainer = UnifiedTrainer(domain, output_dir)
    
    try:
        trainer.setup_training(max_samples)
        results = trainer.train()
        trainer.cleanup()
        return results
    except Exception as e:
        trainer.cleanup()
        raise
