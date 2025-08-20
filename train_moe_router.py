#!/usr/bin/env python3
"""
MoE Router Training Script
Train the Top-1 routing mechanism on multi-domain dataset
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime

from dataset import create_domain_datasets, collate_fn
from utils import (
    setup_logging, 
    print_gpu_memory_summary, 
    save_config
)
from moe_architecture import create_moe_model

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class MoETrainer:
    """MoE Router Trainer"""
    
    def __init__(self, 
                 moe_model,
                 tokenizer,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 scheduler,
                 accelerator,
                 args):
        
        self.moe_model = moe_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.args = args
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Prepare for training
        (self.moe_model, self.optimizer, 
         self.train_dataloader, self.val_dataloader) = self.accelerator.prepare(
            self.moe_model, self.optimizer, 
            self.train_dataloader, self.val_dataloader
        )
        
        logger.info("✅ MoE Trainer initialized")
    
    def train_epoch(self):
        """Train one epoch"""
        self.moe_model.train()
        
        total_loss = 0
        total_aux_loss = 0
        num_batches = 0
        
        routing_stats = defaultdict(lambda: defaultdict(int))
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.epoch}", 
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.moe_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Get main loss and auxiliary loss
            main_loss = outputs.loss
            
            # Collect auxiliary losses from all MoE layers
            aux_loss = 0
            aux_loss_count = 0
            for layer in self.moe_model.base_model.model.layers:
                if hasattr(layer.mlp, '_last_aux_loss'):
                    aux_loss += layer.mlp._last_aux_loss
                    aux_loss_count += 1
            
            if aux_loss_count > 0:
                aux_loss = aux_loss / aux_loss_count
            
            # Total loss
            total_loss_batch = main_loss + aux_loss
            
            # Backward pass
            self.accelerator.backward(total_loss_batch)
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.moe_model.parameters(), 
                    self.args.max_grad_norm
                )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += main_loss.item()
            total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            num_batches += 1
            self.global_step += 1
            
            # Collect routing statistics
            if self.global_step % self.args.routing_stats_steps == 0:
                self._collect_routing_stats(batch['input_ids'], routing_stats)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{main_loss.item():.4f}",
                'aux_loss': f"{aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % self.args.logging_steps == 0:
                self._log_training_step(main_loss.item(), aux_loss)
            
            # Validation
            if self.global_step % self.args.eval_steps == 0:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
            
            # Save checkpoint
            if self.global_step % self.args.save_steps == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        avg_loss = total_loss / num_batches
        avg_aux_loss = total_aux_loss / num_batches
        
        logger.info(f"📊 Epoch {self.epoch} completed:")
        logger.info(f"   Average Loss: {avg_loss:.4f}")
        logger.info(f"   Average Aux Loss: {avg_aux_loss:.4f}")
        
        # Log routing statistics
        if routing_stats:
            self._log_routing_stats(routing_stats)
        
        return avg_loss, avg_aux_loss
    
    def validate(self):
        """Validate the model"""
        self.moe_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                outputs = self.moe_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        
        logger.info(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        
        self.moe_model.train()
        return avg_val_loss
    
    def _collect_routing_stats(self, input_ids, routing_stats):
        """Collect routing statistics"""
        with torch.no_grad():
            layer_stats = self.moe_model.get_routing_stats(input_ids)
            
            for layer_name, stats in layer_stats.items():
                for expert_name, expert_stats in stats.items():
                    routing_stats[layer_name][expert_name] += expert_stats['count']
    
    def _log_routing_stats(self, routing_stats):
        """Log routing statistics"""
        logger.info("🔀 Routing Statistics:")
        
        for layer_name, layer_stats in routing_stats.items():
            total_tokens = sum(layer_stats.values())
            logger.info(f"   {layer_name}:")
            
            for expert_name, count in layer_stats.items():
                ratio = count / total_tokens if total_tokens > 0 else 0
                logger.info(f"     {expert_name}: {count} tokens ({ratio:.1%})")
    
    def _log_training_step(self, main_loss, aux_loss):
        """Log training step"""
        logger.info(f"Step {self.global_step}: loss={main_loss:.4f}, aux_loss={aux_loss:.4f}")
    
    def _save_checkpoint(self, checkpoint_name):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.moe_model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        
        # Save training state
        checkpoint_info = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_name}")
    
    def train(self):
        """Main training loop"""
        logger.info(f"🚀 Starting MoE router training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            
            logger.info(f"📚 Starting epoch {epoch + 1}/{self.args.epochs}")
            
            # Train one epoch
            train_loss, aux_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch}")
            
            logger.info(f"✅ Epoch {epoch + 1} completed: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save final model
        self._save_checkpoint("final")
        
        logger.info("🎉 MoE router training completed!")


def create_multi_domain_dataset(domains, tokenizer, max_length, train_ratio=0.9):
    """Create combined multi-domain dataset"""
    
    logger.info(f"📚 Creating multi-domain dataset for: {domains}")
    
    all_datasets = create_domain_datasets(domains, tokenizer, max_length)
    
    train_datasets = []
    val_datasets = []
    
    for domain in domains:
        dataset = all_datasets[domain]['train']
        
        # Split into train/val
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        
        logger.info(f"   {domain}: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Combine all datasets
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    logger.info(f"📊 Combined dataset: {len(combined_train)} train, {len(combined_val)} val")
    
    return combined_train, combined_val


def train_moe_router(args):
    """Train MoE router"""
    
    logger.info("🚀 Starting MoE router training")
    print_gpu_memory_summary("Initial", args.gpu_id)
    
    # Set device
    torch.cuda.set_device(args.gpu_id)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='fp16' if args.fp16 else 'no',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Load tokenizer
    logger.info("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create MoE model
    logger.info("🔄 Creating LoRA-based MoE model...")
    moe_model = create_moe_model(
        base_model_path=args.base_model,
        domain_adapter_paths=args.domain_adapter_paths,
        aux_loss_weight=args.aux_loss_weight
    )
    
    print_gpu_memory_summary("MoE model created", args.gpu_id)
    
    # Freeze everything except routers
    logger.info("🔒 Freezing all parameters except routers...")
    router_params = 0
    frozen_params = 0
    
    for name, param in moe_model.named_parameters():
        if 'router' in name:
            param.requires_grad = True
            router_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    logger.info(f"🔓 Router parameters: {router_params:,}")
    logger.info(f"🔒 Frozen parameters: {frozen_params:,}")
    
    # Create dataset
    logger.info("📚 Creating multi-domain dataset...")
    train_dataset, val_dataset = create_multi_domain_dataset(
        domains=args.domains,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=4,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in moe_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    
    # Create trainer
    trainer = MoETrainer(
        moe_model=moe_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        args=args
    )
    
    # Save configuration
    config = {
        'base_model': args.base_model,
        'domains': args.domains,
        'domain_model_paths': args.domain_model_paths,
        'router_parameters': router_params,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'aux_loss_weight': args.aux_loss_weight,
        'timestamp': datetime.now().isoformat()
    }
    save_config(config, os.path.join(args.output_dir, 'training_config.json'))
    
    # Start training
    trainer.train()
    
    print_gpu_memory_summary("Training completed", args.gpu_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoE router")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="google/gemma-3-4b-it",
                       help="Base model path")
    parser.add_argument("--domain_adapter_paths", type=str, required=True, nargs='+',
                       help="Paths to domain LoRA adapters: medical_path law_path math_path code_path")
    parser.add_argument("--domains", type=str, nargs='+', 
                       default=["medical", "law", "math", "code"],
                       help="Domain names")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--aux_loss_weight", type=float, default=0.01,
                       help="Auxiliary loss weight")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # System arguments
    parser.add_argument("--output_dir", type=str, default="./moe_router_models",
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
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation interval")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint interval")
    parser.add_argument("--routing_stats_steps", type=int, default=100,
                       help="Routing statistics collection interval")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse domain adapter paths
    if len(args.domain_adapter_paths) != len(args.domains):
        raise ValueError(f"Number of domain adapter paths ({len(args.domain_adapter_paths)}) must match number of domains ({len(args.domains)})")
    
    args.domain_adapter_paths = dict(zip(args.domains, args.domain_adapter_paths))
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train MoE router
    train_moe_router(args)
    
    logger.info("🎉 MoE router training completed!")


if __name__ == "__main__":
    main()

