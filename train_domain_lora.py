#!/usr/bin/env python3
"""
Domain-specific LoRA Fine-tuning for Qwen-MoE
각 도메인별로 Qwen 모델을 LoRA로 훈련하는 핵심 스크립트
"""

import argparse
import logging
import sys
import os
import time
import json
from typing import Dict, Any

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.core import train_domain
from src.configs import domain_manager
from src.utils import validate_environment, print_gpu_memory_summary, clear_gpu_memory, setup_cuda_environment

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('domain_training.log')
        ]
    )

def main():
    """Main function for domain-specific LoRA training"""
    parser = argparse.ArgumentParser(description="Train domain-specific LoRA adapters")
    parser.add_argument("--domain", required=True, 
                       choices=domain_manager.get_available_domains(),
                       help="Domain to train (medical, law, math, code)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to use for training (default: all)")
    parser.add_argument("--output-dir", default="domain_models",
                       help="Output directory for trained models")
    parser.add_argument("--device", default=None,
                       help="Device to use for training (default: from config)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Starting LoRA training for {args.domain} domain")
    logger.info(f"Configuration: max_samples={args.max_samples}, epochs={args.epochs}, "
                f"batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # Setup CUDA environment from config
    setup_cuda_environment()
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        return 1
    
    # Check data availability
    from src.utils import check_data_availability
    availability = check_data_availability([args.domain])
    
    if not availability[args.domain]:
        logger.error(f"❌ Data not available for {args.domain} domain")
        return 1
    
    # Print GPU memory before training
    print_gpu_memory_summary("Before training")
    
    start_time = time.time()
    
    try:
        # Train domain
        results = train_domain(
            domain=args.domain,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ {args.domain.upper()} domain training completed successfully")
        logger.info(f"   Training time: {training_time:.2f} seconds")
        logger.info(f"   Final loss: {results['train_loss']:.4f}")
        logger.info(f"   Adapter saved to: {results['adapter_path']}")
        
        # Save training summary
        summary = {
            "domain": args.domain,
            "training_time": training_time,
            "results": results,
            "config": {
                "max_samples": args.max_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate
            }
        }
        
        summary_path = os.path.join(args.output_dir, f"{args.domain}_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed for {args.domain} domain: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
