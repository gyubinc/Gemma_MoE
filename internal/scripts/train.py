#!/usr/bin/env python3
"""
Training script for Qwen-MoE project
"""

import argparse
import logging
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.core import train_domain
from src.configs import domain_manager
from src.utils import validate_environment, print_gpu_memory_summary

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train domain-specific model")
    parser.add_argument("--domain", required=True, 
                       choices=domain_manager.get_available_domains(),
                       help="Domain to train")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to use for training")
    parser.add_argument("--output-dir", default="domain_models",
                       help="Output directory for trained models")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training for {args.domain} domain")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    # Print GPU memory before training
    print_gpu_memory_summary("Before training")
    
    try:
        # Train domain
        results = train_domain(
            domain=args.domain,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
        
        logger.info(f"Training completed successfully for {args.domain} domain")
        logger.info(f"Results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
