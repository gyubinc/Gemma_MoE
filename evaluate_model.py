#!/usr/bin/env python3
"""
Model Evaluation for Qwen-MoE
임의의 Qwen 모델 또는 MoE Qwen 모델을 특정 데이터셋에 맞게 평가하는 핵심 스크립트
"""

import argparse
import logging
import sys
import os
import json
from typing import Dict, Any, List

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.core import evaluate_domain, evaluate_all_domains
from src.configs import domain_manager
from src.utils import validate_environment, print_gpu_memory_summary

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('model_evaluation.log')
        ]
    )

def main():
    """Main function for model evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Qwen models on domain datasets")
    parser.add_argument("--model-type", required=True,
                       choices=["base", "lora", "moe"],
                       help="Type of model to evaluate (base, lora, moe)")
    parser.add_argument("--model-path", 
                       help="Path to model or adapter (required for lora/moe)")
    parser.add_argument("--domain", 
                       choices=domain_manager.get_available_domains(),
                       help="Domain to evaluate (if not specified, evaluate all)")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum samples to evaluate")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type in ["lora", "moe"] and not args.model_path:
        print("❌ Error: --model-path is required for lora and moe model types")
        return 1
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 Starting {args.model_type.upper()} model evaluation")
    if args.model_path:
        logger.info(f"Model path: {args.model_path}")
    logger.info(f"Domain: {args.domain if args.domain else 'all'}")
    logger.info(f"Max samples: {args.max_samples}")
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        return 1
    
    # Print GPU memory before evaluation
    print_gpu_memory_summary("Before evaluation")
    
    try:
        if args.domain:
            # Evaluate single domain
            logger.info(f"Evaluating {args.domain} domain")
            
            if args.model_type == "base":
                results = evaluate_domain(
                    domain=args.domain,
                    adapter_path=None,  # Use base model
                    max_samples=args.max_samples
                )
            else:
                results = evaluate_domain(
                    domain=args.domain,
                    adapter_path=args.model_path,
                    max_samples=args.max_samples
                )
            
            # Add model info to results
            results["model_type"] = args.model_type
            results["model_path"] = args.model_path
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Evaluation completed for {args.domain} domain")
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
            logger.info(f"Results saved to: {args.output}")
            
        else:
            # Evaluate all domains
            logger.info("Evaluating all domains")
            
            if args.model_type == "base":
                results = evaluate_all_domains(
                    adapter_path=None,  # Use base model
                    max_samples=args.max_samples
                )
            else:
                results = evaluate_all_domains(
                    adapter_path=args.model_path,
                    max_samples=args.max_samples
                )
            
            # Add model info to results
            results["model_type"] = args.model_type
            results["model_path"] = args.model_path
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            print("\n" + "="*80)
            print("📊 EVALUATION SUMMARY")
            print("="*80)
            
            for domain in domain_manager.get_available_domains():
                if domain in results and "error" not in results[domain]:
                    accuracy = results[domain]["accuracy"]
                    samples = results[domain]["total_samples"]
                    correct = results[domain]["correct_predictions"]
                    print(f"🔬 {domain.upper():<10}: {accuracy:.2%} ({correct}/{samples})")
                else:
                    error_msg = results.get(domain, {}).get("error", "Unknown error")
                    print(f"🔬 {domain.upper():<10}: ❌ ERROR - {error_msg}")
            
            if "summary" in results:
                summary = results["summary"]
                print(f"\n📈 OVERALL ACCURACY: {summary['overall_accuracy']:.2%} "
                      f"({summary['total_correct']}/{summary['total_samples']})")
            
            logger.info("✅ Evaluation completed for all domains")
            logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
