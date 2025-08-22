#!/usr/bin/env python3
"""
Evaluation script for Qwen-MoE project
"""

import argparse
import logging
import sys
import os
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

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
            logging.FileHandler('evaluation.log')
        ]
    )

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate domain-specific models")
    parser.add_argument("--domain", 
                       choices=domain_manager.get_available_domains(),
                       help="Domain to evaluate (if not specified, evaluate all)")
    parser.add_argument("--adapter-path", 
                       help="Path to LoRA adapter (optional, uses base model if not specified)")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum samples to evaluate")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    # Print GPU memory before evaluation
    print_gpu_memory_summary("Before evaluation")
    
    try:
        if args.domain:
            # Evaluate single domain
            logger.info(f"Evaluating {args.domain} domain")
            results = evaluate_domain(
                domain=args.domain,
                adapter_path=args.adapter_path,
                max_samples=args.max_samples
            )
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation completed for {args.domain} domain")
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
            
        else:
            # Evaluate all domains
            logger.info("Evaluating all domains")
            results = evaluate_all_domains(
                adapter_path=args.adapter_path,
                max_samples=args.max_samples
            )
            
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
            
            logger.info("Evaluation completed for all domains")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
