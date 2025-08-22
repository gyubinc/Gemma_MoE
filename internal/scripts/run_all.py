#!/usr/bin/env python3
"""
Run all domains training sequentially
"""

import argparse
import logging
import sys
import os
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.core import train_domain
from src.configs import domain_manager
from src.utils import validate_environment, print_gpu_memory_summary, clear_gpu_memory

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('run_all_training.log')
        ]
    )

def main():
    """Main function to run all domains training"""
    parser = argparse.ArgumentParser(description="Train all domains sequentially")
    parser.add_argument("--domains", nargs="+", 
                       choices=domain_manager.get_available_domains(),
                       default=domain_manager.get_available_domains(),
                       help="Domains to train (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per domain")
    parser.add_argument("--output-dir", default="domain_models",
                       help="Output directory for trained models")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip domains that already have trained models")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting sequential training for domains: {args.domains}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    # Check data availability
    from src.utils import check_data_availability
    availability = check_data_availability(args.domains)
    
    missing_domains = [domain for domain, available in availability.items() if not available]
    if missing_domains:
        logger.error(f"Missing data for domains: {missing_domains}")
        return 1
    
    results = {}
    start_time = time.time()
    
    for i, domain in enumerate(args.domains):
        logger.info(f"\n{'='*80}")
        logger.info(f"Training domain {i+1}/{len(args.domains)}: {domain.upper()}")
        logger.info(f"{'='*80}")
        
        # Check if model already exists
        if args.skip_existing:
            adapter_path = os.path.join(args.output_dir, domain, "final_adapter")
            if os.path.exists(adapter_path):
                logger.info(f"Skipping {domain} - model already exists at {adapter_path}")
                results[domain] = {"status": "skipped", "adapter_path": adapter_path}
                continue
        
        # Print GPU memory before training
        print_gpu_memory_summary(f"Before {domain} training")
        
        try:
            # Train domain
            domain_start_time = time.time()
            domain_results = train_domain(
                domain=domain,
                max_samples=args.max_samples,
                output_dir=args.output_dir
            )
            domain_end_time = time.time()
            
            results[domain] = {
                "status": "completed",
                "results": domain_results,
                "training_time": domain_end_time - domain_start_time
            }
            
            logger.info(f"✅ {domain.upper()} training completed successfully")
            logger.info(f"   Training time: {domain_end_time - domain_start_time:.2f} seconds")
            logger.info(f"   Adapter saved to: {domain_results['adapter_path']}")
            
        except Exception as e:
            logger.error(f"❌ {domain.upper()} training failed: {e}")
            results[domain] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Clear GPU memory between domains
        clear_gpu_memory()
        
        # Small delay between domains
        time.sleep(2)
    
    # Print final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    
    completed = sum(1 for r in results.values() if r["status"] == "completed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    logger.info(f"Total domains: {len(args.domains)}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    # Save results
    import json
    results_file = os.path.join(args.output_dir, "training_summary.json")
    with open(results_file, 'w') as f:
        json.dump({
            "domains": args.domains,
            "results": results,
            "total_time": total_time,
            "summary": {
                "completed": completed,
                "skipped": skipped,
                "failed": failed
            }
        }, f, indent=2)
    
    logger.info(f"Training summary saved to: {results_file}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
