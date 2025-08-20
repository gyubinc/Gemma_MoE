#!/usr/bin/env python3
"""
Integrated Domain Training and Evaluation Pipeline
OOP design with separate Train and Evaluation classes
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any

from domain_trainer import DomainTrainer
from domain_evaluator import DomainEvaluator
from utils import (
    setup_logging, load_config, validate_environment, 
    check_data_availability, create_experiment_dir, 
    log_experiment_config, print_gpu_memory_summary
)

logger = logging.getLogger(__name__)

class DomainPipeline:
    """Complete domain training and evaluation pipeline"""
    
    def __init__(self, config_path: str, domain: str, experiment_name: str, gpu_id: int = 0):
        self.domain = domain
        self.gpu_id = gpu_id
        self.experiment_name = experiment_name
        
        # Setup GPU environment
        self.setup_gpu()
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Create experiment directory
        self.experiment_dir = create_experiment_dir(experiment_name)
        
        # Setup logging
        log_file = os.path.join(self.experiment_dir, f"{domain}_pipeline.log")
        setup_logging(log_file=log_file)
        
        # Initialize components
        self.trainer = DomainTrainer(self.config, domain, self.experiment_dir)
        self.evaluator = DomainEvaluator(self.config, domain, self.experiment_dir)
        
        logger.info(f"🚀 Initialized {domain} domain pipeline")
        logger.info(f"📁 Experiment directory: {self.experiment_dir}")
    
    def setup_gpu(self):
        """Setup GPU environment"""
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            logger.info(f"🚀 Setting CUDA_VISIBLE_DEVICES to {self.gpu_id}")
        else:
            logger.info(f"🚀 CUDA_VISIBLE_DEVICES already set to {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    def validate_environment(self) -> bool:
        """Validate environment and data availability"""
        logger.info("🔍 Validating environment...")
        
        # Environment validation
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        # Data availability check
        availability = check_data_availability([self.domain])
        if not availability.get(self.domain, False):
            logger.error(f"❌ Data not available for {self.domain} domain")
            return False
        
        logger.info("✅ Environment validation passed")
        return True
    
    def run_training(self) -> str:
        """Execute training phase"""
        logger.info(f"🎯 Starting {self.domain} training phase...")
        
        try:
            adapter_path = self.trainer.train()
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"📁 Model saved at: {adapter_path}")
            return adapter_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            self.trainer.cleanup()
    
    def run_evaluation(self, adapter_path: str) -> Dict[str, Any]:
        """Execute evaluation phase"""
        logger.info(f"🔍 Starting {self.domain} evaluation phase...")
        
        try:
            # Get evaluation sample size from config
            eval_samples = self.config.get('domain_configs', {}).get(self.domain, {}).get('eval_samples', 1000)
            
            evaluation_results = self.evaluator.evaluate(adapter_path, max_samples=eval_samples)
            logger.info(f"✅ Evaluation completed successfully!")
            logger.info(f"📊 Final accuracy: {evaluation_results['accuracy']:.4f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
        finally:
            self.evaluator.cleanup()
    
    def run_full_pipeline(self, skip_training: bool = False, adapter_path: str = None) -> Dict[str, Any]:
        """Execute complete training and evaluation pipeline"""
        logger.info(f"🚀 Starting full {self.domain} pipeline...")
        print_gpu_memory_summary("Pipeline start")
        
        # Log experiment configuration
        log_experiment_config(self.config, self.experiment_dir)
        
        results = {
            "domain": self.domain,
            "experiment_dir": self.experiment_dir,
            "training_completed": False,
            "evaluation_completed": False,
            "adapter_path": None,
            "evaluation_results": None
        }
        
        try:
            # Training phase
            if not skip_training:
                adapter_path = self.run_training()
                results["training_completed"] = True
                results["adapter_path"] = adapter_path
            else:
                if not adapter_path:
                    raise ValueError("adapter_path must be provided when skip_training=True")
                logger.info(f"⏭️  Skipping training, using existing adapter: {adapter_path}")
                results["adapter_path"] = adapter_path
            
            # Evaluation phase
            evaluation_results = self.run_evaluation(results["adapter_path"])
            results["evaluation_completed"] = True
            results["evaluation_results"] = evaluation_results
            
            # Final summary
            logger.info("🎉 Full pipeline completed successfully!")
            logger.info(f"📊 Domain: {self.domain}")
            logger.info(f"📁 Adapter: {results['adapter_path']}")
            logger.info(f"📊 Accuracy: {results['evaluation_results']['accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
        finally:
            print_gpu_memory_summary("Pipeline end")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Domain Training and Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--domain", type=str, required=True, 
                       choices=["medical", "law", "math", "code"], 
                       help="Domain to train and evaluate")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--experiment_name", type=str, default="domain_pipeline", help="Experiment name")
    parser.add_argument("--skip_training", action="store_true", help="Skip training phase")
    parser.add_argument("--adapter_path", type=str, help="Path to existing adapter (when skipping training)")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = DomainPipeline(
            config_path=args.config,
            domain=args.domain,
            experiment_name=args.experiment_name,
            gpu_id=args.gpu_id
        )
        
        # Validate environment
        if not pipeline.validate_environment():
            sys.exit(1)
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            skip_training=args.skip_training,
            adapter_path=args.adapter_path
        )
        
        print("🎉 Pipeline completed successfully!")
        print(f"📁 Results saved in: {results['experiment_dir']}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
