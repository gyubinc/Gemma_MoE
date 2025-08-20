#!/usr/bin/env python3
"""
Domain-specific Model Evaluator Class
OOP design for modular evaluation
"""

import os
import torch
import gc
import logging
from typing import Dict, Any

from dataset import create_domain_datasets
from utils import (
    evaluate_domain_model, load_model_for_evaluation, 
    save_evaluation_results, clear_gpu_memory
)

logger = logging.getLogger(__name__)

class DomainEvaluator:
    """Domain-specific Model Evaluator"""
    
    def __init__(self, config: Dict[str, Any], domain: str, experiment_dir: str):
        self.config = config
        self.domain = domain
        self.experiment_dir = experiment_dir
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.test_dataset = None
        
    def load_trained_model(self, adapter_path: str):
        """Load trained model with LoRA adapter"""
        logger.info(f"🔍 Loading trained {self.domain} model for evaluation...")
        
        self.model, self.tokenizer = load_model_for_evaluation(
            self.config['model']['name'],
            adapter_path=adapter_path,
            device="cuda:0"
        )
        
        logger.info(f"✅ {self.domain} model loaded successfully")
    
    def load_test_dataset(self, max_samples: int = 1000):
        """Load test dataset for evaluation"""
        logger.info(f"📚 Loading {self.domain} test dataset...")
        
        _, self.test_dataset = create_domain_datasets(
            domain=self.domain,
            tokenizer=self.tokenizer,
            max_length=int(self.config['training']['max_length']),
            max_samples=max_samples
        )
        
        logger.info(f"📊 Test dataset loaded: {len(self.test_dataset):,} samples")
    
    def evaluate(self, adapter_path: str, max_samples: int = 1000) -> Dict[str, Any]:
        """Execute evaluation process"""
        logger.info(f"🔍 Starting {self.domain} model evaluation...")
        
        try:
            # Load trained model
            self.load_trained_model(adapter_path)
            
            # Load test dataset
            self.load_test_dataset(max_samples)
            
            # Run evaluation
            logger.info("📊 Running evaluation...")
            evaluation_results = evaluate_domain_model(
                model=self.model,
                tokenizer=self.tokenizer,
                test_dataset=self.test_dataset,
                domain=self.domain,
                max_samples=max_samples,
                device="cuda:0"
            )
            
            # Save results
            eval_path = os.path.join(self.experiment_dir, f"{self.domain}_evaluation_results.json")
            save_evaluation_results(evaluation_results, eval_path)
            
            logger.info(f"📊 {self.domain.title()} evaluation completed!")
            logger.info(f"📊 Accuracy: {evaluation_results['accuracy']:.4f}")
            logger.info(f"📁 Results saved to: {eval_path}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Error during {self.domain} evaluation: {e}")
            raise
        finally:
            self.cleanup()
    
    def evaluate_multiple_samples(self, adapter_path: str, sample_sizes: list = [100, 500, 1000]) -> Dict[str, Any]:
        """Evaluate with multiple sample sizes"""
        logger.info(f"🔍 Running multi-sample evaluation for {self.domain}")
        
        results = {}
        
        for samples in sample_sizes:
            logger.info(f"📊 Evaluating with {samples} samples...")
            result = self.evaluate(adapter_path, max_samples=samples)
            results[f"samples_{samples}"] = result
            
            # Short break between evaluations
            self.cleanup()
        
        # Save combined results
        combined_path = os.path.join(self.experiment_dir, f"{self.domain}_multi_evaluation_results.json")
        save_evaluation_results(results, combined_path)
        
        return results
    
    def compare_with_baseline(self, adapter_path: str, baseline_path: str = None) -> Dict[str, Any]:
        """Compare trained model with baseline"""
        logger.info(f"🔍 Comparing {self.domain} model with baseline...")
        
        # Evaluate trained model
        trained_results = self.evaluate(adapter_path)
        
        # If baseline path provided, evaluate baseline too
        baseline_results = None
        if baseline_path:
            logger.info("📊 Evaluating baseline model...")
            baseline_results = self.evaluate(baseline_path)
        
        comparison = {
            "domain": self.domain,
            "trained_model": trained_results,
            "baseline_model": baseline_results,
            "improvement": None
        }
        
        if baseline_results:
            improvement = trained_results['accuracy'] - baseline_results['accuracy']
            comparison["improvement"] = improvement
            logger.info(f"📊 Accuracy improvement: {improvement:.4f}")
        
        # Save comparison
        comparison_path = os.path.join(self.experiment_dir, f"{self.domain}_comparison_results.json")
        save_evaluation_results(comparison, comparison_path)
        
        return comparison
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            del self.test_dataset
        
        clear_gpu_memory()
        gc.collect()
        
        logger.info("🧹 Evaluator resources cleaned up")
