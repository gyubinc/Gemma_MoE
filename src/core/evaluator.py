#!/usr/bin/env python3
"""
Unified evaluator for Qwen-MoE project
Supports evaluation for all domains with centralized configuration
"""

import torch
import os
import logging
import json
import re
from typing import Dict, Any, List, Tuple
from ..configs.domains import domain_manager
from .dataset import UnifiedDataset
from .model import model_manager

logger = logging.getLogger(__name__)

class UnifiedEvaluator:
    """Unified evaluator supporting all domains"""
    
    def __init__(self, domain: str = None, device: str = "cuda:0"):
        self.domain = domain
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_model(self, adapter_path: str = None):
        """Load model for evaluation"""
        if adapter_path:
            self.model, self.tokenizer = model_manager.load_model_with_adapter(adapter_path)
        else:
            self.model, self.tokenizer = model_manager.load_base_model()
    
    def evaluate_domain(self, domain: str, max_samples: int = 1000, split: str = 'test') -> Dict[str, Any]:
        """Evaluate a specific domain"""
        logger.info(f"Evaluating {domain} domain")
        
        # Get domain configuration
        domain_config = domain_manager.get_domain(domain)
        
        # Load test dataset
        test_dataset = UnifiedDataset(
            tokenizer=self.tokenizer,
            domain=domain,
            split=split,
            max_samples=max_samples
        )
        
        # Evaluate
        results = self._evaluate_dataset(test_dataset, domain)
        
        # Save results
        output_dir = f"experiments/{domain}_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{domain}_evaluation_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {output_path}")
        return results
    
    def evaluate_all_domains(self, domains: List[str] = None, max_samples: int = 1000) -> Dict[str, Any]:
        """Evaluate all domains"""
        if domains is None:
            domains = domain_manager.get_available_domains()
        
        logger.info(f"Evaluating all domains: {domains}")
        
        results = {}
        total_samples = 0
        total_correct = 0
        
        for domain in domains:
            try:
                domain_results = self.evaluate_domain(domain, max_samples)
                results[domain] = domain_results
                
                if "error" not in domain_results:
                    total_samples += domain_results['total_samples']
                    total_correct += domain_results['correct_predictions']
                    
            except Exception as e:
                logger.error(f"Error evaluating {domain} domain: {e}")
                results[domain] = {"error": str(e)}
        
        # Calculate overall accuracy
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Add summary
        results["summary"] = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "domains_evaluated": len([r for r in results.values() if "error" not in r])
        }
        
        return results
    
    def _evaluate_dataset(self, dataset: UnifiedDataset, domain: str) -> Dict[str, Any]:
        """Evaluate a dataset"""
        self.model.eval()
        predictions = []
        references = []
        
        logger.info(f"Evaluating {len(dataset)} samples for {domain} domain")
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}/{len(dataset)}")
                
                sample = dataset[i]
                input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                
                # Find assistant start position
                assistant_start = self._find_assistant_start(input_ids[0])
                assistant_start = min(assistant_start, input_ids.shape[1] - 1)
                
                # Generate prediction
                max_new_tokens = 50 if domain == 'math' else 20
                outputs = self.model.generate(
                    input_ids=input_ids[:, :assistant_start],
                    attention_mask=attention_mask[:, :assistant_start],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                    repetition_penalty=1.0
                )
                
                # Decode prediction and reference
                prediction = self.tokenizer.decode(outputs[0][assistant_start:], skip_special_tokens=True)
                reference = self.tokenizer.decode(sample["labels"][sample["labels"] != -100], skip_special_tokens=True)
                
                predictions.append(prediction.strip())
                references.append(reference.strip())
        
        # Calculate accuracy
        accuracy, correct, total = self._calculate_accuracy(predictions, references, domain)
        
        return {
            "domain": domain,
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "predictions": predictions[:10],  # Save first 10 for inspection
            "references": references[:10],
            "evaluation_type": domain_manager.get_domain(domain).evaluation_type
        }
    
    def _find_assistant_start(self, input_ids) -> int:
        """Find the position where assistant response should start"""
        # Method 1: Look for <|im_start|>assistant\n pattern
        for i in range(len(input_ids) - 2):
            if (input_ids[i] == self.tokenizer.encode("<|im_start|")[0] and 
                input_ids[i+1] == self.tokenizer.encode("assistant")[0] and
                input_ids[i+2] == self.tokenizer.encode("\n")[0]):
                return i + 3
        
        # Method 2: If not found, look for just "assistant\n"
        for i in range(len(input_ids) - 1):
            if (input_ids[i] == self.tokenizer.encode("assistant")[0] and
                input_ids[i+1] == self.tokenizer.encode("\n")[0]):
                return i + 2
        
        # Method 3: If still not found, use the last 20% of tokens
        fallback_position = int(len(input_ids) * 0.8)
        logger.warning(f"Using fallback assistant start position {fallback_position}")
        return fallback_position
    
    def _calculate_accuracy(self, predictions: List[str], references: List[str], domain: str) -> Tuple[float, int, int]:
        """Calculate accuracy based on domain-specific criteria"""
        correct = 0
        total = len(predictions)
        
        domain_config = domain_manager.get_domain(domain)
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if domain_config.evaluation_type == "multiple_choice":
                pred_clean = self._clean_multiple_choice_prediction(pred, domain_config.num_choices)
                ref_clean = ref.strip().upper()
            elif domain_config.evaluation_type == "math_answer":
                pred_clean = self._extract_math_answer(pred)
                ref_clean = self._extract_math_answer(ref)
            else:  # exact_match
                pred_clean = pred.strip().lower()
                ref_clean = ref.strip().lower()
            
            if pred_clean == ref_clean:
                correct += 1
            
            # Log first few samples for debugging
            if i < 3:
                logger.info(f"Sample {i}: pred='{pred}' -> '{pred_clean}', ref='{ref}' -> '{ref_clean}', correct={pred_clean == ref_clean}")
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total
    
    def _clean_multiple_choice_prediction(self, prediction: str, num_choices: int) -> str:
        """Clean multiple choice prediction to extract answer choice"""
        valid_choices = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D, E
        
        # Remove common prefixes and suffixes
        pred = prediction.strip()
        
        # Create pattern for valid choices
        choices_pattern = '|'.join(valid_choices)
        
        # Extract answer choice (A, B, C, D, E)
        patterns = [
            rf'answer[:\s]*([{choices_pattern}])',
            rf'([{choices_pattern}])[\.\)]',
            rf'option[:\s]*([{choices_pattern}])',
            rf'choice[:\s]*([{choices_pattern}])',
            rf'([{choices_pattern}])\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, pred, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern found, try to extract single letter
        letters = re.findall(rf'\b[{choices_pattern}]\b', pred.upper())
        if letters:
            return letters[0]
        
        return pred.upper()
    
    def _extract_math_answer(self, text: str) -> str:
        """Extract final numerical answer from math solution text"""
        # Remove common prefixes and suffixes
        text = text.strip()
        
        # Look for patterns like "#### 18", "Answer: 18", "The answer is 18"
        patterns = [
            r'####\s*(\d+(?:\.\d+)?)',  # GSM8K format: #### number
            r'answer[:\s]*(\d+(?:\.\d+)?)',
            r'the\s+answer\s+is[:\s]*(\d+(?:\.\d+)?)',
            r'final\s+answer[:\s]*(\d+(?:\.\d+)?)',
            r'result[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*$'  # Number at the end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern found, try to extract the last number
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return text.strip()
    
    def cleanup(self):
        """Cleanup evaluation resources"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Evaluation cleanup completed")

def evaluate_domain(domain: str, adapter_path: str = None, max_samples: int = 1000) -> Dict[str, Any]:
    """Convenience function to evaluate a domain"""
    evaluator = UnifiedEvaluator(domain)
    
    try:
        evaluator.load_model(adapter_path)
        results = evaluator.evaluate_domain(domain, max_samples)
        evaluator.cleanup()
        return results
    except Exception as e:
        evaluator.cleanup()
        raise

def evaluate_all_domains(domains: List[str] = None, adapter_path: str = None, max_samples: int = 1000) -> Dict[str, Any]:
    """Convenience function to evaluate all domains"""
    evaluator = UnifiedEvaluator()
    
    try:
        evaluator.load_model(adapter_path)
        results = evaluator.evaluate_all_domains(domains, max_samples)
        evaluator.cleanup()
        return results
    except Exception as e:
        evaluator.cleanup()
        raise
