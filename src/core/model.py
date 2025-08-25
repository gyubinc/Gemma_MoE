#!/usr/bin/env python3
"""
Model management for Qwen-MoE project
Handles base model and adapter loading
"""

import torch
import os
import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model management"""
    
    def __init__(self, model_name: str = None, device: str = None):
        config = get_config()
        self.model_name = model_name or config.get('model.base_model', "Qwen/Qwen3-4B-Instruct-2507")
        self.device = device or config.get_cuda_device()
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load base model with minimal memory settings"""
        logger.info(f"Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Apply generation config from settings
        self._apply_generation_config()
        
        self.model.eval()
        logger.info("Base model loaded successfully")
        
        return self.model, self.tokenizer
    
    def load_model_with_adapter(self, adapter_path: str) -> Tuple[PeftModel, AutoTokenizer]:
        """Load model with LoRA adapter"""
        logger.info(f"Loading model with adapter: {adapter_path}")
        
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        # Load base model first
        self.load_base_model()
        
        # Load adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Apply generation config from settings
        self._apply_generation_config()
        
        self.model.eval()
        logger.info("Model with adapter loaded successfully")
        
        return self.model, self.tokenizer
    
    def _apply_generation_config(self):
        """Apply generation configuration from settings"""
        if self.model is None:
            return
        
        config = get_config()
        generation_config = config.get('model.generation', {})
        
        # Apply generation settings
        if hasattr(self.model, 'generation_config'):
            if generation_config.get('temperature') is not None:
                self.model.generation_config.temperature = generation_config['temperature']
            if generation_config.get('top_p') is not None:
                self.model.generation_config.top_p = generation_config['top_p']
            if generation_config.get('top_k') is not None:
                self.model.generation_config.top_k = generation_config['top_k']
            if generation_config.get('max_new_tokens') is not None:
                self.model.generation_config.max_new_tokens = generation_config['max_new_tokens']
            if generation_config.get('do_sample') is not None:
                self.model.generation_config.do_sample = generation_config['do_sample']
            if generation_config.get('pad_token_id') is not None:
                self.model.generation_config.pad_token_id = generation_config['pad_token_id']
            if generation_config.get('eos_token_id') is not None:
                self.model.generation_config.eos_token_id = generation_config['eos_token_id']
        
        logger.info("Generation config applied from settings")
    
    def get_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Get current model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_base_model() or load_model_with_adapter() first.")
        
        return self.model, self.tokenizer
    
    def save_adapter(self, adapter_path: str):
        """Save current adapter"""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        if hasattr(self.model, 'save_pretrained'):
            os.makedirs(adapter_path, exist_ok=True)
            self.model.save_pretrained(adapter_path)
            logger.info(f"Adapter saved to: {adapter_path}")
        else:
            logger.warning("Current model is not a PEFT model, cannot save adapter")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text from prompt"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def clear_memory(self):
        """Clear model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model memory cleared")

# Global model manager instance
model_manager = ModelManager()
