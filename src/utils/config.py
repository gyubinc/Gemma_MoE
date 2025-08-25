#!/usr/bin/env python3
"""
Configuration utilities for Qwen-MoE project
"""

import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_environment()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _setup_environment(self):
        """Setup environment variables from config"""
        # Set CUDA_VISIBLE_DEVICES
        cuda_devices = self.config.get('gpu', {}).get('cuda_visible_devices', '3')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={cuda_devices}")
    
    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration"""
        return self.config.get('gpu', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration"""
        return self.config.get('lora', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get('output', {})
    
    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific configuration"""
        return self.config.get('domains', {}).get(domain, {})
    
    def get_cuda_device(self) -> str:
        """Get CUDA device string"""
        return self.config.get('gpu', {}).get('device', 'cuda:0')
    
    def get_cuda_visible_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES value"""
        return self.config.get('gpu', {}).get('cuda_visible_devices', '3')

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get global config manager instance"""
    return config_manager

def setup_cuda_environment():
    """Setup CUDA environment from config"""
    cuda_devices = config_manager.get_cuda_visible_devices()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    logger.info(f"CUDA environment setup: CUDA_VISIBLE_DEVICES={cuda_devices}")

def apply_generation_config(model):
    """Apply generation configuration to model"""
    if model is None or not hasattr(model, 'generation_config'):
        return
    
    generation_config = config_manager.get('model.generation', {})
    
    # Apply generation settings
    for key, value in generation_config.items():
        if value is not None and hasattr(model.generation_config, key):
            setattr(model.generation_config, key, value)
    
    logger.info("Generation config applied to model")
