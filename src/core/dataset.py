#!/usr/bin/env python3
"""
Unified dataset for Qwen-MoE project
Supports all domains with centralized configuration
"""

import torch
from torch.utils.data import Dataset
import json
import os
from typing import Dict, Any, List
import logging
from ..configs.domains import domain_manager

logger = logging.getLogger(__name__)

class UnifiedDataset(Dataset):
    """Unified dataset class supporting all domains"""
    
    def __init__(self, tokenizer, domain: str, split: str = 'train', 
                 max_samples: int = None, max_length: int = None):
        self.tokenizer = tokenizer
        self.domain = domain
        self.split = split
        self.max_samples = max_samples
        
        # Get domain configuration
        self.domain_config = domain_manager.get_domain(domain)
        self.max_length = max_length or self.domain_config.max_length
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {domain} domain ({split} split)")
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load data for the specified domain and split"""
        file_path = self.domain_config.get_file_path(self.split)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Process data based on domain
        processed_data = []
        
        for item in raw_data:
            try:
                # Preprocess item based on domain
                processed_item = self._preprocess_item(item)
                
                # Format prompt and response based on domain
                instruction = self._format_instruction(processed_item)
                response = self._format_response(processed_item)
                
                processed_data.append({
                    'instruction': instruction,
                    'response': response
                })
                
                if self.max_samples and len(processed_data) >= self.max_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
        
        return processed_data
    
    def _preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess item based on domain-specific requirements"""
        # 모든 도메인에서 정답이 있으므로 -1 처리 로직 제거
        return item
    
    def _format_instruction(self, item: Dict[str, Any]) -> str:
        """Format instruction based on domain"""
        if self.domain == "medical":
            return domain_manager.format_prompt(
                self.domain,
                question=item['question'],
                options=item['options']
            )
        
        elif self.domain == "law":
            return domain_manager.format_prompt(
                self.domain,
                context=item['context'],
                endings=item['endings']
            )
        
        elif self.domain == "math":
            return domain_manager.format_prompt(
                self.domain,
                question=item['question'],
                options=item['options']
            )
        
        elif self.domain == "code":
            return domain_manager.format_prompt(
                self.domain,
                question=item['question']
            )
        
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
    
    def _format_response(self, item: Dict[str, Any]) -> str:
        """Format response based on domain"""
        if self.domain == "medical":
            return domain_manager.format_response(
                self.domain,
                correct_option=item['correct_option']
            )
        
        elif self.domain == "law":
            return domain_manager.format_response(
                self.domain,
                correct_ending_idx=item['correct_ending_idx'],
                endings=item['endings']
            )
        
        elif self.domain == "math":
            return domain_manager.format_response(
                self.domain,
                correct_option=item['correct_option']
            )
        
        elif self.domain == "code":
            return domain_manager.format_response(
                self.domain,
                code=item['code']
            )
        
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format for Qwen instruction-following format
        full_text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Keep tensors on CPU, let trainer handle device placement
        
        # Labels for causal LM training - only train on assistant response
        labels = encoding["input_ids"].clone()
        
        # Find the start of assistant response
        assistant_start = self._find_assistant_start(encoding["input_ids"][0])
        
        # Set labels to -100 for everything before assistant response (context only)
        labels[0, :assistant_start] = -100
        
        # Find the end of assistant response (before <|im_end|>)
        assistant_end = self._find_assistant_end(encoding["input_ids"][0], assistant_start)
        
        if assistant_end is not None:
            # Set labels to -100 for everything after assistant response
            labels[0, assistant_end:] = -100
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
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
    
    def _find_assistant_end(self, input_ids, start_pos: int) -> int:
        """Find the end of assistant response"""
        for i in range(start_pos, len(input_ids)):
            if input_ids[i] == self.tokenizer.encode("<|im_end|>")[0]:
                return i
        return None

def create_datasets(domain: str, tokenizer, max_samples: int = None) -> Dict[str, UnifiedDataset]:
    """Create training and evaluation datasets for a domain"""
    datasets = {}
    
    # Training dataset
    try:
        datasets['train'] = UnifiedDataset(
            tokenizer=tokenizer,
            domain=domain,
            split='train',
            max_samples=max_samples
        )
    except Exception as e:
        logger.error(f"Failed to load training dataset for {domain}: {e}")
        raise
    
    # Evaluation dataset (use test split)
    try:
        datasets['test'] = UnifiedDataset(
            tokenizer=tokenizer,
            domain=domain,
            split='test',
            max_samples=1000  # Limit evaluation samples
        )
        logger.info(f"Using test split for {domain} domain evaluation")
    except Exception as e:
        logger.error(f"Failed to load test split for {domain} domain: {e}")
        raise
    
    # Print dataset info
    print(f"📊 {domain.upper()} Dataset:")
    print(f"  Train: {len(datasets['train']):,} samples")
    print(f"  Test: {len(datasets['test']):,} samples")
    
    return datasets
