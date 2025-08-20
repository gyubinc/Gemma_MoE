#!/usr/bin/env python3
"""
Domain-specific dataset loader for Qwen-MoE project
Supports Medical, Law, Math, and Code domains
"""

import torch
from torch.utils.data import Dataset
import json
import os
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DomainDataset(Dataset):
    """Domain-specific dataset class for Qwen-MoE"""
    
    def __init__(self, tokenizer, domain: str, max_length: int = 512, split: str = 'train', max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain = domain
        self.data = self._load_domain_data(domain, split, max_samples)
    
    def _load_domain_data(self, domain: str, split: str, max_samples: int) -> List[Dict[str, str]]:
        """Load domain-specific data from local JSON files"""
        data_path = f"data/{domain}"
        
        if domain == "medical":
            return self._load_medical_data(data_path, split, max_samples)
        elif domain == "law":
            return self._load_law_data(data_path, split, max_samples)
        elif domain == "math":
            return self._load_math_data(data_path, split, max_samples)
        elif domain == "code":
            return self._load_code_data(data_path, split, max_samples)
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def _load_medical_data(self, data_path: str, split: str, max_samples: int) -> List[Dict[str, str]]:
        """Load medical data (MedMCQA)"""
        file_path = os.path.join(data_path, f"medmcqa_{split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Medical data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        data = []
        for item in dataset:
            question = item['question']
            choices = [item['opa'], item['opb'], item['opc'], item['opd']]
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            
            # English instruction for medical domain
            instruction = f"Answer the following medical question:\n\n{question}\n\n{choices_text}\n\nAnswer:"
            response = chr(65 + item['cop'])  # 0,1,2,3 -> A,B,C,D
            
            data.append({
                'instruction': instruction,
                'response': response
            })
            
            if max_samples and len(data) >= max_samples:
                break
        
        return data
    
    def _load_law_data(self, data_path: str, split: str, max_samples: int) -> List[Dict[str, str]]:
        """Load law data (LegalBench case_hold)"""
        file_path = os.path.join(data_path, f"case_hold_{split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Law data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        data = []
        for item in dataset:
            question = item['formatted_question']
            correct_answer = item['correct_ending']
            
            # English instruction for law domain
            instruction = f"Analyze the following legal case and select the correct holding:\n\n{question}"
            response = correct_answer
            
            data.append({
                'instruction': instruction,
                'response': response
            })
            
            if max_samples and len(data) >= max_samples:
                break
        
        return data
    
    def _load_math_data(self, data_path: str, split: str, max_samples: int) -> List[Dict[str, str]]:
        """Load math data (GSM8K)"""
        # GSM8K only has train and test, use test for validation
        if split == 'validation':
            split = 'test'
            
        file_path = os.path.join(data_path, f"gsm8k_{split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Math data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        data = []
        for item in dataset:
            question = item['question']
            answer = item['answer']
            
            # English instruction for math domain
            instruction = f"Solve the following math problem step by step:\n\n{question}"
            response = answer
            
            data.append({
                'instruction': instruction,
                'response': response
            })
            
            if max_samples and len(data) >= max_samples:
                break
        
        return data
    
    def _load_code_data(self, data_path: str, split: str, max_samples: int) -> List[Dict[str, str]]:
        """Load code data (CodeXGLUE)"""
        file_path = os.path.join(data_path, f"codexglue_{split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Code data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        data = []
        for item in dataset:
            # CodeXGLUE structure may vary, handle common fields
            if 'nl' in item and 'code' in item:
                instruction = f"Generate Python code for the following requirement:\n\n{item['nl']}"
                response = item['code']
            elif 'question' in item and 'answer' in item:
                instruction = f"Write code to solve the following problem:\n\n{item['question']}"
                response = item['answer']
            elif 'formatted_question' in item and 'code' in item:
                instruction = item['formatted_question']
                response = item['code']
            elif 'docstring' in item and 'code' in item:
                instruction = f"Write Python code for the following docstring:\n\n{item['docstring']}"
                response = item['code']
            else:
                # Skip invalid items
                continue
            
            data.append({
                'instruction': instruction,
                'response': response
            })
            
            if max_samples and len(data) >= max_samples:
                break
        
        return data
    
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
        
        # Labels for causal LM training - only train on assistant response
        labels = encoding["input_ids"].clone()
        
        # Find the start of assistant response
        assistant_start = None
        input_ids = encoding["input_ids"][0]
        
        # Find "<|im_start|>assistant\n" token sequence
        for i in range(len(input_ids) - 2):
            if (input_ids[i] == self.tokenizer.encode("<|im_start|>")[0] and 
                input_ids[i+1] == self.tokenizer.encode("assistant")[0] and
                input_ids[i+2] == self.tokenizer.encode("\n")[0]):
                assistant_start = i + 3
                break
        
        if assistant_start is not None:
            # Set labels to -100 for everything before assistant response (context only)
            labels[0, :assistant_start] = -100
            
            # Find the end of assistant response (before <|im_end|>)
            assistant_end = None
            for i in range(assistant_start, len(input_ids)):
                if input_ids[i] == self.tokenizer.encode("<|im_end|>")[0]:
                    assistant_end = i
                    break
            
            if assistant_end is not None:
                # Set labels to -100 for everything after assistant response
                labels[0, assistant_end:] = -100
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
        }

def create_domain_datasets(domain: str, tokenizer, max_length: int = 512, max_samples: int = None) -> Tuple[DomainDataset, DomainDataset]:
    """Create training and evaluation datasets for a specific domain"""
    
    # Training dataset
    train_dataset = DomainDataset(
        tokenizer=tokenizer,
        domain=domain,
        max_length=max_length,
        split='train',
        max_samples=max_samples  # Use provided max_samples
    )
    
    # Evaluation dataset (try validation first, then test)
    eval_dataset = None
    for eval_split in ['validation', 'test']:
        try:
            eval_dataset = DomainDataset(
                tokenizer=tokenizer,
                domain=domain,
                max_length=max_length,
                split=eval_split,
                max_samples=1000  # Limit evaluation samples
            )
            logger.info(f"Using {eval_split} split for {domain} domain evaluation")
            break
        except Exception as e:
            logger.warning(f"Failed to load {eval_split} split for {domain} domain: {e}")
            continue
    
    if eval_dataset is None:
        raise ValueError(f"No evaluation split available for {domain} domain")
    
    print(f"📊 {domain.upper()} Dataset:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Eval: {len(eval_dataset):,} samples")
    
    return train_dataset, eval_dataset

def get_domain_info(domain: str) -> Dict[str, Any]:
    """Get information about a specific domain dataset"""
    summary_path = f"data/{domain}/summary.json"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    else:
        return {"error": f"Summary not found for {domain} domain"}

if __name__ == "__main__":
    # Test the dataset loading
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test each domain
    for domain in ["medical", "law", "math", "code"]:
        try:
            print(f"\n{'='*50}")
            print(f"Testing {domain.upper()} domain")
            print(f"{'='*50}")
            
            train_ds, eval_ds = create_domain_datasets(domain, tokenizer, max_length=512)
            print(f"Sample data: {train_ds[0]}")
            
        except Exception as e:
            print(f"Error loading {domain} domain: {e}")