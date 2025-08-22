#!/usr/bin/env python3
"""
Domain configurations for Qwen-MoE project
Centralized domain management with OOP design
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os

@dataclass
class DomainConfig:
    """Configuration for a specific domain"""
    name: str
    data_path: str
    train_file: str
    validation_file: str
    test_file: str
    instruction_template: str
    response_format: str
    evaluation_type: str  # 'multiple_choice', 'math_answer', 'exact_match'
    max_length: int = 512
    num_choices: int = 4  # For multiple choice questions
    
    def get_file_path(self, split: str) -> str:
        """Get file path for specific split"""
        if split == 'train':
            return os.path.join(self.data_path, self.train_file)
        elif split == 'validation':
            return os.path.join(self.data_path, self.validation_file)
        elif split == 'test':
            return os.path.join(self.data_path, self.test_file)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def file_exists(self, split: str) -> bool:
        """Check if file exists for specific split"""
        return os.path.exists(self.get_file_path(split))

class DomainManager:
    """Centralized domain management"""
    
    def __init__(self):
        self.domains = self._initialize_domains()
    
    def _initialize_domains(self) -> Dict[str, DomainConfig]:
        """Initialize all domain configurations"""
        return {
            "medical": DomainConfig(
                name="medical",
                data_path="data/medical",
                train_file="medmcqa_train.json",
                validation_file="medmcqa_validation.json", 
                test_file="medmcqa_test.json",
                instruction_template="Answer the following medical question:\n\n{question}\n\n{choices}\n\nAnswer:",
                response_format="A",  # Single letter response
                evaluation_type="multiple_choice",
                num_choices=4
            ),
            "law": DomainConfig(
                name="law", 
                data_path="data/law",
                train_file="case_hold_train.json",
                validation_file="case_hold_validation.json",
                test_file="case_hold_test.json", 
                instruction_template="Legal case analysis:\n\n{context}\n\nSelect the appropriate holding:\n\n{choices}\n\nAnswer:",
                response_format="A",  # Single letter response
                evaluation_type="multiple_choice",
                num_choices=5
            ),
            "math": DomainConfig(
                name="math",
                data_path="data/math", 
                train_file="mathqa_train.json",
                validation_file="mathqa_validation.json",
                test_file="mathqa_test.json",
                instruction_template="Solve this math problem:\n\n{question}\n\n{choices}\n\nAnswer:",
                response_format="A",  # Single letter response
                evaluation_type="multiple_choice",
                num_choices=5,
                max_length=1024  # Math problems can be longer
            ),
            "code": DomainConfig(
                name="code",
                data_path="data/code",
                train_file="cybermetric_train.json", 
                validation_file="cybermetric_validation.json",
                test_file="cybermetric_test.json",
                instruction_template="Write code to solve the following problem:\n\n{question}\n\nCode:",
                response_format="code",  # Code response
                evaluation_type="exact_match",
                max_length=1024
            )
        }
    
    def get_domain(self, domain_name: str) -> DomainConfig:
        """Get domain configuration by name"""
        if domain_name not in self.domains:
            raise ValueError(f"Unknown domain: {domain_name}. Available: {list(self.domains.keys())}")
        return self.domains[domain_name]
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return list(self.domains.keys())
    
    def check_data_availability(self, domain_name: str = None) -> Dict[str, bool]:
        """Check data availability for domain(s)"""
        if domain_name:
            domain = self.get_domain(domain_name)
            return {
                domain_name: all([
                    domain.file_exists('train'),
                    domain.file_exists('test')
                ])
            }
        else:
            availability = {}
            for name, domain in self.domains.items():
                availability[name] = all([
                    domain.file_exists('train'),
                    domain.file_exists('test')
                ])
            return availability
    
    def get_domain_stats(self, domain_name: str) -> Dict[str, Any]:
        """Get statistics for a specific domain"""
        domain = self.get_domain(domain_name)
        stats = {
            "name": domain.name,
            "data_path": domain.data_path,
            "evaluation_type": domain.evaluation_type,
            "max_length": domain.max_length,
            "num_choices": domain.num_choices
        }
        
        # Check file sizes and sample counts
        for split in ['train', 'validation', 'test']:
            if domain.file_exists(split):
                file_path = domain.get_file_path(split)
                stats[f"{split}_file"] = file_path
                stats[f"{split}_size_mb"] = os.path.getsize(file_path) / (1024 * 1024)
                
                # Count samples (read first few lines to estimate)
                try:
                    import json
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        stats[f"{split}_samples"] = len(data)
                except:
                    stats[f"{split}_samples"] = "unknown"
            else:
                stats[f"{split}_file"] = "missing"
                stats[f"{split}_size_mb"] = 0
                stats[f"{split}_samples"] = 0
        
        return stats
    
    def format_prompt(self, domain_name: str, **kwargs) -> str:
        """Format prompt for a specific domain"""
        domain = self.get_domain(domain_name)
        
        if domain_name == "medical":
            question = kwargs.get('question', '')
            options = kwargs.get('options', [])
            choices = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
            return domain.instruction_template.format(question=question, choices=choices)
        
        elif domain_name == "law":
            context = kwargs.get('context', '')
            endings = kwargs.get('endings', [])
            choices = "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings)])
            return domain.instruction_template.format(context=context, choices=choices)
        
        elif domain_name == "math":
            question = kwargs.get('question', '')
            options = kwargs.get('options', [])
            choices = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
            return domain.instruction_template.format(question=question, choices=choices)
        
        elif domain_name == "code":
            question = kwargs.get('question', '')
            return domain.instruction_template.format(question=question)
        
        else:
            raise ValueError(f"Unknown domain: {domain_name}")
    
    def format_response(self, domain_name: str, **kwargs) -> str:
        """Format expected response for a specific domain"""
        domain = self.get_domain(domain_name)
        
        if domain_name == "medical":
            correct_option = kwargs.get('correct_option', 0)
            return chr(65 + correct_option)
        
        elif domain_name == "law":
            correct_idx = kwargs.get('correct_ending_idx', 0)
            return chr(65 + correct_idx)
        
        elif domain_name == "math":
            correct_option = kwargs.get('correct_option', 0)
            return chr(65 + correct_option)
        
        elif domain_name == "code":
            code = kwargs.get('code', '')
            return code
        
        else:
            raise ValueError(f"Unknown domain: {domain_name}")

# Global domain manager instance
domain_manager = DomainManager()
