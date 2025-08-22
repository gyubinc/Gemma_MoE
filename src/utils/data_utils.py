#!/usr/bin/env python3
"""
Data utilities for Qwen-MoE project
"""

import os
import json
import logging
from typing import Dict, Any, List
from ..configs.domains import domain_manager

logger = logging.getLogger(__name__)

def check_data_availability(domains: List[str] = None) -> Dict[str, bool]:
    """Check if data files are available for domains"""
    if domains is None:
        domains = domain_manager.get_available_domains()
    
    availability = {}
    for domain in domains:
        availability[domain] = domain_manager.check_data_availability(domain)[domain]
    
    return availability

def analyze_dataset_samples(domains: List[str] = None, max_samples: int = 3) -> Dict[str, Dict]:
    """Analyze sample data from domains"""
    if domains is None:
        domains = domain_manager.get_available_domains()
    
    analysis = {}
    
    for domain in domains:
        print(f"\n📊 {domain.upper()} DOMAIN DATA SAMPLE")
        print("="*60)
        
        domain_config = domain_manager.get_domain(domain)
        
        # Check test file
        test_file = domain_config.get_file_path('test')
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            print(f"📁 File: {test_file}")
            print(f"📊 Total samples: {len(data)}")
            
            # Show first few samples
            for i in range(min(max_samples, len(data))):
                sample = data[i]
                print(f"\n🔍 Sample {i+1}:")
                print(f"Keys: {list(sample.keys())}")
                
                if domain == 'medical':
                    print(f"Question: {sample.get('question', 'N/A')[:200]}...")
                    print(f"Options: {sample.get('options', 'N/A')}")
                    print(f"Correct: {sample.get('correct', 'N/A')}")
                    print(f"Subject: {sample.get('subject', 'N/A')}")
                    
                elif domain == 'law':
                    print(f"Context: {sample.get('context', 'N/A')[:200]}...")
                    print(f"Endings: {sample.get('endings', 'N/A')}")
                    print(f"Correct idx: {sample.get('correct_ending_idx', 'N/A')}")
                    
                elif domain == 'math':
                    print(f"Question: {sample.get('question', 'N/A')[:200]}...")
                    print(f"Answer: {sample.get('answer', 'N/A')[:200]}...")
            
            analysis[domain] = {
                "file": test_file,
                "total_samples": len(data),
                "status": "available"
            }
        else:
            print(f"❌ Test file not found: {test_file}")
            analysis[domain] = {
                "file": test_file,
                "status": "missing"
            }
    
    return analysis

def get_domain_statistics(domain: str) -> Dict[str, Any]:
    """Get comprehensive statistics for a domain"""
    return domain_manager.get_domain_stats(domain)

def validate_environment():
    """Validate the training environment"""
    issues = []
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        issues.append("CUDA is not available")
    else:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            issues.append("No GPU devices found")
        else:
            print(f"Found {gpu_count} GPU device(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check required packages
    required_packages = [
        "transformers", "peft", "accelerate", "datasets", 
        "torch", "numpy", "yaml"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Package '{package}' is not installed")
    
    # Check data directories
    if not os.path.exists("data"):
        issues.append("Data directory not found")
    
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Environment validation passed")
        return True
