#!/usr/bin/env python3
"""
GPU memory management utilities
"""

import torch
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

def print_gpu_memory_summary(stage: str = ""):
    """Print detailed GPU memory usage summary"""
    if not torch.cuda.is_available():
        print(f"[GPU] {stage}: CUDA not available")
        return
    
    try:
        # Initialize CUDA if needed
        if torch.cuda.device_count() == 0:
            torch.cuda.init()
        
        # Get GPU info using PyTorch
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            free = total - reserved
            
            print(f"[GPU{i}] {stage}: mem: alloc={allocated:.2f} GiB, "
                  f"reserved={reserved:.2f} GiB, max_alloc={max_allocated:.2f} GiB, "
                  f"free={free:.2f} GiB/ total={total:.2f} GiB")
    
    except Exception as e:
        print(f"[GPU] {stage}: Error getting memory info - {e}")
        # Fallback to basic info
        try:
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"[GPU{i}] {stage}: Total memory: {total:.2f} GiB")
        except Exception as e2:
            print(f"[GPU] {stage}: Fallback also failed - {e2}")

def print_system_memory_summary():
    """Print system memory usage"""
    memory = psutil.virtual_memory()
    print(f"[SYSTEM] Memory: used={memory.used/1024**3:.2f} GiB, "
          f"available={memory.available/1024**3:.2f} GiB, "
          f"total={memory.total/1024**3:.2f} GiB")

def clear_gpu_memory():
    """Clear GPU memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_optimal_batch_size(model_size_gb: float, gpu_memory_gb: float = 46.0) -> dict:
    """Calculate optimal batch sizes for A6000 46GB VRAM"""
    # Conservative memory allocation (leave 20% for overhead)
    available_memory = gpu_memory_gb * 0.8
    
    # Base memory per sample (rough estimate)
    base_memory_per_sample = model_size_gb * 0.1  # Conservative estimate
    
    # Calculate batch sizes
    max_batch_size = int(available_memory / base_memory_per_sample)
    
    # Conservative batch sizes for different scenarios
    return {
        "per_device_batch_size": min(4, max_batch_size),
        "gradient_accumulation_steps": max(8, max_batch_size // 4),
        "eval_batch_size": min(2, max_batch_size // 2)
    }
