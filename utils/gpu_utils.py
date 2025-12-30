#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Utility functions.

This module provides utilities for GPU device management and
memory optimization.
"""

import torch
import os


def get_device(gpu_index=None):
    """
    Get the appropriate device (GPU or CPU).
    
    Args:
        gpu_index: int, optional - specific GPU index to use
                   If None, uses GPU with most free memory or CPU if no GPU
    
    Returns:
        torch.device: The selected device
    
    Example:
        >>> device = get_device()
        >>> tensor = torch.randn(100, 100, device=device)
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    if gpu_index is not None:
        if gpu_index < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_index}')
        else:
            raise ValueError(f"GPU {gpu_index} not available. Only {torch.cuda.device_count()} GPUs found.")
    
    # Auto-select GPU with most free memory
    return torch.device(f'cuda:{select_best_gpu()}')


def select_best_gpu(min_free_gb=2.0):
    """
    Select the GPU with the most free memory.
    
    Args:
        min_free_gb: float - minimum required free memory in GB
    
    Returns:
        int: GPU index with most free memory
    
    Raises:
        RuntimeError: If no GPU has enough free memory
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available")
    
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA devices found")
    
    best_gpu = 0
    best_free = 0
    min_free_bytes = min_free_gb * 1024 ** 3
    
    for i in range(n_gpus):
        try:
            free, total = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_gpu = i
        except Exception:
            pass
    
    if best_free < min_free_bytes:
        print(f"Warning: Best GPU has only {best_free / 1024**3:.1f}GB free "
              f"(threshold: {min_free_gb}GB)")
    
    return best_gpu


def get_gpu_memory_info(gpu_index=None):
    """
    Get memory information for a GPU.
    
    Args:
        gpu_index: int, optional - GPU index (default: current device)
    
    Returns:
        dict: Memory information including free, used, and total
    """
    if not torch.cuda.is_available():
        return {'available': False}
    
    if gpu_index is None:
        gpu_index = torch.cuda.current_device()
    
    try:
        free, total = torch.cuda.mem_get_info(gpu_index)
        used = total - free
        
        return {
            'available': True,
            'gpu_index': gpu_index,
            'free_gb': free / 1024**3,
            'used_gb': used / 1024**3,
            'total_gb': total / 1024**3,
            'utilization_pct': (used / total) * 100,
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


def clear_gpu_cache():
    """
    Clear CUDA cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_gpu_memory_fraction(fraction=0.9, gpu_index=None):
    """
    Limit GPU memory usage to a fraction of total.
    
    Args:
        fraction: float - fraction of memory to allow (0.0-1.0)
        gpu_index: int, optional - GPU index
    """
    if not torch.cuda.is_available():
        return
    
    if gpu_index is None:
        gpu_index = torch.cuda.current_device()
    
    try:
        total = torch.cuda.get_device_properties(gpu_index).total_memory
        limit = int(total * fraction)
        torch.cuda.set_per_process_memory_fraction(fraction, gpu_index)
    except Exception as e:
        print(f"Warning: Could not set memory fraction: {e}")


def setup_gpu(gpu_index=None, enable_tf32=True, enable_cudnn_benchmark=True):
    """
    Set up GPU for optimal performance.
    
    Args:
        gpu_index: int, optional - GPU to use
        enable_tf32: bool - enable TF32 for faster matrix operations
        enable_cudnn_benchmark: bool - enable cuDNN auto-tuner
    
    Returns:
        torch.device: The configured device
    """
    device = get_device(gpu_index)
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        
        if enable_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        
        if enable_cudnn_benchmark:
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    
    return device


if __name__ == "__main__":
    # Test the functions
    print("GPU Utilities Test")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        
        for i in range(torch.cuda.device_count()):
            info = get_gpu_memory_info(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Free: {info['free_gb']:.1f} GB")
            print(f"  Used: {info['used_gb']:.1f} GB")
            print(f"  Total: {info['total_gb']:.1f} GB")
            print(f"  Utilization: {info['utilization_pct']:.1f}%")
        
        device = setup_gpu()
        print(f"\nUsing device: {device}")
    else:
        print("CUDA not available, using CPU")
        device = get_device()
        print(f"Using device: {device}")

