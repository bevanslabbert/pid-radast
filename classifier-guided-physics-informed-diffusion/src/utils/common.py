import torch
import gc
import os

def clear_gpu_memory():
    # 1. Clear Python's garbage collector
    gc.collect()
    
    # 2. Clear PyTorch's internal cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA Memory Cleared. Current VRAM Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
