"""
GPU compatibility utilities for handling different GPU architectures.
Particularly for handling FlashAttention compatibility with older GPUs like V100.
"""

import torch
import os
import warnings
from typing import Optional, Tuple


def get_gpu_capability() -> Optional[Tuple[int, int]]:
    """Get the compute capability of the current GPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return None


def is_flash_attention_supported() -> bool:
    """Check if the current GPU supports FlashAttention (requires Ampere or newer)."""
    capability = get_gpu_capability()
    if capability is None:
        return False
    
    # FlashAttention requires compute capability 8.0 or higher (Ampere architecture)
    return capability[0] >= 8


def setup_v100_compatibility():
    """Set up environment variables and warnings for V100 GPU compatibility."""
    capability = get_gpu_capability()
    
    if capability is None:
        print(f"\033[93mCUDA not available, using CPU mode.\033[0m")
        return
    
    print(f"\033[93mDetected GPU capability: {capability}\033[0m")
    
    if capability[0] < 8:
        print(f"\033[91mGPU capability {capability} is below Ampere architecture (8.0).\033[0m")
        print(f"\033[91mFlashAttention not supported. Configuring fallback attention implementation.\033[0m")
        
        # Set environment variables to disable FlashAttention
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        os.environ["FLASH_ATTENTION_DISABLE"] = "1"
        os.environ["FORCE_FLASH_ATTENTION"] = "0"
        
        # Filter out flash attention warnings
        warnings.filterwarnings("ignore", message=".*flash_attention.*")
        warnings.filterwarnings("ignore", message=".*FlashAttention.*")
        warnings.filterwarnings("ignore", message=".*flash-attn.*")
        
        return False
    else:
        print(f"\033[92mGPU supports FlashAttention. Using optimal attention implementation.\033[0m")
        return True


def get_safe_model_kwargs(base_kwargs: dict = None) -> dict:
    """Get safe model loading kwargs that work with both V100 and newer GPUs."""
    kwargs = base_kwargs.copy() if base_kwargs else {}
    
    if not is_flash_attention_supported():
        # For older GPUs, use eager attention implementation
        kwargs.update({
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
        })
    
    return kwargs


def safe_model_loading_context():
    """Context manager for safe model loading with V100 compatibility."""
    class SafeLoadingContext:
        def __init__(self):
            self.flash_supported = None
            
        def __enter__(self):
            self.flash_supported = setup_v100_compatibility()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def is_flash_supported(self):
            return self.flash_supported
    
    return SafeLoadingContext()
