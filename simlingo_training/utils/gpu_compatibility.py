"""
GPU compatibility utilities for handling different GPU architectures.
Particularly for handling FlashAttention compatibility with Volta GPUs (V100, etc).
"""

import torch
import os
import warnings
from typing import Optional, Tuple
import enum


# ANSI colors enum
class AnsiColor(enum.Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


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
    # Volta (7.x) and earlier are not supported
    return capability[0] >= 8


def setup_volta_compatibility():
    """
    Set up environment variables and configurations for Volta GPU compatibility.
    This applies to all Volta architecture GPUs (V100, Tesla V100S, etc).
    """
    # Ensure this function only runs once
    if getattr(setup_volta_compatibility, "has_already_run", False):
        return
    setup_volta_compatibility.has_already_run = True

    print("Checking GPU compatibility...")
    
    if not torch.cuda.is_available():
        print(f"{AnsiColor.YELLOW}Warning: CUDA not available{AnsiColor.RESET}")
        return False
    
    capability = get_gpu_capability()
    gpu_name = torch.cuda.get_device_name()
    
    print(f"{AnsiColor.GREEN}- GPU: {gpu_name}{AnsiColor.RESET}")
    print(f"{AnsiColor.GREEN}- Compute Capability: {capability}{AnsiColor.RESET}")
    
    if capability[0] >= 8:
        print(f"{AnsiColor.GREEN}Ampere+ GPU detected - FlashAttention supported!{AnsiColor.RESET}")
        return True

    print(f"{AnsiColor.MAGENTA}Volta/Pre-Ampere GPU detected - Applying compatibility fixes...{AnsiColor.RESET}")

    # Set environment variables to disable FlashAttention
    env_vars = {
        "DISABLE_FLASH_ATTN": "1",
        "FLASH_ATTENTION_DISABLE": "1",
        "FORCE_FLASH_ATTENTION": "0",
        "TORCH_USE_FLASH_ATTENTION": "0",
        "TRANSFORMERS_USE_FLASH_ATTENTION": "0",
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
    
    # Suppress FlashAttention warnings
    warnings.filterwarnings("ignore", message=".*flash.*attention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*FlashAttention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*flash-attn.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*flash_attn.*", category=UserWarning)
    
    # Try to prevent flash attention at PyTorch level
    try:
        # Force PyTorch to use math attention instead of flash attention
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        print("Disabled FlashAttention in PyTorch SDP kernel")
    except Exception as e:
        print(f"{AnsiColor.YELLOW}Warning: Could not disable SDP flash attention: {e}{AnsiColor.RESET}")

    print(f"{AnsiColor.GREEN}Volta GPU compatibility setup complete!{AnsiColor.RESET}")
    return True


def get_safe_model_kwargs(base_kwargs: dict = None) -> dict:
    """Get safe model loading kwargs that work with both Volta and newer GPUs."""
    kwargs = base_kwargs.copy() if base_kwargs else {}
    kwargs["trust_remote_code"] = True
    
    if not is_flash_attention_supported():
        # For Volta and older GPUs, use eager attention implementation
        kwargs.update({
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
        })
        print("🔧 Using eager attention for Volta GPU compatibility")
    
    return kwargs


def safe_model_loading_context():
    """Context manager for safe model loading with Volta compatibility."""
    class SafeLoadingContext:
        def __init__(self):
            self.flash_supported = None
            
        def __enter__(self):
            self.flash_supported = setup_volta_compatibility()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def is_flash_supported(self):
            return self.flash_supported
    
    return SafeLoadingContext()

# Apply compatibility fixes immediately when this module is imported
setup_volta_compatibility()
