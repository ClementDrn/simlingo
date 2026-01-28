#!/usr/bin/env python3
"""
Copyright 2025-2026 Clément Darne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Test script to verify GPU capability detection and Volta compatibility setup.
Run this script before running SimLingo evaluation to check if your GPU setup will work.
"""

import torch
import sys
import os

def test_gpu_compatibility():
    print("=" * 60)
    print("SimLingo Volta GPU Compatibility Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires CUDA.")
        return False
    
    print("✅ CUDA is available")
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    print(f"📊 Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_capability = torch.cuda.get_device_capability(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"\nGPU {i}: {gpu_name}")
        print(f"  Compute Capability: {gpu_capability}")
        print(f"  Memory: {memory_total:.1f} GB")
        
        # Check FlashAttention compatibility
        if gpu_capability[0] >= 8:
            print(f"  ✅ FlashAttention: Supported (Ampere or newer)")
        elif gpu_capability[0] == 7:
            print(f"  ⚠️  FlashAttention: NOT Supported (Volta architecture)")
            print(f"      Will use fallback attention implementation")
        else:
            print(f"  ⚠️  FlashAttention: NOT Supported (Pre-Volta architecture)")
            print(f"      Will use fallback attention implementation")
    
    # Test the utility function
    print("\n" + "=" * 40)
    print("Testing GPU compatibility utility...")
    
    try:
        # Add the simlingo_training directory to Python path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        simlingo_training_dir = os.path.join(script_dir, "simlingo_training")
        if os.path.exists(simlingo_training_dir):
            sys.path.insert(0, script_dir)
        
        from simlingo_training.utils.gpu_compatibility import (
            get_gpu_capability, is_flash_attention_supported, 
            setup_volta_compatibility, safe_model_loading_context
        )
        
        print("✅ GPU compatibility utility imported successfully")
        
        capability = get_gpu_capability()
        flash_supported = is_flash_attention_supported()
        
        print(f"📊 Detected capability: {capability}")
        print(f"🔥 FlashAttention supported: {flash_supported}")
        
        # Test setup function (it's already called on import, so this is just a verification)
        print("\n🔧 Volta compatibility already set up on import")
        
        # Test context manager
        print("🔧 Testing safe model loading context...")
        with safe_model_loading_context() as ctx:
            print(f"   Context flash support: {ctx.is_flash_supported()}")
        
        print("\n✅ All compatibility tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import GPU compatibility utility: {e}")
        print("   The utility should be available when running SimLingo.")
        return False
    
    except Exception as e:
        print(f"❌ Error testing GPU compatibility: {e}")
        return False

def test_environment_variables():
    print("\n" + "=" * 40)
    print("Environment Variables Check")
    print("=" * 40)
    
    flash_env_vars = [
        "DISABLE_FLASH_ATTN",
        "FLASH_ATTENTION_DISABLE", 
        "FORCE_FLASH_ATTENTION"
    ]
    
    for var in flash_env_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")

if __name__ == "__main__":
    print("Testing GPU compatibility for SimLingo...")
    
    success = test_gpu_compatibility()
    test_environment_variables()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 GPU compatibility test completed successfully!")
        print("\nYour system should be able to run SimLingo evaluation.")
        
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            print("\n⚠️  NOTE: Your GPU does not support FlashAttention.")
            print("   SimLingo will automatically use a fallback attention implementation.")
            print("   This may be slightly slower but should work correctly.")
    else:
        print("❌ GPU compatibility test failed!")
        print("\nThere may be issues running SimLingo evaluation.")
    
    print("=" * 60)