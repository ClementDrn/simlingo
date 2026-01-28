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
GPU compatibility smoke test for InternVL2 (Qwen2 backend).

Updated to use the new gpu_compatibility patch abstraction (get_gpu_compatibility_patch).
Focus: ensure that on Volta (or pre-Ampere) we (a) load with safe kwargs, (b) apply the
per-model patch, and (c) confirm a dummy forward works without FlashAttention.
"""
import torch
from contextlib import suppress


def _summarize_flash_flags(model):
    """Return counts of modules still advertising flash attention usage (for diagnostics)."""
    attrs = ("use_flash_attn", "use_flash_attn_2", "flash_attn", "flash_attention")
    remaining = {a: 0 for a in attrs}
    for m in model.modules():
        for a in attrs:
            if hasattr(m, a) and getattr(m, a):
                remaining[a] += 1
    return {k: v for k, v in remaining.items() if v > 0}


def test_model_loading():
    print("=" * 60)
    print("TESTING GPU COMPATIBILITY (InternVL2)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    capability = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {capability}")

    from simlingo_training.utils.gpu_compatibility import (
        get_gpu_compatibility_patch,
        get_safe_model_kwargs,
    )

    patch = get_gpu_compatibility_patch()
    is_volta = patch is not None and not patch.is_flash_attention_supported()

    if patch is None:
        print("⚠️  No GPU patch instance (CPU or unknown device) – continuing in degraded mode.")
    else:
        patch.apply_globally()
        print("✅ Global GPU compatibility patch applied (idempotent)")

    try:
        from transformers import AutoModel, AutoProcessor, AutoConfig
    except Exception as e:  # pragma: no cover
        print(f"❌ transformers import failed: {e}")
        return False

    model_name = "OpenGVLab/InternVL2-1B"
    print(f"Loading model: {model_name}")

    # Processor first (trust_remote_code required for InternVL2)
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Processor loaded")
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")
        return False

    # Pre-load config so we can nudge attention flags *before* weights load
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    for flag in ("use_flash_attn", "use_flash_attn_2"):
        if hasattr(config, flag):
            setattr(config, flag, False)
    # Nested LLM config (InternVL passes this along)
    if hasattr(config, "llm_config") and config.llm_config is not None:
        llm_cfg = config.llm_config
        if hasattr(llm_cfg, "attn_implementation"):
            llm_cfg.attn_implementation = "eager"
        for flag in ("use_flash_attn", "use_flash_attn_2"):
            if hasattr(llm_cfg, flag):
                setattr(llm_cfg, flag, False)

    # Compose safe kwargs (forces fp16 + eager on Volta)
    base_kwargs = {"trust_remote_code": True}
    if is_volta:
        base_kwargs["torch_dtype"] = torch.float16
    safe_kwargs = get_safe_model_kwargs(base_kwargs)

    try:
        model = AutoModel.from_pretrained(model_name, config=config, **safe_kwargs)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return False

    # Apply per‑model patch if available (it may enforce eager & disable flags)
    if patch is not None:
        try:
            patch.apply_to_model(model, verbose=True)
            print("✅ Per-model patch applied")
        except Exception as e:
            print(f"⚠️  apply_to_model failed (continuing): {e}")

    # Defensive final enforcement (PyTorch side)
    with suppress(Exception):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    # Diagnostics: list any lingering flash flags
    lingering = _summarize_flash_flags(model)
    if lingering:
        print(f"⚠️  Lingering flash-related flags still True: {lingering}")
    else:
        print("✅ No lingering flash-related module flags detected")

    # Dummy forward (text only) through language model submodule if present
    dummy_text = "This is a test"
    try:
        inputs = processor(dummy_text, return_tensors="pt")
    except Exception as e:
        print(f"❌ Processor failed to tokenize: {e}")
        return False

    device = next(model.parameters()).device
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Try to find an LM forward entrypoint
    forward_target = getattr(model, "language_model", model)

    try:
        with torch.no_grad():
            _ = forward_target(**inputs)
        print("✅ Forward pass successful — compatibility confirmed")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = test_model_loading()
    print("\n" + "=" * 60)
    if ok:
        print("✅ GPU compatibility test PASSED")
    else:
        print("❌ GPU compatibility test FAILED")
    print("=" * 60)
