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


# Abstract class
class GPUPatch:

    @staticmethod
    def get_gpu_capability() -> Optional[Tuple[int, int]]:
        """Get the compute capability of the current GPU."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
        return None
    
    @staticmethod
    def is_flash_attention_supported() -> bool:
        """Check if the current GPU supports FlashAttention (requires Ampere or newer)."""
        capability = GPUPatch.get_gpu_capability()
        if capability is None:
            return False
        
        # FlashAttention requires compute capability 8.0 or higher (Ampere architecture)
        # Volta (7.x) and earlier are not supported
        return capability[0] >= 8
    
    @staticmethod
    def print_gpu_info():
        """Print information about the current GPU and its FlashAttention support."""
        if not torch.cuda.is_available():
            print(f"{AnsiColor.YELLOW.value}Warning: CUDA not available!{AnsiColor.RESET.value}")
            return

        capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name()
        flash_attention_support = GPUPatch.is_flash_attention_supported()
        print(f"{AnsiColor.GREEN.value}Detected GPU:{AnsiColor.RESET.value}")
        print(f"{AnsiColor.GREEN.value}- GPU: {gpu_name}{AnsiColor.RESET.value}")
        print(f"{AnsiColor.GREEN.value}- Compute Capability: {capability}{AnsiColor.RESET.value}")
        print(f"{AnsiColor.GREEN.value}- FlashAttention Support: {flash_attention_support}{AnsiColor.RESET.value}")

    def get_preferred_compute_dtype(self) -> torch.dtype:
        """Return the preferred compute dtype for the current GPU."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def apply_globally(self, verbose = False) -> bool:
        """Apply the compatibility patch globally (e.g., set env vars, patch libs)."""
        raise NotImplementedError("Subclasses should implement this method.")

    def apply_to_model(self, model, verbose = False):
        """Apply compatibility fixes to a given model instance."""
        raise NotImplementedError("Subclasses should implement this method.")


# Patches specific to Ampere+ GPUs
class AmperePlusGPUPatch(GPUPatch):

    def get_preferred_compute_dtype(self) -> torch.dtype:
        """Return the preferred compute dtype for the current GPU."""
        return torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16

    def apply_globally(self, verbose = False) -> bool:
        pass

    def apply_to_model(self, model, verbose = False):
        return model


# Patches specific to Volta GPUs
class VoltaGPUPatch(GPUPatch):

    def get_preferred_compute_dtype(self) -> torch.dtype:
        """Return the preferred compute dtype for the current GPU."""
        return torch.float16

    def apply_globally(self, verbose = False) -> bool:
        """
        Set up environment variables and configurations for Volta GPU compatibility.
        This applies to all Volta architecture GPUs (V100, Tesla V100S, etc).
        """
        # Ensure this function only runs once per instance
        if getattr(self, "_global_patch_applied", False):
            if verbose:
                print(f"{AnsiColor.BLUE.value}[gpu_compatibility] Volta global patch already applied (skipping){AnsiColor.RESET.value}")
            return False
        self._global_patch_applied = True

        if not torch.cuda.is_available():
            print(f"{AnsiColor.YELLOW.value}Warning: CUDA not available! Aborting setup.{AnsiColor.RESET.value}")
            return False
        
        # Set environment variables to disable FlashAttention
        env_vars = {
            "DISABLE_FLASH_ATTN": "1",
            "FLASH_ATTENTION_DISABLE": "1",
            "FORCE_FLASH_ATTENTION": "0",
            "TORCH_USE_FLASH_ATTENTION": "0",
            "TRANSFORMERS_USE_FLASH_ATTENTION": "0",
            "TRANSFORMERS_FORCE_EAGER_ATTENTION": "1",  # Force eager attention in transformers
            "HF_USE_FLASH_ATTENTION": "0",  # Hugging Face specific
            # Common toggles seen in HF code / community recipes
            "FLASH_ATTENTION_2": "0",
            "USE_FLASH_ATTENTION_2": "0",
            "HF_USE_FLASH_ATTENTION_2": "0",
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
            print(f"{AnsiColor.YELLOW.value}Warning: Could not disable SDP flash attention: {e}{AnsiColor.RESET.value}")

        # Prevent Transformers from choosing FlashAttention by faking its unavailability
        try:
            import transformers.utils.import_utils as _hf_import_utils
            _orig_is_package_available = _hf_import_utils._is_package_available

            def _patched_is_package_available(pkg_name: str) -> bool:
                if pkg_name in ("flash_attn", "flash_attn_2"):
                    return False
                return _orig_is_package_available(pkg_name)

            _hf_import_utils._is_package_available = _patched_is_package_available
            print("Patched Transformers to ignore flash_attn availability")
        except Exception as e:
            print(f"{AnsiColor.YELLOW.value}Warning: Could not patch Transformers flash_attn availability: {e}{AnsiColor.RESET.value}")

        # Force Transformers to prefer eager attention and never toggle FA2
        try:
            import transformers.modeling_utils as _hf_modeling_utils
            PreTrainedModel = _hf_modeling_utils.PreTrainedModel

            # Patch _autoset_attn_implementation to force eager
            _orig_autoset = getattr(PreTrainedModel, "_autoset_attn_implementation", None)

            if _orig_autoset is not None:
                # Note: original is a @classmethod bound to the class; when retrieved above,
                # it is a bound function expecting (config, *args, **kwargs)
                def _patched_autoset(cls, config, *args, **kwargs):
                    try:
                        # Force eager and disable FA2 flags on the incoming config
                        if getattr(config, "attn_implementation", None) != "eager":
                            config.attn_implementation = "eager"
                        if hasattr(config, "use_flash_attn"):
                            config.use_flash_attn = False
                        if hasattr(config, "use_flash_attn_2"):
                            config.use_flash_attn_2 = False
                    except Exception:
                        pass
                    # Call original to let it fill any defaults
                    try:
                        result = _orig_autoset(config, *args, **kwargs)
                    except TypeError:
                        # Fallback in case original signature differs
                        result = config
                    # Re-enforce eager on the returned config
                    try:
                        if getattr(result, "attn_implementation", None) != "eager":
                            result.attn_implementation = "eager"
                        if hasattr(result, "use_flash_attn"):
                            result.use_flash_attn = False
                        if hasattr(result, "use_flash_attn_2"):
                            result.use_flash_attn_2 = False
                    except Exception:
                        pass
                    return result

                PreTrainedModel._autoset_attn_implementation = classmethod(_patched_autoset)

            # Patch FA2 enabling check to no-op
            if hasattr(PreTrainedModel, "_check_and_enable_flash_attn_2"):
                def _patched_check_fa2(cls, config, *args, **kwargs):
                    try:
                        if hasattr(config, "use_flash_attn"):
                            config.use_flash_attn = False
                        if hasattr(config, "use_flash_attn_2"):
                            config.use_flash_attn_2 = False
                    except Exception:
                        pass
                    return False

                PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(_patched_check_fa2)

            print("Patched Transformers attention hooks to force eager and disable FA2")
        except Exception as e:
            print(f"{AnsiColor.YELLOW.value}Warning: Could not patch Transformers attention hooks: {e}{AnsiColor.RESET.value}")

        # Ensure modules that reference _flash_attention_forward have a safe fallback
        try:
            import torch.nn.functional as F

            _warned_shapes = {"value": False}

            def _sdpa_fallback_forward(q, k, v, attention_mask=None, dropout_p: float = 0.0, is_causal: bool = False, *args, **kwargs):
                """
                Robust fallback for FlashAttention forward using PyTorch SDPA.
                Tries common layout orders and returns in the same layout as the input query.
                """
                def _call_sdpa(qt, kt, vt, mask):
                    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)

                # Record original layout assumption based on dims (B, H, S, D) vs (B, S, H, D)
                returned_perm = None
                out = None
                last_err = None
                try:
                    out = _call_sdpa(q, k, v, attention_mask)
                    returned_perm = "as_is"
                except Exception as e1:
                    last_err = e1
                    # Try switching (B, S, H, D) -> (B, H, S, D)
                    try:
                        if q.dim() == 4:
                            q2, k2, v2 = q, k, v
                            # Heuristic: if dim order likely (B, S, H, D), permute to (B, H, S, D)
                            q2 = q.permute(0, 2, 1, 3)
                            k2 = k.permute(0, 2, 1, 3)
                            v2 = v.permute(0, 2, 1, 3)
                            out = _call_sdpa(q2, k2, v2, attention_mask)
                            returned_perm = "bshd->bhsd"
                    except Exception as e2:
                        last_err = e2
                        # As a last resort, try treating inputs as (B*H, S, D) packed heads
                        try:
                            if q.dim() == 3:  # (B*H, S, D)
                                out = _call_sdpa(q, k, v, attention_mask)
                                returned_perm = "packed_heads"
                        except Exception as e3:
                            last_err = e3

                if out is None:
                    if not _warned_shapes["value"]:
                        print(f"{AnsiColor.YELLOW.value}Warning: SDPA fallback failed with shapes q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}; last error: {last_err}{AnsiColor.RESET.value}")
                        _warned_shapes["value"] = True
                    # Re-raise the last error to surface the issue
                    raise last_err

                # Return to original layout if we permuted
                if returned_perm == "bshd->bhsd":
                    # Caller likely expects (B, S, H, D); permute back
                    out = out.permute(0, 2, 1, 3)
                return out

            # Try to also alias in HF utility module when present (not required)
            try:
                import transformers.modeling_flash_attention_utils as _mfa_utils
                _mfa_utils._flash_attention_forward = _sdpa_fallback_forward
            except Exception:
                pass

            # Patch Qwen2 (primary target)
            try:
                import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod
                _qwen2_mod._flash_attention_forward = _sdpa_fallback_forward
            except Exception:
                pass
            # Patch LLaMA as a bonus (shared patterns sometimes reused)
            try:
                import transformers.models.llama.modeling_llama as _llama_mod
                _llama_mod._flash_attention_forward = _sdpa_fallback_forward
            except Exception:
                pass

            print("Installed SDPA-based fallback for _flash_attention_forward")
        except Exception as e:
            print(f"{AnsiColor.YELLOW.value}Warning: Could not install SDPA fallback: {e}{AnsiColor.RESET.value}")

        print(f"{AnsiColor.GREEN.value}Volta GPU compatibility setup complete!{AnsiColor.RESET.value}")
        return True

    def apply_to_model(self, model, verbose = False):
        """Apply compatibility fixes to a given model instance."""
        # Call static helpers
        self._force_eager_everywhere(model, verbose)
        self._install_sdpa_qwen2_fallback(verbose)
        self._disable_flash_attention_modules(model, verbose)
        return model

    @staticmethod
    def _disable_flash_attention_modules(root, verbose: bool = False):
        """Force-disable any module attributes that would trigger FlashAttention paths.

        Many community / remote code bases gate usage behind flags like `use_flash_attn`.
        This walks the module tree and sets those to False. Safe on all architectures.
        """
        count = 0
        visited = set()
        stack = [root]
        while stack:
            m = stack.pop()
            if id(m) in visited:
                continue
            visited.add(id(m))
            for attr in ("use_flash_attn", "use_flash_attention", "flash_attn", "flash_attention"):
                if hasattr(m, attr):
                    try:
                        if getattr(m, attr):
                            setattr(m, attr, False)
                            count += 1
                    except Exception:  # pragma: no cover
                        pass
            for child in getattr(m, "children", lambda: [])():
                if id(child) not in visited:
                    stack.append(child)
        if verbose and count:
            print(f"🔧 Disabled FlashAttention flags in {count} module(s)")
        
        return root

    @staticmethod
    def _install_sdpa_qwen2_fallback(verbose: bool = False):
        """(Idempotent) Install SDPA fallback for Qwen2 / LLaMA _flash_attention_forward symbol.

        This is exposed so that external scripts (tests, notebooks) can explicitly
        re-apply the fallback if another library overwrites it after import.
        """
        try:
            import torch.nn.functional as _F
            import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod
        except Exception:
            if verbose:
                print("[gpu_compatibility] Qwen2 modules not available; skipping SDPA fallback install")
            return False

        # Only wrap once
        if getattr(_qwen2_mod, '_simlingo_sdpa_fallback_installed', False):
            return True

        def _sdpa_fallback_forward(
            query_layer,
            key_layer,
            value_layer,
            *unused_args,
            attention_mask=None,
            is_causal=False,
            **unused_kwargs,
        ):
            try:
                return _F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
            except Exception as e:  # pragma: no cover - defensive
                if not hasattr(_sdpa_fallback_forward, "_warned"):
                    print(f"{AnsiColor.YELLOW.value}[gpu_compatibility] SDPA fallback failed: {e}{AnsiColor.RESET.value}")
                    _sdpa_fallback_forward._warned = True
                raise

        _qwen2_mod._flash_attention_forward = _sdpa_fallback_forward
        _qwen2_mod._simlingo_sdpa_fallback_installed = True
        if verbose:
            print("(gpu_compatibility) Installed SDPA fallback for Qwen2 _flash_attention_forward")
        # LLaMA optional
        try:  # pragma: no cover - optional
            import transformers.models.llama.modeling_llama as _llama_mod
            _llama_mod._flash_attention_forward = _sdpa_fallback_forward
        except Exception:
            pass
        return True

    @staticmethod
    def _force_eager_everywhere(root, verbose: bool = False):
        """Traverse a model hierarchy and enforce eager attention implementation.

        Safe to call on any model; silently ignores modules without the interface.
        """
        visited = set()
        stack = [root]
        while stack:
            m = stack.pop()
            if id(m) in visited:
                continue
            visited.add(id(m))
            # Direct API
            if hasattr(m, "set_attn_implementation"):
                try:
                    m.set_attn_implementation("eager")
                except Exception:  # pragma: no cover - best effort
                    pass
            # Config objects commonly used in HF
            cfg = getattr(m, "config", None)
            if cfg is not None:
                try:
                    if hasattr(cfg, "attn_implementation"):
                        cfg.attn_implementation = "eager"
                    if hasattr(cfg, "use_flash_attn"):
                        cfg.use_flash_attn = False
                    if hasattr(cfg, "use_flash_attn_2"):
                        cfg.use_flash_attn_2 = False
                except Exception:  # pragma: no cover
                    pass
            # Nested .model attribute
            inner = getattr(m, "model", None)
            if inner is not None and id(inner) not in visited:
                stack.append(inner)
            # Children
            for child in getattr(m, "children", lambda: [])():
                if id(child) not in visited:
                    stack.append(child)
        return root





def get_safe_model_kwargs(base_kwargs: dict = None) -> dict:
    """Get safe model loading kwargs that work with both Volta and newer GPUs."""
    kwargs = base_kwargs.copy() if base_kwargs else {}
    kwargs["trust_remote_code"] = True

    if not get_gpu_compatibility_patch().is_flash_attention_supported():
        # For Volta and older GPUs, use eager attention implementation
        # and prefer float16 over bfloat16 for better support on V100
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16
        kwargs["attn_implementation"] = "eager"
        print(f"Using eager attention with dtype={kwargs['torch_dtype']} for Volta GPU compatibility")
    
    return kwargs


def get_gpu_compatibility_patch():
    """Get the appropriate GPU compatibility patch instance based on current GPU."""
    # Ensure this function only runs once
    if getattr(get_gpu_compatibility_patch, "patch", None) is not None:
        return get_gpu_compatibility_patch.patch

    # Get GPU capability
    if not torch.cuda.is_available():
        print(f"{AnsiColor.YELLOW.value}Warning: CUDA not available{AnsiColor.RESET.value}")
        return None
    capability = GPUPatch.get_gpu_capability()
        
    # Select patch based on capability
    if capability is None:
        print(f"{AnsiColor.YELLOW.value}Warning: Unable to determine GPU capability{AnsiColor.RESET.value}")
        get_gpu_compatibility_patch.patch = None
    elif capability[0] >= 8:
        print(f"{AnsiColor.GREEN.value}Ampere+ GPU - no compatibility patch needed{AnsiColor.RESET.value}")
        get_gpu_compatibility_patch.patch = AmperePlusGPUPatch()
    elif capability[0] == 7:
        print(f"{AnsiColor.YELLOW.value}Volta GPU - applying compatibility patch{AnsiColor.RESET.value}")
        get_gpu_compatibility_patch.patch = VoltaGPUPatch()
    elif capability[0] < 7:
        print(f"{AnsiColor.YELLOW.value}Pre-Volta GPU - applying the same compatibility patch as for Volta GPU{AnsiColor.RESET.value}")
        get_gpu_compatibility_patch.patch = VoltaGPUPatch()
    else:
        print(f"{AnsiColor.YELLOW.value}Warning: Unrecognized GPU capability {capability}{AnsiColor.RESET.value}")
        get_gpu_compatibility_patch.patch = None

    get_gpu_compatibility_patch.patch.print_gpu_info()
    return get_gpu_compatibility_patch.patch


# Apply compatibility fixes immediately when this module is imported
get_gpu_compatibility_patch().apply_globally()

__all__ = [
    'setup_compatibility_patch',
    'get_safe_model_kwargs',
    'safe_model_loading_context',
    'get_preferred_compute_dtype',
    'install_sdpa_qwen2_fallback',
    'force_eager_everywhere',
    'disable_flash_attention_modules',
]
