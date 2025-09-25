# Volta (V100) FlashAttention compatibility — status & notes

This note explains why SimLingo’s InternVL2/Qwen2 stack fails on Tesla V100 (Volta), what we changed, what works now, and the remaining blocker. Share this with other threads so everyone has the same context.

## TL;DR
- FlashAttention requires Ampere or newer (compute capability ≥ 8.0). V100 is Volta (7.0) → not supported.
- We centralized a “Volta compatibility” setup that:
  - Disables FlashAttention/FlashAttention2 via env vars and Transformers hooks.
  - Forces PyTorch scaled_dot_product_attention (math) instead of flash kernels.
  - Forces attn_implementation="eager" in configs (including nested llm_config) and deep submodules.
- Model loads and FA2 toggling stops. The remaining issue: during Qwen2 forward, Transformers still referenced `_flash_attention_forward`. We added a robust SDPA-based fallback to cover that.

## Context
- GPU: Tesla V100S (Volta, CC 7.0)
- Model: OpenGVLab/InternVL2-1B (InternVL2 vision + Qwen2 LLM)
- Libraries: PyTorch + Hugging Face Transformers

## Symptoms we saw
1. RuntimeError: “FlashAttention only supports Ampere GPUs or newer”.
2. Transformers enabling FlashAttention2 internally → errors when FA2 unavailable.
3. Even with eager selected, a call to `_flash_attention_forward` remained in the Qwen2 path → NameError/AttributeError depending on HF version.

## What we changed (centralized and repeatable)
- File: `simlingo_training/utils/gpu_compatibility.py`
  - On import, sets env toggles to discourage FA/FA2 and prefer eager.
  - Forces PyTorch SDP kernels to math mode: no flash or mem-efficient variants.
  - Patches Transformers to lie about flash_attn availability and to no-op FA2 enablement.
  - Forces attn_implementation="eager" in PreTrainedModel hooks and flips any `use_flash_attn_2=False`.
  - Installs an SDPA-based fallback function and aliases it to `_flash_attention_forward` for Qwen2 (and LLaMA) so any stray flash calls hit SDPA instead.
- Agent/model loaders now use that module’s helpers to pass safe kwargs (`attn_implementation="eager"`, `torch_dtype=float16` on Volta).
- Test harness: `test_model_loading_v100.py`
  - Patches both top-level config and nested `llm_config` before from_pretrained.
  - Enforces eager at the module/submodule level post-instantiation.
  - Executes a tiny forward pass to verify we don’t touch flash code paths.

## Current status
- Eager attention paths are selected and FA/FA2 toggles are disabled consistently.
- An SDPA fallback for `_flash_attention_forward` is installed from the compatibility module to catch any lingering references.

Validated on remote V100 (Tesla V100S, CC 7.0):
- `test_gpu_compatibility.py` passes and shows FA disabled and SDPA fallback installed.
- Direct call to `transformers.models.qwen2.modeling_qwen2._flash_attention_forward` executes on CUDA via SDPA with fp16 tensors.

We also adjusted the runtime agent to avoid bfloat16 on Volta by selecting float16 automatically (see `team_code/agent_simlingo.py`).

## If issues persist
- Transformers internals around flash helpers differ across versions. If a failure still shows up around `_flash_attention_forward`, try one of:
  - Track A (quick): Keep our SDPA alias; ensure you’re running a fresh Python process so the alias takes effect at import-time.
  - Track B (cleaner): Pin Transformers to a version that never references the flash path when `attn_implementation="eager"` (we can recommend a pin once exact versions are listed).
  - Track C (dtype): On V100 use `torch.float16` everywhere (inputs, default dtype, model load). We've wired this into the agent so it happens automatically.

## How to validate locally
- Script: `test_model_loading_v100.py`
  - It applies the compatibility setup, loads `OpenGVLab/InternVL2-1B`, enforces eager across configs/submodules, and runs a minimal forward.
  - Expect: no FlashAttention errors, forward completes.

On the remote V100 host, you can run quick checks without pulling weights:
- `python -u test_gpu_compatibility.py`
- Or import the compatibility module and call Qwen2's flash stub directly with random fp16 tensors to ensure the alias routes to SDPA (already verified once).

## Notes
- V100 doesn’t do bfloat16 well → we prefer `torch.float16`.
- This is a pragmatic workaround to enable evaluation on V100; performance will be lower than flash kernels on Ampere+.

Next steps
- Keep using the centralized module `simlingo_training/utils/gpu_compatibility.py` and ensure it’s imported before any model loads (agent and tests already do this).
- For evaluation on V100, the agent now sets the compute dtype to float16 transparently.
- If a new Transformers release changes internal flash helpers, retain the SDPA alias patch or pin Transformers.

---
Questions or failures: paste your torch/transformers versions and the exact stack trace, and we’ll advise whether to tweak the alias or pin versions.