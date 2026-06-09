# serving-llms-on-instinct — Reference

## Table of Contents
1. [GPU Architecture](#gpu-architecture)
2. [Model Compatibility Matrix](#model-compatibility-matrix)
3. [vLLM Flags](#vllm-flags)
4. [Environment Variables](#environment-variables)
5. [Known Quirks](#known-quirks)

---

## GPU Architecture

| GPU | gfx_version | VRAM | AITER | FP4BMM |
|---|---|---|---|---|
| MI355X | gfx950 | 288 GB HBM3E | Yes | Safe (= 1) |
| MI350X | gfx950 | 288 GB HBM3E | Yes | Safe (= 1) |
| MI325X | gfx942 | 256 GB HBM3E | Yes | Crash bug (= 0) |
| MI300X | gfx942 | 192 GB HBM3 | Yes | Crash bug (= 0) |
| MI300A | gfx942 | 128 GB unified | Yes | Crash bug (= 0) |

Detect gfx_version:
```bash
amd-smi static --asic --json | python3 -c \
  "import sys,json; d=json.load(sys.stdin); gs=d if isinstance(d,list) else d.get('gpu_data',[d]); [print(g['asic']['target_graphics_version']) for g in gs]"
# or
rocminfo | grep "gfx"
```

### Precision Compatibility

| Format | gfx942 (MI300X) | gfx950 (MI350X) | Notes |
|---|---|---|---|
| BF16 / FP16 | Native | Native | Default for all models |
| FP8 (FNUZ) | Native | Emulated | MI300X uses E4M3FNUZ dialect |
| FP8 (OCP) | Emulated | Native | MI350X uses E4M3FN (OCP standard) |
| INT8 | Native | Native | |
| MXFP4 | Emulated | Native | On gfx942: dequants to BF16, no VRAM savings |
| MXFP6 | Emulated | Native | On gfx942: dequants to BF16, no VRAM savings |
| NVFP4 | Emulated | Emulated | NVIDIA-specific, always dequants to BF16 on AMD |

"Emulated" means the format is handled via dequantization to BF16 at runtime.
The model still loads and runs correctly but there are no VRAM savings compared
to BF16. vLLM auto-converts between FP8 dialects (FNUZ/OCP) transparently.

Recipe `nvfp4` variant VRAM numbers assume NVIDIA hardware with native FP4.
On AMD, these variants use the same VRAM as the `default` (BF16) variant.
Prefer `default` or `fp8` variants on gfx942. On gfx950, `mxfp4` variants
(when available) will provide real VRAM savings.

---

## Model Compatibility Matrix

Model configs are auto-synced from [vllm-project/recipes](https://github.com/vllm-project/recipes)
into `data/recipes_cache.json`. Read the cache file directly for the full current
list. The table below covers commonly used models for reference.
VRAM at FP16/BF16 unless noted.

| Model | HF ID | Arch | VRAM FP16 | VRAM FP8 | Min TP | Notes |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Qwen/Qwen3-0.6B | Dense | 2 GB | — | 1 | |
| Qwen3-1.7B | Qwen/Qwen3-1.7B | Dense | 4 GB | — | 1 | |
| Qwen3-4B | Qwen/Qwen3-4B | Dense | 9 GB | — | 1 | |
| Qwen3-8B | Qwen/Qwen3-8B | Dense | 18 GB | -- | 1 | Apache 2.0. |
| Qwen3.5-9B | Qwen/Qwen3.5-9B | Dense+MM | 22 GB | -- | 1 | **Default. Apache 2.0. MTP.** |
| Qwen3-14B | Qwen/Qwen3-14B | Dense | 30 GB | — | 1 | |
| Qwen3-32B | Qwen/Qwen3-32B | Dense | 66 GB | — | 1 | |
| Qwen3-72B | Qwen/Qwen3-72B | Dense | 148 GB | — | 1 | Fits MI300X+ |
| Qwen3-235B-A22B | Qwen/Qwen3-235B-A22B | MoE | 564 GB | 282 GB | 4 | TP=4, mp executor |
| Qwen3-VL-7B | Qwen/Qwen3-VL-7B-Instruct | MM | 18 GB | — | 1 | |
| Qwen3-VL-32B | Qwen/Qwen3-VL-32B-Instruct | MM | 70 GB | — | 1 | |
| Qwen3-VL-235B | Qwen/Qwen3-VL-235B-A22B-Instruct | MoE MM | 564 GB | — | 4 | |
| Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B-Instruct | MM | 18 GB | — | 1 | |
| Qwen2.5-VL-72B | Qwen/Qwen2.5-VL-72B-Instruct | MM | 148 GB | — | 1 | TP=4 for throughput |
| DeepSeek-R1 | deepseek-ai/DeepSeek-R1 | MLA+MoE | 805 GB | 402 GB | 8 | `--block-size 1` mandatory |
| DeepSeek-V3 | deepseek-ai/DeepSeek-V3 | MLA+MoE | 805 GB | 402 GB | 8 | `--block-size 1` mandatory |
| Gemma 4-2B | google/gemma-4-2B-it | MM | 5 GB | — | 1 | |
| Gemma 4-4B | google/gemma-4-4B-it | MM | 9 GB | — | 1 | |
| Gemma 4-27B | google/gemma-4-27B-it | MM | 56 GB | — | 1 | |
| Gemma 4-31B | google/gemma-4-31B-it | MM | 64 GB | — | 1 | |
| Llama 4 Scout | meta-llama/Llama-4-Scout-17B-16E-Instruct | MoE MM | 110 GB | 55 GB | 1 | HF token required |
| GPT-OSS-20B | openai/gpt-oss-20b | Dense | 42 GB | — | 1 | |
| GPT-OSS-120B | openai/gpt-oss-120b | Dense | 247 GB | — | 2 | 1x MI350X or 2x MI300X |
| MiniMax-M2.7 | MiniMaxAI/MiniMax-M2.7 | MoE | 200 GB | 100 GB | 2 | FP8 fits 1x MI350X |
| Kimi-K2.5 | moonshotai/Kimi-K2.5 | MLA+MoE MM | 700 GB | 350 GB | 8 | ROCm 7.2.1+ required |
| GLM-4.5 | zai-org/GLM-4.5 | MoE | 400 GB | 200 GB | 8 | Pin to rocm:v0.15.1 image |
| InternVL3.5-8B | OpenGVLab/InternVL3_5-8B | MM | 18 GB | — | 1 | |

---

## vLLM Flags

### Mandatory Docker flags (all AMD Instinct)

| Flag | Why |
|---|---|
| `--group-add=video` | amdgpu exposes GPUs to the `video` group |
| `--group-add=render` | GPU render nodes require the `render` group on many hosts |
| `--cap-add=SYS_PTRACE` | ROCm JIT compilation requires ptrace |
| `--security-opt seccomp=unconfined` | ROCm mmap variants blocked by default seccomp |
| `--device /dev/kfd` | Kernel Fusion Driver — primary GPU access |
| `--device /dev/dri` | Render nodes for GPU command submission |
| `--ipc=host` | ROCm shared memory needs host IPC namespace |

### Recommended Docker flags

| Flag | Why |
|---|---|
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | Reuse downloaded models |
| `--env HF_TOKEN=${HF_TOKEN}` | Required for gated models |
| `-p 8000:8000` | Expose OpenAI-compatible API |

### Docker image

`vllm/vllm-openai-rocm:<tag>` -- tag is auto-resolved from Docker Hub
during recipe sync (currently `v0.22.0`). Includes gfx942 and gfx950 kernels.
Do NOT use `vllm/vllm-openai` (CUDA-only).
**Exception:** GLM-4.5 must use `vllm/vllm-openai-rocm:v0.15.1`.

### vLLM server arguments

| Argument | When | Notes |
|---|---|---|
| `--enable-auto-tool-choice` | Always (agent use) | |
| `--trust-remote-code` | Qwen3, Kimi, InternVL | |
| `--tensor-parallel-size N` | Multi-GPU | N must divide attention heads |
| `--distributed-executor-backend mp` | MoE on multi-GPU | Required on ROCm |
| `--block-size 1` | DeepSeek, Kimi (MLA) | MANDATORY for MLA attention |
| `--mm-encoder-tp-mode data` | Multimodal + TP | Prevent redundant vision encoding |
| `--attention-backend ROCM_AITER_FA` | MiniMax-M2 | After vLLM v0.21.0 |
| `--max-model-len 32768` | Large MoE | Reduce KV cache VRAM |

### Tool call parsers

| Parser | Models |
|---|---|
| `hermes` | Qwen3, Gemma 4, most models |
| `openai` | GPT-OSS |
| `minimax_m2` | MiniMax-M2 |
| `kimi_k2` | Kimi-K2.5 |
| `deepseek_v32` | DeepSeek-V3.2 |

---

## Environment Variables

### AITER (AMD Instinct TEnsor Runtime)

| Variable | gfx950 | gfx942 | Effect |
|---|---|---|---|
| `VLLM_ROCM_USE_AITER` | **1** | **1** | Master AITER switch. 1.2-4.4x throughput. |
| `VLLM_ROCM_USE_AITER_FP4BMM` | **1** | **0** | FP4 matmul. Crash on gfx942 (vLLM #34641). Safe on gfx950. |
| `VLLM_ROCM_USE_AITER_MHA` | 1 | 1 | AITER multi-head attention. Set 0 for Qwen3 MoE, Llama 4, GPT-OSS. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | 1 | 1 | AITER RMSNorm. Set 0 for Llama 4 Scout, Kimi-K2.5. |
| `VLLM_USE_AITER_UNIFIED_ATTENTION` | - | - | Set 1 for GPT-OSS models. |

### Attention backend

| Variable | When | Effect |
|---|---|---|
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | MiniMax-M2 | Restore AITER FA after vLLM v0.21.0 change |
| `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1` | Qwen3 MoE, Llama 4 | Split prefill/decode kernels |
| `VLLM_USE_TRITON_FLASH_ATTN=0` | Qwen3 MoE, Llama 4 | Use ROCm-native FA instead of Triton |
| `VLLM_USE_V1=1` | DeepSeek, Llama 4, Qwen3 MoE | Force vLLM V1 engine |

### Multi-GPU

| Variable | When | Effect |
|---|---|---|
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | Kimi-K2.5, GPT-OSS | INT4 all-reduce compression |
| `NCCL_MIN_NCHANNELS=112` | Multi-GPU TP | Max RCCL channels |

### Visibility (critical footguns)

| Variable | Rule |
|---|---|
| `CUDA_VISIBLE_DEVICES` | ROCm maps this to `HIP_VISIBLE_DEVICES`. Works with explicit indices (e.g. `0,1`). **Never set to empty string** -- hides all GPUs. |
| `HIP_VISIBLE_DEVICES=0,1,2,3` | Canonical AMD variable. Use to restrict visible GPUs by index on multi-GPU hosts. |

### Loading / performance

| Variable | When | Effect |
|---|---|---|
| `SAFETENSORS_FAST_GPU=1` | Large models (235B+) | Fast safetensors GPU loading |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | General | Prefer hipBLASLt for GEMM |
| `MIOPEN_USER_DB_PATH=$(pwd)/miopen` | Qwen3-VL, Qwen2.5-VL | MIOpen tuning DB path |
| `MIOPEN_FIND_MODE=FAST` | Qwen3-VL, Qwen2.5-VL | Reduce first-launch latency |

---

## Known Quirks

**vLLM #34641 — FP4BMM crash on gfx942**
Segfault or illegal instruction during model warmup on MI300X/MI325X/MI300A.
Triggered when `VLLM_ROCM_USE_AITER_FP4BMM=1` on gfx942.
Fix: always set `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942.
This is set correctly in `data/gpu_overrides.json` for gfx942.

**CUDA_VISIBLE_DEVICES empty string**
ROCm maps `CUDA_VISIBLE_DEVICES` to `HIP_VISIBLE_DEVICES`. Setting it to an
empty string hides all GPUs. Setting it to explicit indices (e.g. `0,1`) works
correctly. If the host has it set to empty, unset it: `unset CUDA_VISIBLE_DEVICES`.
Do not pass `--env CUDA_VISIBLE_DEVICES=` (empty) into Docker.

**NUMA balancing latency spikes**
`/proc/sys/kernel/numa_balancing=1` periodically migrates pages between NUMA nodes.
For GPU workloads this causes latency spikes as GPU DMA must follow moved pages.
Disable: `echo 0 | sudo tee /proc/sys/kernel/numa_balancing`
Non-persistent — resets on reboot.

**First-token warmup delay**
vLLM compiles and caches HIP kernels on first use per input shape.
First inference after model load: ~40-45 seconds on gfx942.
Send a warmup request immediately after `/health` returns 200 for demos.

**hipBLASLt path discovery**
Some environments need `TORCH_BLAS_PREFER_HIPBLASLT=1` set explicitly.
Without it, PyTorch may fall back to a slower BLAS path.
`validate.py` checks for this and reports if the path is not found.

**"Engine core initialization failed"**
This opaque error covers many root causes. Check early container logs
(`docker logs <name> 2>&1 | head -50`) -- the actual error is often from the
engine core subprocess, not the main process. Common causes:
- Gated model: HF license not accepted (not just missing token)
- Unsupported architecture on this vLLM version
- OOM during weight loading
- Missing `--trust-remote-code` for custom model architectures
- vLLM version too old (check `min_vllm_version` in the recipe)

**ROCm version for Kimi-K2.5**
Kimi-K2.5 requires ROCm 7.2.1 minimum. Fails silently or with obscure errors
on ROCm 7.0.x. Check ROCm version: `rocminfo | grep "ROCm"` or `amd-smi version`.
