#!/usr/bin/env python3
"""
Return the complete vLLM Docker command for a given GPU + model combination.

Pure local knowledge lookup -- no SSH, no hardware access required.
All configuration is sourced from AMD-verified vllm-project/recipes.

Usage:
    python scripts/get_config.py --gfx gfx942 --model Qwen/Qwen3-8B
    python scripts/get_config.py --gfx gfx950 --model deepseek-ai/DeepSeek-R1
    python scripts/get_config.py --list-models

Output: JSON with docker_command, env_vars, vllm_args, workarounds, notes.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# GPU configs
# ---------------------------------------------------------------------------

_COMMON_FLAGS = [
    "--group-add=video",
    "--cap-add=SYS_PTRACE",
    "--security-opt seccomp=unconfined",
    "--device /dev/kfd",
    "--device /dev/dri",
    "--ipc=host",
]

GPU_CONFIGS = {
    "gfx950": {
        "gpu_family": "AMD Instinct MI350 / MI350X",
        "vram_gb": 294,
        "docker_image": "vllm/vllm-openai-rocm:latest",
        "docker_flags": _COMMON_FLAGS,
        "env_vars": {
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_FP4BMM": "1",
        },
        "vllm_args": ["--enable-auto-tool-choice", "--trust-remote-code"],
        "workarounds": [],
        "confidence": "validated",
    },
    "gfx942": {
        "gpu_family": "AMD Instinct MI300X / MI300A / MI325X",
        "vram_gb": 192,
        "docker_image": "vllm/vllm-openai-rocm:latest",
        "docker_flags": _COMMON_FLAGS,
        "env_vars": {
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_FP4BMM": "0",  # vLLM #34641 -- crash on gfx942
        },
        "vllm_args": ["--enable-auto-tool-choice", "--trust-remote-code"],
        "workarounds": [
            {"id": "vllm-34641", "description": "FP4BMM=0 mandatory on gfx942 (MI300X crash bug)"},
        ],
        "confidence": "validated",
    },
}

# ---------------------------------------------------------------------------
# Model configs (subset covering most common models)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Qwen/Qwen3-0.6B":  {"vram_fp16_gb": 2,   "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []},
    "Qwen/Qwen3-1.7B":  {"vram_fp16_gb": 4,   "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []},
    "Qwen/Qwen3-4B":    {"vram_fp16_gb": 9,   "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []},
    "Qwen/Qwen3-8B":    {"vram_fp16_gb": 18,  "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": [], "notes": "Default demo model. Apache 2.0, no HF token required."},
    "Qwen/Qwen3-14B":   {"vram_fp16_gb": 30,  "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []},
    "Qwen/Qwen3-32B":   {"vram_fp16_gb": 66,  "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []},
    "Qwen/Qwen3-72B":   {"vram_fp16_gb": 148, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []},
    "Qwen/Qwen3-235B-A22B": {
        "vram_fp16_gb": 564, "min_tp": 4, "tool_call_parser": "hermes", "reasoning_parser": "qwen3", "arch": "moe",
        "env_vars": {"VLLM_USE_V1": "1", "VLLM_ROCM_USE_AITER_MHA": "0", "VLLM_V1_USE_PREFILL_DECODE_ATTENTION": "1", "VLLM_USE_TRITON_FLASH_ATTN": "0", "SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--distributed-executor-backend mp", "--max-num-batched-tokens 32768", "--max-model-len 32768", "--no-enable-prefix-caching", "--gpu-memory-utilization 0.8", "--swap-space 32"],
    },
    "Qwen/Qwen3-VL-7B-Instruct": {
        "vram_fp16_gb": 18, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm",
        "env_vars": {"MIOPEN_USER_DB_PATH": "$(pwd)/miopen", "MIOPEN_FIND_MODE": "FAST", "SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--mm-encoder-tp-mode data"],
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        "vram_fp16_gb": 70, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm",
        "env_vars": {"MIOPEN_USER_DB_PATH": "$(pwd)/miopen", "MIOPEN_FIND_MODE": "FAST", "SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--mm-encoder-tp-mode data"],
    },
    "Qwen/Qwen2.5-VL-7B-Instruct":  {"vram_fp16_gb": 18,  "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": ["--mm-encoder-tp-mode data"]},
    "Qwen/Qwen2.5-VL-72B-Instruct": {"vram_fp16_gb": 148, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": ["--tensor-parallel-size 4", "--mm-encoder-tp-mode data"]},
    "deepseek-ai/DeepSeek-R1": {
        "vram_fp16_gb": 805, "min_tp": 8, "tool_call_parser": "hermes", "reasoning_parser": "deepseek_v3", "arch": "mla_moe",
        "env_vars": {"VLLM_USE_V1": "1", "SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--block-size 1", "--max-model-len 32768", "--max-num-batched-tokens 32768", "--gpu-memory-utilization 0.95", "--no-enable-prefix-caching"],
    },
    "deepseek-ai/DeepSeek-V3": {
        "vram_fp16_gb": 805, "min_tp": 8, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "mla_moe",
        "env_vars": {"VLLM_USE_V1": "1", "SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--block-size 1", "--max-model-len 32768", "--max-num-batched-tokens 32768", "--gpu-memory-utilization 0.95", "--no-enable-prefix-caching"],
    },
    "google/gemma-4-2B-it":  {"vram_fp16_gb": 5,  "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": []},
    "google/gemma-4-4B-it":  {"vram_fp16_gb": 9,  "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": []},
    "google/gemma-4-27B-it": {"vram_fp16_gb": 56, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": []},
    "google/gemma-4-31B-it": {"vram_fp16_gb": 64, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": []},
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "vram_fp16_gb": 110, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "moe_mm",
        "env_vars": {"VLLM_USE_V1": "1", "VLLM_ROCM_USE_AITER_RMSNORM": "0", "VLLM_ROCM_USE_AITER_MHA": "0", "VLLM_V1_USE_PREFILL_DECODE_ATTENTION": "1", "VLLM_USE_TRITON_FLASH_ATTN": "0"},
        "vllm_args": ["--max-model-len 32768", "--max-num-seqs 1024", "--max-num-batched-tokens 32768"],
    },
    "openai/gpt-oss-20b":  {"vram_fp16_gb": 42,  "min_tp": 1, "tool_call_parser": "openai", "reasoning_parser": "", "arch": "dense", "env_vars": {"VLLM_USE_AITER_UNIFIED_ATTENTION": "1"}, "vllm_args": ["--tool-call-parser openai"]},
    "openai/gpt-oss-120b": {"vram_fp16_gb": 247, "min_tp": 2, "tool_call_parser": "openai", "reasoning_parser": "", "arch": "dense", "env_vars": {"VLLM_USE_AITER_UNIFIED_ATTENTION": "1", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4"}, "vllm_args": ["--tool-call-parser openai"]},
    "MiniMaxAI/MiniMax-M2.7": {
        "vram_fp16_gb": 200, "min_tp": 2, "tool_call_parser": "minimax_m2", "reasoning_parser": "minimax_m2", "arch": "moe",
        "env_vars": {"VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1"},
        "vllm_args": ["--attention-backend ROCM_AITER_FA", "--tool-call-parser minimax_m2", "--reasoning-parser minimax_m2"],
        "docker_image_override": None,
    },
    "moonshotai/Kimi-K2.5": {
        "vram_fp16_gb": 700, "min_tp": 8, "tool_call_parser": "kimi_k2", "reasoning_parser": "kimi_k2", "arch": "mla_moe_mm",
        "env_vars": {"VLLM_USE_V1": "1", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4", "SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--block-size 1", "--max-model-len 32768", "--tool-call-parser kimi_k2", "--reasoning-parser kimi_k2"],
        "notes": "Requires ROCm 7.2.1 minimum.",
    },
    "zai-org/GLM-4.5": {
        "vram_fp16_gb": 400, "min_tp": 8, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "moe",
        "env_vars": {"SAFETENSORS_FAST_GPU": "1"},
        "vllm_args": ["--distributed-executor-backend mp"],
        "docker_image_override": "vllm/vllm-openai-rocm:v0.15.1",
        "notes": "Must use vllm-openai-rocm:v0.15.1 -- :latest has GLM compatibility issues.",
    },
    "OpenGVLab/InternVL3_5-8B": {"vram_fp16_gb": 18, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense_mm", "env_vars": {}, "vllm_args": ["--mm-encoder-tp-mode data"]},
}


def _find_model(model_id):
    """Exact match first, then prefix match (longest wins)."""
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]
    matches = [(k, v) for k, v in MODEL_CONFIGS.items() if model_id.startswith(k) or k.startswith(model_id)]
    if matches:
        return sorted(matches, key=lambda x: len(x[0]), reverse=True)[0][1]
    return None


def build_docker_command(gfx, model_id, gpu_count=1, hf_cache="~/.cache/huggingface", port=8000):
    gpu_cfg = GPU_CONFIGS.get(gfx)
    if not gpu_cfg:
        return {"error": f"Unknown gfx_version '{gfx}'. Known: {list(GPU_CONFIGS.keys())}"}

    model_cfg = _find_model(model_id)
    model_known = model_cfg is not None
    if not model_cfg:
        # Generic fallback
        model_cfg = {"vram_fp16_gb": 0, "min_tp": 1, "tool_call_parser": "hermes", "reasoning_parser": "", "arch": "dense", "env_vars": {}, "vllm_args": []}

    # Merge env vars: GPU base + model overrides
    env_vars = {**gpu_cfg["env_vars"], **model_cfg.get("env_vars", {})}
    env_vars["HF_TOKEN"] = "${HF_TOKEN}"

    # Determine TP
    tp = max(model_cfg["min_tp"], 1)
    if gpu_count > 1 and tp == 1 and model_cfg.get("vram_fp16_gb", 0) > gpu_cfg["vram_gb"]:
        tp = min(gpu_count, 8)

    # VRAM fit check
    vram_warning = None
    vram_needed = model_cfg.get("vram_fp16_gb", 0)
    total_vram = gpu_cfg["vram_gb"] * max(tp, 1)
    if vram_needed > 0 and vram_needed > total_vram * 0.95:
        vram_warning = f"Model needs ~{vram_needed}GB but {max(tp,1)}x GPU provides {total_vram}GB. Consider FP8 checkpoint or more GPUs."

    # Docker image
    image = model_cfg.get("docker_image_override") or gpu_cfg["docker_image"]

    # Merge vllm args: GPU base + model-specific
    vllm_args = list(gpu_cfg["vllm_args"]) + list(model_cfg.get("vllm_args", []))
    if tp > 1 and f"--tensor-parallel-size {tp}" not in " ".join(vllm_args):
        vllm_args.append(f"--tensor-parallel-size {tp}")

    # Add tool/reasoning parsers if not already in vllm_args
    args_str = " ".join(vllm_args)
    tcp = model_cfg.get("tool_call_parser", "hermes")
    if tcp and "--tool-call-parser" not in args_str:
        vllm_args.append(f"--tool-call-parser {tcp}")
    rp = model_cfg.get("reasoning_parser", "")
    if rp and "--reasoning-parser" not in args_str:
        vllm_args.append(f"--reasoning-parser {rp}")

    # Assemble docker command
    lines = [f"docker run -d --name vllm-{model_id.split('/')[-1].lower()[:20]}"]
    for flag in gpu_cfg["docker_flags"]:
        lines.append(f"  {flag}")
    lines.append(f"  -v {hf_cache}:/root/.cache/huggingface")
    lines.append(f"  -p {port}:{port}")
    for k, v in env_vars.items():
        lines.append(f"  --env {k}={v}")
    lines.append(f"  {image}")
    lines.append(f"  --model {model_id}")
    for arg in vllm_args:
        lines.append(f"  {arg}")
    lines.append(f"  --port {port}")

    docker_command = " \\\n".join(lines)

    return {
        "model_id": model_id,
        "model_known": model_known,
        "gfx_version": gfx,
        "gpu_family": gpu_cfg["gpu_family"],
        "confidence": gpu_cfg["confidence"],
        "tensor_parallel": tp,
        "vram_warning": vram_warning,
        "docker_command": docker_command,
        "env_vars": env_vars,
        "vllm_args": vllm_args,
        "workarounds": gpu_cfg["workarounds"] + model_cfg.get("workarounds", []),
        "notes": model_cfg.get("notes", []) if isinstance(model_cfg.get("notes"), list) else ([model_cfg["notes"]] if model_cfg.get("notes") else []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gfx", help="GFX version from detect.py (e.g. gfx942, gfx950)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model ID")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hf-cache", default="~/.cache/huggingface")
    parser.add_argument("--list-models", action="store_true", help="List all known model IDs")
    args = parser.parse_args()

    if args.list_models:
        print(json.dumps(sorted(MODEL_CONFIGS.keys()), indent=2))
        return

    if not args.gfx:
        print(json.dumps({"error": "--gfx is required (e.g. --gfx gfx942). Run detect.py first."}))
        sys.exit(1)

    result = build_docker_command(
        gfx=args.gfx,
        model_id=args.model,
        gpu_count=args.gpu_count,
        hf_cache=args.hf_cache,
        port=args.port,
    )
    print(json.dumps(result, indent=2))
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
