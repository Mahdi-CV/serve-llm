# serving-llms-on-instinct

Claude Code skill for serving LLMs on AMD Instinct GPUs (MI300X, MI325X, MI350X, MI355X) using vLLM.

Handles GPU detection, environment validation, vLLM configuration, Docker launch, and health verification. Supports 98+ models from vLLM recipes with automatic config resolution.

## Supported hardware

| GPU | gfx_version | VRAM |
|---|---|---|
| MI355X | gfx950 | 288 GB HBM3E |
| MI350X | gfx950 | 288 GB HBM3E |
| MI325X | gfx942 | 256 GB HBM3E |
| MI300X | gfx942 | 192 GB HBM3 |
| MI300A | gfx942 | 128 GB unified |

## Install as Claude Code skill

```bash
git clone <repo-url> ~/.claude/skills/serve-llm
```

## Prerequisites

- ROCm driver and `amd-smi` installed on the GPU host
- Docker running and accessible
- `/dev/kfd` and `/dev/dri` present
- `HF_TOKEN` env var set (for gated models)
- SSH key access for remote GPU hosts

## Usage

Just tell Claude what you want:

- "Run Qwen3 on my MI300X"
- "Serve DeepSeek-R1 on the MI350X server"
- "Start a vLLM endpoint for Gemma 4"
- "Get Llama 4 running on root@10.0.0.5"

The skill handles the rest: detects GPUs, validates the environment, looks up model config, constructs the Docker command, launches vLLM, and verifies the endpoint is healthy.

## Structure

```
SKILL.md              # Skill instructions (read by the agent)
reference.md          # Precision compatibility, Docker flags, quirks
data/
  recipes_cache.json  # 98 models from vllm-project/recipes
  gpu_overrides.json  # GPU-specific configs and legacy models
  blacklist.json      # Models that can't be served as LLM endpoints
scripts/
  detect.py           # GPU detection via amd-smi
  validate.py         # Environment validation with auto-fix
  sync_recipes.py     # Refresh recipes from GitHub + Docker Hub
  estimate_vram.py    # VRAM estimation (weights + KV cache)
  test_e2e.py         # Overnight E2E test harness
```
