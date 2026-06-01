---
name: serving-llms-on-instinct
description: >-
  Serves AI models on AMD Instinct GPU hardware using vLLM. Use this skill
  whenever the user wants to run, serve, deploy, start, host, or launch a
  language model on an AMD GPU, AMD Instinct, MI300X, MI350X, or MI325X.
  Also use when the user mentions vLLM on ROCm, vLLM on AMD, serving on HBM,
  or asks how to get a model running on AMD data center hardware. Use when the
  user asks "run Qwen3", "serve DeepSeek", "start a vLLM endpoint", "get a
  model running on my AMD machine", or any similar phrasing. Handles the full
  flow: GPU detection, environment validation, vLLM configuration, launch, and
  health verification. Do not use for NVIDIA GPUs, consumer AMD GPUs (RX
  series, Radeon), Ryzen AI, NPU, MI250X, or MI100.
allowed-tools: Bash, Read
---

# Serving LLMs on AMD Instinct

Get a vLLM endpoint running on AMD Instinct GPU hardware. The full flow is:
detect GPU, validate environment, get the correct Docker command, launch,
poll until healthy, return the endpoint.

Run each step, verify before moving to the next. The scripts handle the
deterministic parts; use your judgment for the rest.

## Prerequisites

- ROCm driver and `amd-smi` installed on the GPU host
- Docker running and accessible (check with `docker ps`)
- `/dev/kfd` and `/dev/dri` present on the GPU host
- HuggingFace token in `HF_TOKEN` env var (required for gated models; not
  required for Qwen3 or Gemma)
- For remote GPU: SSH key access configured (`ssh <user>@<host>` must work
  without a password prompt)

## The four-step flow

```
[ ] 1. Detect the GPU — identify gfx_version and available VRAM
[ ] 2. Validate the environment — check devices, Docker, env vars
[ ] 3. Get the Docker command — model + GPU specific config
[ ] 4. Launch and verify — start the container, poll health, return endpoint
```

Run `scripts/run_model.py` to execute all four steps in one shot:

```bash
python scripts/run_model.py --model Qwen/Qwen3-8B
# Remote GPU:
python scripts/run_model.py --model Qwen/Qwen3-8B --host root@10.0.0.5
```

Or run each step individually using the scripts below.

---

## Step 1: Detect the GPU

```bash
python scripts/detect.py
# Remote:
python scripts/detect.py --host root@10.0.0.5
```

The script returns the `gfx_version` — the true hardware identity. Always
work from this, not the market name. Full architecture reference in
[reference.md](reference.md#gpu-architecture).

Expected output includes: `gfx_version`, `vram_gb`, `gpu_count`, `rocm_version`.

| gfx_version | Hardware | VRAM |
|---|---|---|
| gfx950 | MI350 / MI350X | 192–294 GB |
| gfx942 | MI300X / MI300A / MI325X | 128–288 GB |

If `gfx_version` comes back `unknown`: `amd-smi` ran but found no GPU.
Check that the amdgpu kernel module is loaded: `lsmod | grep amdgpu`.

## Step 2: Validate the environment

```bash
python scripts/validate.py
# With auto-fix for safe issues (NUMA balancing, hipBLASLt path):
python scripts/validate.py --auto-fix
# Remote:
python scripts/validate.py --host root@10.0.0.5
```

The script checks: `/dev/kfd` access, `/dev/dri` presence, Docker status,
NUMA balancing, hipBLASLt, and `HF_TOKEN`. It classifies issues as `error`
(blocks launch), `warning` (degrades performance), or `advisory` (informational).

Do not proceed to Step 3 if any `error`-severity issues remain unresolved.
`warning` issues reduce throughput but do not prevent launch.

## Step 3: Get the Docker command

```bash
python scripts/get_config.py --gfx gfx942 --model Qwen/Qwen3-8B
```

Returns the complete Docker command with all required flags, environment
variables, and model-specific workarounds applied. Copy and run it directly.
Do not modify the generated flags without a specific reason — each one exists
to work around a real failure mode.

For the full flag and env var reference, see [reference.md](reference.md#vllm-flags).

## Step 4: Launch and verify

Run the Docker command from Step 3. Then poll health:

```bash
# Poll until healthy (up to 15 minutes for first load):
until curl -sf http://localhost:8000/health; do sleep 10; done && echo "READY"
```

Expected timelines after the model is cached locally:
- Qwen3-8B: 2–4 minutes
- 70B model: 8–15 minutes

A 503 response during this window is normal — vLLM is loading weights.
Only conclude failure after 15+ minutes with no change.

**Verify the endpoint works:**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"say hi"}]}'
```

**Success:** return to the user:
- `base_url`: `http://<host>:8000/v1`
- `api_key`: none required for local
- `model`: the model ID used

**Demo tip:** Send one warmup inference request immediately after health
passes. The first request triggers HIP kernel compilation (30–90 seconds
on gfx942). Subsequent requests are fast.

---

## Gotchas

**`CUDA_VISIBLE_DEVICES` set on the host** — AMD GPUs disappear. The ROCm
runtime treats this NVIDIA variable as "no visible GPUs." Unset it before
launching: `unset CUDA_VISIBLE_DEVICES`. Pass `--env CUDA_VISIBLE_DEVICES=`
in the Docker command to block it inside the container.

**FP4BMM crash on gfx942 (MI300X)** — If the container exits immediately
with a segfault or illegal instruction: `VLLM_ROCM_USE_AITER_FP4BMM` is set
to `1` on gfx942. Set it to `0`. `get_config.py` handles this automatically;
only relevant if the Docker command was hand-edited. See vLLM issue #34641.

**`HIP error: no kernel image`** — The Docker image has no compiled kernel
for your GPU's gfx version. Use `vllm/vllm-openai-rocm:latest`; it includes
gfx942 and gfx950 kernels. Older images may not.

**MLA models need `--block-size 1`** — DeepSeek-R1/V3, Kimi-K2.5, Kimi-K2.
Without it the MLA attention backend silently falls back to a slower path.
`get_config.py` includes this automatically for MLA models.

**MoE models on multi-GPU need `--distributed-executor-backend mp`** —
Qwen3-235B, GLM-4.5, MiniMax-M2. The default distributed executor does not
work reliably with MoE on ROCm.

**`/dev/kfd` permission denied** — User is not in the `video` or `render`
group. Fix: `sudo usermod -aG video,render $USER` (requires re-login).

**SSH key not configured** — The scripts use `BatchMode=yes` SSH (no
interactive prompts). If SSH fails with `Permission denied (publickey)`,
the user must configure key-based SSH access to the GPU host before the
remote path works.

**Model not specified** — Default to `Qwen/Qwen3-8B`: AMD Day 0 support,
Apache 2.0 license (no HF token needed), fits on any MI-series GPU, strong
reasoning and tool-calling. Good for any demo.

---

## Remote vs. local

All scripts accept an optional `--host user@hostname` argument. When given,
the script SSHs to the target machine and runs the relevant commands there
via `BatchMode=yes`. When omitted, everything runs locally.

Set `ROCM_SSH_HOST` and `ROCM_SSH_USER` environment variables to avoid
passing `--host` on every command.

---

## Reference

Full model compatibility matrix, complete vLLM flag and env var reference,
multi-GPU tensor parallelism guide, and known hardware quirks:
[reference.md](reference.md)
