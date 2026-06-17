# Adapting the demo for gpt-oss

This is the model-specific layer for `gpt-oss`. The architecture (standing container,
exec-launch, kill-to-reset, output style) is unchanged. You only change the model id,
the vLLM flags, any env, and the spoken name.

> Important: confirm the exact vLLM flags below against your vLLM build before the
> demo. gpt-oss parsing/flags differ from Qwen and have changed across vLLM versions.
> Verify with `vllm serve --help | grep -i parser` and the vLLM gpt-oss serving recipe.
> Always test the raw `vllm serve` command by hand inside the container first (watch
> `/tmp/vllm.log`) before wiring it into the demo.

## Models

| Spoken name | HF model id | Notes |
|-------------|-------------|-------|
| gpt-oss 20B | `openai/gpt-oss-20b` | small, fits easily on one MI300X |
| gpt-oss 120B | `openai/gpt-oss-120b` | MoE, MXFP4; fits on a single MI300X (192 GB) with TP=1 |

gpt-oss ships natively in MXFP4, so there is no separate FP8 vs BF16 choice to pin
(unlike the Qwen demo). The "do not ask precision" instruction in `CLAUDE.md` still
applies, just point it at the one model id.

## Placeholder values to use

| Placeholder | Value |
|-------------|-------|
| `<SERVER_IP>` | your GPU box IP |
| `<SSH_USER>` | `root` (or your box user) |
| `<PORT>` | `8002` (or your choice) |
| `<CONTAINER>` | `vllm-openai-rocm-ctr` (same standing container is fine) |
| `<IMAGE>` | a vLLM ROCm image with gpt-oss support (verify the tag supports gpt-oss) |
| `<MODEL>` | `openai/gpt-oss-120b` |
| `<MODEL_SPOKEN>` | `gpt-oss 120B` |
| `<HF_CACHE>` | your host HF cache dir |
| `<VLLM_CACHE>` | your host vLLM cache dir |
| `<VLLM_MODEL_ARGS>` | gpt-oss tool/reasoning parser flags - VERIFY (see below) |
| `<VLLM_ENV>` | ROCm/AITER env - VERIFY for your image |

## vLLM flags (verify these)

The Qwen demo used `--tool-call-parser qwen3_coder --reasoning-parser qwen3`. Those are
Qwen-specific and must be replaced for gpt-oss. gpt-oss uses the "harmony" format and
vLLM provides gpt-oss-specific parsing. Recent vLLM exposes gpt-oss reasoning/tool
parsers; the exact names depend on your version, so check:

```bash
docker exec <CONTAINER> vllm serve --help | grep -iE 'reasoning-parser|tool-call-parser'
```

Then assemble and TEST the serve command by hand inside the container:

```bash
docker exec -it <CONTAINER> bash -lc \
  "vllm serve openai/gpt-oss-120b \
     --enable-auto-tool-choice \
     <the gpt-oss tool/reasoning parser flags you confirmed> \
     --port <PORT>"
```

Watch it come up, confirm `/health` returns 200 and a chat completion works, then put
the confirmed flags into `<VLLM_MODEL_ARGS>` in `CLAUDE.md` and `stage_demo.sh`.

Notes specific to gpt-oss:
- `--trust-remote-code` is generally not required for gpt-oss; include it only if your
  image needs it.
- gpt-oss exposes a reasoning effort (low/medium/high) via the request, not via a Qwen
  `enable_thinking` kwarg. Replace the Qwen "After the endpoint is healthy" thinking
  note in `CLAUDE.md` with the gpt-oss equivalent (set reasoning effort in the test
  request if you want a short, direct answer).

## Pre-warm before the demo

gpt-oss-120b will compile/warm on first launch. Do the one-time warm well ahead:

```bash
cd ~/gpt-oss-demo && ./stage.sh           # ensures container, kills any process
claude --dangerously-skip-permissions     # say the prompt once, let it fully come up
```

Then `./stage.sh` again and re-run to confirm the warm path timing. That second run is
your demo timing.

## Client (hermes)

`set_hermes_custom_model.sh` already points hermes at `http://<SERVER_IP>:<PORT>/v1`
with `model.default = <MODEL>`. Just set `<MODEL>` to `openai/gpt-oss-120b` in that
script for this demo.
