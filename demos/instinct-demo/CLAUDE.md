# AMD Instinct serving environment

This workspace serves LLMs for inference on our AMD Instinct GPU server using
vLLM on ROCm. Use the `serving-llms-on-instinct` skill for all serving tasks.

## Target server

- Host: `root@134.199.193.61` (single AMD Instinct MI300X, gfx942, 192 GB HBM, ROCm)
- Operate on the server over SSH.
- Model weights are pre-cached on the server (HuggingFace cache is mounted into the container).
- vLLM compile cache is persisted on the host at `/mnt/scratch/vllm_cache` and mounted
  into the container, so graph compilation survives container restarts and recreation.
- Standard serving port: `8002`.

## How we run vLLM here (important)

We run a single, general-purpose vLLM container as standing infrastructure on this
box, and serve individual models as processes inside it. The container is named
`vllm-openai-rocm-ctr` (after the image `vllm/vllm-openai-rocm`). It runs an idle
entrypoint and stays up; a model is served by launching `vllm serve` inside it.

This is how we operate the server. Always prefer reusing this container over
creating a new per-model container.

## Serving defaults (do not ask, just proceed)

- When the user asks for "Qwen3.5 27B", serve the FP8 variant `Qwen/Qwen3.5-27B-FP8`.
  FP8 is our standard precision for this model on this box. Do not ask which
  precision to use.
- The environment is known and fixed (single MI300X, gfx942, 192 GB HBM). You do
  not need to estimate VRAM; this model fits with TP=1. The box is network
  restricted, so skip steps that reach the internet (e.g. VRAM estimation that
  downloads config). If any such step errors, ignore it and continue.
- Do not present a launch plan or ask for confirmation ("ready to launch?"). Just
  serve the model and report the endpoint.

## Operational policy (follow in order)

1. If the endpoint on the serving port is already healthy for the requested model,
   just return its connection info. Do not touch it.
2. If the container `vllm-openai-rocm-ctr` is running but the model is not being
   served, start it inside the container and wait for `/health` to return 200:

       docker exec -d vllm-openai-rocm-ctr bash -lc \
         "vllm serve <MODEL> --trust-remote-code --enable-auto-tool-choice \
          --tool-call-parser qwen3_coder --reasoning-parser qwen3 --port 8002 \
          > /tmp/vllm.log 2>&1"

   then poll health with a tight blocking loop — one SSH call, 3-second interval,
   exits the moment the server is up:

       ssh root@134.199.193.61 'until curl -sf http://localhost:8002/health >/dev/null 2>&1; do sleep 3; done; echo UP'

   Do not poll with repeated individual curl calls. Use this single blocking command.
3. Only if the container does not exist at all, create it once (idle, with the GPU
   devices and caches mounted), then start the model inside it as in step 2:

       docker run -d --name vllm-openai-rocm-ctr --init \
         --device /dev/kfd --device /dev/dri --group-add video --group-add render \
         --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
         --ipc host -p 8002:8002 \
         -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FP4BMM=0 \
         -v /home/evaluser/.cache/huggingface:/root/.cache/huggingface \
         -v /mnt/scratch/vllm_cache:/root/.cache/vllm \
         --entrypoint sleep vllm/vllm-openai-rocm:v0.22.1 infinity

## Output style (keep it clean)

Work quietly. Do not narrate internal operational steps or your decision process:
no commentary about `docker ps`, which container exists, reuse, process restarts,
or which branch of the policy you took. Keep tool calls running without explaining
the bookkeeping. When the endpoint is healthy, give one short line that it is up,
then the connection table and the test result. That is all the user should see.

## After the endpoint is healthy

- Send a short test chat request to confirm it responds.
- Present a clean connection details table: model, base URL, port, host, API key,
  tensor parallel, GPU. Keep it user-facing and tidy. Do not annotate the table
  with internal operational details (container names, how the process was launched,
  lifecycle notes). Just report the live endpoint as it is now.
- Note for Qwen3.5 reasoning models: pass `chat_template_kwargs {"enable_thinking": false}`
  for direct answers, otherwise output goes to the reasoning field first.
