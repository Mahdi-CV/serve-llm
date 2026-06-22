# AMD Instinct MI350 serving environment

This workspace serves LLMs for inference on our AMD Instinct MI350X GPU server using
vLLM on ROCm. Use the `serving-llms-on-instinct` skill for all serving tasks.

## Target server

- Host: `root@134.199.199.97` (8x AMD Instinct MI350X, gfx950, 2304 GB HBM total, ROCm 7.2.0)
- Operate on the server over SSH.
- Model weights are pre-cached on the server at `/root/.cache/huggingface` (mounted into the container).
- vLLM compile cache is persisted on the host at `/data/vllm_cache` and mounted into
  the container, so compiled graphs survive process kills and container recreation.
- Standard serving port: `8000`.

## How we run vLLM here (important)

We run a single, general-purpose vLLM container as standing infrastructure on this
box, named `vllm-openai-rocm-ctr` (after the image `vllm/vllm-openai-rocm:latest`).
It runs an idle entrypoint and stays up; a model is served by launching `vllm serve`
inside it via `docker exec -d`.

Always prefer reusing this container over creating a new per-model container.

## Serving defaults (do not ask, just proceed)

### Server aliases — all of these mean `root@134.199.199.97`

Any of the following in the user's prompt refers to the same box:
- do-mi350-node, DO server, digital ocean, digital ocean server, my DO,
  mi350, mi355, mi355x, instinct server, the server, my server

If you cannot tell which server is meant, default to `root@134.199.199.97`.

### Model mapping — use these exact IDs and TP values

Any mention of "gpt-oss", "gpt oss", or "openai model" maps to one of these.
Default to 120B if the size is not specified.

| User says | Model ID | TP |
|-----------|----------|----|
| gpt-oss 20B, 20b, the small one | `openai/gpt-oss-20b` | 1 |
| gpt-oss 120B, 120b, the large one, (size not mentioned) | `openai/gpt-oss-120b` | 8 |

Do not ask which precision or variant to use. Do not estimate VRAM. The box is
network restricted — skip any steps that reach the internet and ignore such errors.
Do not present a launch plan or ask for confirmation. Just serve and report the endpoint.

## Operational policy (follow in order)

1. If the endpoint on port 8000 is already healthy, just return its connection info.
2. If the container `vllm-openai-rocm-ctr` is running but the model is not being
   served, launch it inside the container using the correct model ID and TP from the
   table above, then poll `/health`:

   For gpt-oss 20B:

       docker exec -d vllm-openai-rocm-ctr bash -lc \
         "vllm serve openai/gpt-oss-20b \
          --tensor-parallel-size 1 \
          --gpu-memory-utilization 0.95 \
          --compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}' \
          --block-size 64 \
          --async-scheduling \
          --port 8000 \
          > /tmp/vllm.log 2>&1"

   For gpt-oss 120B:

       docker exec -d vllm-openai-rocm-ctr bash -lc \
         "vllm serve openai/gpt-oss-120b \
          --tensor-parallel-size 8 \
          --gpu-memory-utilization 0.95 \
          --compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}' \
          --block-size 64 \
          --async-scheduling \
          --port 8000 \
          > /tmp/vllm.log 2>&1"

   Then poll health with a tight blocking loop — one SSH call, 3-second interval,
   exits the moment the server is up:

       ssh root@134.199.199.97 'until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 3; done; echo UP'

   Do not poll with repeated individual curl calls. Use this single blocking command.
3. Only if the container does not exist, create it once (idle, GPU and caches mounted),
   then start the model inside it as in step 2:

       docker run -d --name vllm-openai-rocm-ctr --init \
         --device /dev/kfd --device /dev/dri \
         --group-add video --group-add render \
         --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
         --ipc host \
         -p 8000:8000 \
         -e VLLM_ROCM_USE_AITER=1 \
         -e VLLM_ROCM_USE_AITER_FP4BMM=1 \
         -e VLLM_ROCM_USE_AITER_MOE=0 \
         -e VLLM_ROCM_USE_AITER_MHA=0 \
         -e VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1 \
         -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 \
         -e HIP_FORCE_DEV_KERNARG=1 \
         -v /root/.cache/huggingface:/root/.cache/huggingface \
         -v /data/vllm_cache:/root/.cache/vllm \
         --entrypoint sleep \
         vllm/vllm-openai-rocm:latest infinity

## Output style (keep it clean)

Work quietly. Do not narrate internal steps, docker ps results, container reuse logic,
or which policy branch you took. Keep tool calls running silently. When the endpoint is
healthy, give one short line that it is up, then the connection table and the test
result.

## After the endpoint is healthy

- Send a short test chat request to confirm it responds.
- Present a clean connection details table: model, base URL, port, host, API key,
  tensor parallel, GPU.
- gpt-oss does not use a thinking/reasoning kwarg. Send a direct question and expect
  a direct answer.
