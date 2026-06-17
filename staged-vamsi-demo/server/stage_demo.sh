#!/usr/bin/env bash
# Stage the demo box for the persistent-container demo.
# - Ensures the general-purpose vLLM container is up (creates it if missing).
# - Kills any running model process so the next run shows the warmup.
set -u

CTR=<CONTAINER>
IMAGE=<IMAGE>
PORT=<PORT>

ensure_container() {
  if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
    # Remove a stale stopped one if present, then create fresh.
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    echo "[stage] container not running - creating $CTR ..."
    docker run -d --name "$CTR" --init \
      --device /dev/kfd --device /dev/dri \
      --group-add video --group-add render \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
      --ipc host \
      -p ${PORT}:${PORT} \
      <VLLM_ENV_AS_-e_FLAGS> \
      -v <HF_CACHE>:/root/.cache/huggingface \
      -v <VLLM_CACHE>:/root/.cache/vllm \
      --entrypoint sleep \
      "$IMAGE" infinity >/dev/null
  fi
}

ensure_container

# Kill any model process inside; leave the container running.
docker exec "$CTR" pkill -f "vllm serve" >/dev/null 2>&1 || true
sleep 3

echo "----------------------------------------"
echo "STAGED. Box is ready for the demo."
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | grep "$CTR"
echo "----------------------------------------"
echo "Now run Claude in your demo folder and say:"
echo "  run <MODEL_SPOKEN> on my Instinct server at <SERVER_IP> with vLLM and give me the endpoint"
