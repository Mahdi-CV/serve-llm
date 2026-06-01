#!/usr/bin/env python3
"""
Run a model on AMD Instinct GPU hardware end-to-end.

Orchestrates the full flow:
  1. Detect GPU (gfx_version, VRAM)
  2. Validate environment (devices, Docker, NUMA)
  3. Get the correct vLLM Docker command
  4. Launch the container
  5. Poll /health until ready
  6. Send a warmup inference request
  7. Return the endpoint URL

Usage:
    python scripts/run_model.py --model Qwen/Qwen3-8B
    python scripts/run_model.py --model Qwen/Qwen3-8B --host root@10.0.0.5
    python scripts/run_model.py --model deepseek-ai/DeepSeek-R1 --gpu-count 8
    python scripts/run_model.py --dry-run --model Qwen/Qwen3-72B

Options:
    --dry-run     Print the Docker command without launching
    --port        Port to expose (default: 8000)
    --gpu-count   Number of GPUs (default: auto-detected)
    --no-warmup   Skip warmup inference after health check passes

Env vars:
    ROCM_SSH_HOST, ROCM_SSH_USER, ROCM_SSH_PORT
    HF_TOKEN (for gated models)
"""

import argparse
import json
import os
import subprocess
import sys
import time


def _is_local(host):
    return not host or host in ("local", "localhost", "127.0.0.1")


def _run(cmd, host, user, port, timeout=30):
    if _is_local(host):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    else:
        ssh = [
            "ssh",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            "-p", str(port),
            f"{user}@{host}", cmd,
        ]
        r = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _run_script(script, extra_args, host, user, port):
    """Run one of the sibling scripts and return parsed JSON output."""
    script_path = os.path.join(_script_dir(), script)
    host_arg = f"--host {user}@{host}" if not _is_local(host) else ""
    cmd = f"python {script_path} {host_arg} {extra_args}"
    rc, out, err = _run(cmd, host="", user="", port=22, timeout=60)  # always local -- scripts handle their own SSH
    if rc != 0:
        return None, err or out
    try:
        return json.loads(out), None
    except json.JSONDecodeError:
        return None, f"Script output is not valid JSON: {out[:200]}"


def step(n, msg):
    print(f"\n[{n}] {msg}", flush=True)


def ok(msg):
    print(f"    OK  {msg}", flush=True)


def warn(msg):
    print(f"    WARN {msg}", flush=True)


def fail(msg):
    print(f"    FAIL {msg}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model ID (default: Qwen/Qwen3-8B)")
    parser.add_argument("--host", default="", help="[user@]host (default: local or ROCM_SSH_HOST)")
    parser.add_argument("--user", default="", help="SSH user (default: root)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--ssh-port", type=int, default=0, help="SSH port (default: 22)")
    parser.add_argument("--gpu-count", type=int, default=0, help="Number of GPUs (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Print Docker command without launching")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup inference")
    parser.add_argument("--health-timeout", type=int, default=900, help="Max seconds to wait for /health (default: 900)")
    args = parser.parse_args()

    host = args.host
    user = args.user
    if "@" in host:
        user, host = host.split("@", 1)
    host = host or os.environ.get("ROCM_SSH_HOST", "")
    user = user or os.environ.get("ROCM_SSH_USER", "root")
    ssh_port = args.ssh_port or int(os.environ.get("ROCM_SSH_PORT", "22"))

    target = f"{user}@{host}" if not _is_local(host) else "local"
    print(f"\nAMD model run: {args.model}")
    print(f"Target: {target}  Port: {args.port}")

    # ------------------------------------------------------------------
    # Step 1: Detect GPU
    # ------------------------------------------------------------------
    step(1, "Detecting GPU hardware...")
    detect_args = f"--model {args.model}"  # detect.py doesn't take model but pass host
    detect_data, err = _run_script("detect.py", "", host, user, ssh_port)
    if err or not detect_data:
        fail(f"GPU detection failed: {err}")
        sys.exit(1)

    gfx = detect_data.get("gfx_version", "unknown")
    gpu_count = args.gpu_count or detect_data.get("gpu_count", 1)
    rocm = detect_data.get("rocm_version", "unknown")

    if gfx == "unknown":
        fail("gfx_version is unknown. Is the amdgpu kernel module loaded? Try: lsmod | grep amdgpu")
        sys.exit(1)

    ok(f"GPU: {detect_data['gpus'][0]['market_name']}  gfx={gfx}  VRAM={detect_data['gpus'][0].get('vram_gb','?')}GB  ROCm={rocm}")
    if gpu_count > 1:
        ok(f"GPU count: {gpu_count}")

    # ------------------------------------------------------------------
    # Step 2: Validate environment
    # ------------------------------------------------------------------
    step(2, "Validating environment...")
    val_data, err = _run_script("validate.py", "--auto-fix", host, user, ssh_port)
    if err or not val_data:
        fail(f"Environment validation failed: {err}")
        sys.exit(1)

    for issue in val_data.get("errors", []):
        fail(f"{issue['check']}: {issue['message']}")
        print(f"    Fix: {issue['fix']}", file=sys.stderr)

    if not val_data.get("ready", False):
        fail("Environment has blocking errors. Fix them and retry.")
        sys.exit(1)

    for w in val_data.get("warnings", []):
        warn(f"{w['check']}: {w['message']}")
    for fix in val_data.get("fixes_applied", []):
        ok(f"Auto-fixed: {fix}")

    ok("Environment ready")

    # ------------------------------------------------------------------
    # Step 3: Get Docker config
    # ------------------------------------------------------------------
    step(3, "Resolving vLLM configuration...")
    cfg_data, err = _run_script(
        "get_config.py",
        f"--gfx {gfx} --model {args.model} --gpu-count {gpu_count} --port {args.port}",
        host="", user="", port=0,  # get_config is always local
    )
    if err or not cfg_data:
        fail(f"Config lookup failed: {err}")
        sys.exit(1)

    if "error" in cfg_data:
        fail(cfg_data["error"])
        sys.exit(1)

    if cfg_data.get("vram_warning"):
        warn(cfg_data["vram_warning"])

    if not cfg_data.get("model_known"):
        warn(f"Model '{args.model}' not in verified config list. Using generic config. Results may vary.")

    for w in cfg_data.get("workarounds", []):
        ok(f"Workaround applied: {w['description']}")

    ok(f"Config: {cfg_data['gpu_family']}  TP={cfg_data['tensor_parallel']}  confidence={cfg_data['confidence']}")

    docker_cmd = cfg_data["docker_command"]

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN -- Docker command (not executed):")
        print(f"{'='*60}")
        print(docker_cmd)
        print(f"{'='*60}\n")
        return

    # ------------------------------------------------------------------
    # Step 4: Launch container
    # ------------------------------------------------------------------
    step(4, "Launching vLLM container...")

    # Stop any existing container on this port first
    model_short = args.model.split("/")[-1].lower()[:20]
    container_name = f"vllm-{model_short}"
    _run(f"docker rm -f {container_name} 2>/dev/null || true", host, user, ssh_port, timeout=15)

    rc, container_id, err = _run(docker_cmd, host, user, ssh_port, timeout=60)
    if rc != 0:
        fail(f"Docker launch failed: {err}")
        print(f"\nDocker command was:\n{docker_cmd}", file=sys.stderr)
        sys.exit(1)

    ok(f"Container started: {container_id[:12] if container_id else container_name}")

    # ------------------------------------------------------------------
    # Step 5: Poll /health
    # ------------------------------------------------------------------
    step(5, f"Waiting for /health to return 200 (up to {args.health_timeout}s)...")

    health_url = f"http://localhost:{args.port}/health"
    if not _is_local(host):
        health_url = f"http://{host}:{args.port}/health"

    poll_interval = 10
    elapsed = 0
    ready = False
    last_status = ""

    while elapsed < args.health_timeout:
        rc, out, _ = _run(
            f"curl -s -o /dev/null -w '%{{http_code}}' {health_url}",
            host, user, ssh_port, timeout=15,
        )
        status = out.strip()

        if status == "200":
            ready = True
            break

        if status != last_status:
            print(f"    {elapsed}s: /health returned {status or 'no response'}", flush=True)
            last_status = status

        # Check container is still running
        if elapsed > 30:
            rc2, running, _ = _run(
                f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'",
                host, user, ssh_port, timeout=10,
            )
            if not running.strip():
                fail("Container stopped unexpectedly. Check logs:")
                _run(f"docker logs --tail 50 {container_name}", host, user, ssh_port, timeout=10)
                rc3, logs, _ = _run(f"docker logs --tail 50 {container_name}", host, user, ssh_port, timeout=10)
                print(logs, file=sys.stderr)
                sys.exit(1)

        time.sleep(poll_interval)
        elapsed += poll_interval

    if not ready:
        fail(f"/health did not return 200 after {args.health_timeout}s")
        fail("Check container logs: docker logs " + container_name)
        sys.exit(1)

    ok(f"/health returned 200 after {elapsed}s")

    # ------------------------------------------------------------------
    # Step 6: Warmup inference
    # ------------------------------------------------------------------
    if not args.no_warmup:
        step(6, "Sending warmup inference (triggers HIP kernel compilation)...")
        models_url = f"http://localhost:{args.port}/v1/models"
        if not _is_local(host):
            models_url = f"http://{host}:{args.port}/v1/models"

        warmup_payload = json.dumps({
            "model": args.model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        })
        chat_url = models_url.replace("/v1/models", "/v1/chat/completions")
        warmup_cmd = (
            f"curl -s -o /dev/null -w '%{{http_code}}' "
            f"-X POST {chat_url} "
            f"-H 'Content-Type: application/json' "
            f"-d '{warmup_payload}'"
        )
        rc, status, _ = _run(warmup_cmd, host, user, ssh_port, timeout=120)
        if status == "200":
            ok("Warmup complete. First real request will be fast.")
        else:
            warn(f"Warmup returned {status}. Endpoint is up but first request may be slow (HIP kernel compile).")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    base_url = f"http://localhost:{args.port}/v1"
    if not _is_local(host):
        base_url = f"http://{host}:{args.port}/v1"

    print(f"\n{'='*60}")
    print("ENDPOINT READY")
    print(f"{'='*60}")
    print(f"  base_url:  {base_url}")
    print(f"  model:     {args.model}")
    print(f"  api_key:   (none required)")
    print(f"  gpu:       {detect_data['gpus'][0]['market_name']} ({gfx})")
    print(f"{'='*60}\n")

    # Output JSON for agent parsing
    print(json.dumps({
        "status": "ready",
        "base_url": base_url,
        "model": args.model,
        "gfx_version": gfx,
        "gpu": detect_data["gpus"][0]["market_name"],
        "tensor_parallel": cfg_data["tensor_parallel"],
        "container_name": container_name,
    }))


if __name__ == "__main__":
    main()
