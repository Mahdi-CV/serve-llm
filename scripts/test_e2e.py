#!/usr/bin/env python3
"""
End-to-end test harness for the serving-llms-on-instinct skill.

Iterates through a model list, constructs Docker commands using the same
data files the agent reads (recipes_cache.json, gpu_overrides.json,
blacklist.json), launches each model on a target GPU server, verifies
health and inference, then tears down.

Usage:
    python3 scripts/test_e2e.py --models-file ~/models_to_test.txt --host root@10.0.0.5
    python3 scripts/test_e2e.py --models-file ~/models_to_test.txt --host root@10.0.0.5 --timeout 25
    python3 scripts/test_e2e.py --dry-run --models-file ~/models_to_test.txt

Output: JSON results file with pass/fail/skip per model.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# VRAM estimate: params_billions * 2 GB * 1.15 overhead (BF16)
PARAM_MULTIPLIER = 2.3


def load_data():
    with open(DATA_DIR / "recipes_cache.json") as f:
        recipes = json.load(f)
    with open(DATA_DIR / "gpu_overrides.json") as f:
        overrides = json.load(f)
    with open(DATA_DIR / "blacklist.json") as f:
        blacklist = json.load(f)
    return recipes, overrides, blacklist


def get_blacklisted_models(blacklist):
    models = set()
    for category in blacklist.values():
        if isinstance(category, dict) and "models" in category:
            models.update(category["models"])
    return models


def get_blacklist_reason(model_id, blacklist):
    for cat_key, category in blacklist.items():
        if isinstance(category, dict) and "models" in category:
            if model_id in category["models"]:
                return category.get("_comment", cat_key)
    return "blacklisted"


def estimate_vram_gb(recipe_entry):
    if not recipe_entry:
        return None
    model_info = recipe_entry.get("model_info", {})
    param_str = model_info.get("parameter_count", "")
    if not param_str:
        recipe = recipe_entry.get("recipe", {})
        model = recipe.get("model", {})
        param_str = model.get("parameter_count", "")
    if not param_str:
        return None
    match = re.search(r"([\d.]+)\s*[Bb]", param_str)
    if match:
        params_b = float(match.group(1))
        return round(params_b * PARAM_MULTIPLIER)
    return None


def get_min_tp(recipe_entry):
    if not recipe_entry:
        return 1
    recipe = recipe_entry.get("recipe", {})
    variants = recipe.get("variants", {})
    for key, variant in variants.items():
        if "tp" in key.lower() or "single" in key.lower():
            args = variant.get("args", [])
            for i, arg in enumerate(args):
                if arg == "--tensor-parallel-size" and i + 1 < len(args):
                    try:
                        return int(args[i + 1])
                    except ValueError:
                        pass
    strategies = recipe.get("compatible_strategies", [])
    if strategies and all("multi_node" in s for s in strategies):
        return 2
    return 1


def slugify(model_id):
    slug = re.sub(r"[^a-zA-Z0-9]", "-", model_id)[:40].strip("-").lower()
    return slug


def _run(cmd, host=None, user=None, port=22, timeout=30):
    try:
        if not host or host in ("local", "localhost", "127.0.0.1"):
            r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        else:
            ssh_target = f"{user}@{host}" if user else host
            ssh = [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15",
                "-o", "BatchMode=yes",
                "-o", "LogLevel=ERROR",
                "-p", str(port),
                ssh_target, cmd,
            ]
            r = subprocess.run(ssh, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Timed out after {timeout}s"


def detect_gpu(host, user, port):
    rc, out, err = _run(
        f"python3 {SCRIPT_DIR / 'detect.py'}" if not host else "amd-smi static --asic --vram --json",
        host, user, port, timeout=30
    )
    if rc != 0 and host:
        rc, out, err = _run("sudo amd-smi static --asic --vram --json", host, user, port, timeout=30)
    if rc != 0:
        return None

    try:
        if host:
            data = json.loads(out)
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict):
                gpu_list = data.get("gpu_data", [data])
            else:
                gpu_list = [data]
            if not gpu_list:
                return None
            asic = gpu_list[0].get("asic", {})
            vram_info = gpu_list[0].get("vram", {})
            vram_size = vram_info.get("size", {})
            vram_mb = vram_size.get("value") if isinstance(vram_size, dict) else vram_size
            return {
                "gfx_version": asic.get("target_graphics_version", "unknown").lower(),
                "vram_gb": round(vram_mb / 1024, 1) if vram_mb else 192,
                "gpu_count": len(gpu_list),
            }
        else:
            data = json.loads(out)
            return {
                "gfx_version": data.get("gfx_version", "unknown"),
                "vram_gb": data["gpus"][0]["vram_gb"] if data.get("gpus") else 192,
                "gpu_count": data.get("gpu_count", 1),
            }
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def build_docker_cmd(model_id, recipe_entry, overrides, docker_image, gfx_version, port, host, user, hf_cache):
    gpu_config = overrides.get("gpu_configs", {}).get(gfx_version, {})
    docker_flags = overrides.get("docker_flags", [])

    env_vars = {}
    env_vars.update(gpu_config.get("env_defaults", {}))

    vllm_args = []

    if recipe_entry:
        recipe = recipe_entry.get("recipe", {})
        model = recipe.get("model", {})
        env_vars.update(model.get("base_env", {}))
        hw = recipe.get("hardware_overrides", {}).get("amd", {})
        env_vars.update(hw.get("extra_env", {}))

        base_args = model.get("base_args", [])
        vllm_args.extend(base_args)
        vllm_args.extend(hw.get("extra_args", []))

        features = recipe.get("features", {})
        for feat_key, feat_val in features.items():
            if isinstance(feat_val, dict) and "args" in feat_val:
                vllm_args.extend(feat_val["args"])
    else:
        legacy = overrides.get("legacy_models", {}).get(model_id, {})
        if legacy:
            env_vars.update(legacy.get("env_vars", {}))
            vllm_args.extend(legacy.get("vllm_args", []))
            parser = legacy.get("tool_call_parser", "hermes")
            vllm_args.extend(["--tool-call-parser", parser])

    has_tool_choice = any("enable-auto-tool-choice" in str(a) for a in vllm_args)
    if not has_tool_choice:
        vllm_args.append("--enable-auto-tool-choice")

    has_trust = any("trust-remote-code" in str(a) for a in vllm_args)
    if not has_trust:
        vllm_args.append("--trust-remote-code")

    if not recipe_entry and model_id not in overrides.get("legacy_models", {}):
        has_parser = any("tool-call-parser" in str(a) for a in vllm_args)
        if not has_parser:
            vllm_args.extend(["--tool-call-parser", "hermes"])

    slug = slugify(model_id)
    container_name = f"vllm-test-{slug}"

    # Pair up consecutive args that look like --flag value
    # Quote values that contain special characters (JSON, spaces)
    paired_args = []
    i = 0
    while i < len(vllm_args):
        arg = str(vllm_args[i])
        if arg.startswith("--") and i + 1 < len(vllm_args) and not str(vllm_args[i + 1]).startswith("--"):
            val = str(vllm_args[i + 1])
            if any(c in val for c in '{}[] "'):
                val = f"'{val}'"
            paired_args.append(f"{arg} {val}")
            i += 2
        else:
            paired_args.append(arg)
            i += 1

    parts = ["docker run -d"]
    parts.append(f"--name {container_name}")
    for flag in docker_flags:
        parts.append(flag)
    parts.append(f"-v {hf_cache}:/root/.cache/huggingface")
    parts.append(f"-p {port}:{port}")
    for k, v in env_vars.items():
        parts.append(f"--env {k}={v}")
    parts.append("--env HF_TOKEN=${HF_TOKEN}")
    parts.append(docker_image)
    parts.append(f"--model {model_id}")
    for arg in paired_args:
        parts.append(arg)
    parts.append(f"--port {port}")

    return container_name, " ".join(parts)


def run_test(model_id, docker_cmd, container_name, host, user, port, ssh_port, timeout_min):
    result = {
        "model": model_id,
        "container": container_name,
        "docker_cmd": docker_cmd,
    }

    # Stop any existing container with same name
    _run(f"docker rm -f {container_name} 2>/dev/null", host, user, ssh_port, timeout=15)
    time.sleep(2)

    # Check port
    rc, out, _ = _run(f"ss -tlnp 2>/dev/null | grep ':{port} '", host, user, ssh_port, timeout=10)
    if out.strip():
        _run(f"docker ps --filter 'publish={port}' -q | xargs -r docker rm -f", host, user, ssh_port, timeout=15)
        time.sleep(2)

    # Launch
    print(f"  Launching container...")
    t_start = time.time()
    rc, out, err = _run(docker_cmd, host, user, ssh_port, timeout=120)
    if rc != 0:
        result["status"] = "fail"
        result["reason"] = f"Docker launch failed: {err[:300]}"
        return result

    # Poll health
    print(f"  Waiting for /health (timeout: {timeout_min}min)...")
    deadline = time.time() + timeout_min * 60
    healthy = False
    while time.time() < deadline:
        rc, out, _ = _run(f"curl -sf http://localhost:{port}/health", host, user, ssh_port, timeout=10)
        if rc == 0:
            healthy = True
            break
        time.sleep(15)

    load_time = round(time.time() - t_start)

    if not healthy:
        # Grab logs for debugging
        _, logs, _ = _run(f"docker logs {container_name} 2>&1 | tail -30", host, user, ssh_port, timeout=15)
        result["status"] = "fail"
        result["reason"] = f"Health check timed out after {timeout_min}min"
        result["load_time_s"] = load_time
        result["logs_tail"] = logs[:500]
        _run(f"docker rm -f {container_name}", host, user, ssh_port, timeout=15)
        return result

    result["load_time_s"] = load_time
    print(f"  Healthy after {load_time}s. Sending inference request...")

    # Inference test
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 20,
    })
    t_infer = time.time()
    rc, out, err = _run(
        f'curl -s http://localhost:{port}/v1/chat/completions '
        f'-H "Content-Type: application/json" '
        f"-d '{payload}'",
        host, user, ssh_port, timeout=120
    )
    infer_time = round(time.time() - t_infer, 1)

    if rc != 0:
        result["status"] = "fail"
        result["reason"] = f"Inference request failed: {err[:200]}"
        result["inference_time_s"] = infer_time
    else:
        try:
            resp = json.loads(out)
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                result["status"] = "pass"
                result["inference_time_s"] = infer_time
                result["response_preview"] = content[:100]
            else:
                result["status"] = "fail"
                result["reason"] = f"Empty response: {out[:200]}"
                result["inference_time_s"] = infer_time
        except json.JSONDecodeError:
            result["status"] = "fail"
            result["reason"] = f"Invalid JSON response: {out[:200]}"
            result["inference_time_s"] = infer_time

    # Teardown
    print(f"  Tearing down container...")
    _run(f"docker rm -f {container_name}", host, user, ssh_port, timeout=15)
    time.sleep(5)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="E2E test harness for serving-llms-on-instinct skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-file", required=True, help="File with one HF model ID per line")
    parser.add_argument("--host", default="", help="[user@]host for remote GPU server (default: local)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port (default: 8000)")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--results-file", default="test_results.json", help="Output JSON results file")
    parser.add_argument("--timeout", type=int, default=20, help="Max minutes to wait for health per model")
    parser.add_argument("--vram-gb", type=int, default=0, help="Available VRAM in GB (auto-detected if 0)")
    parser.add_argument("--gfx-version", default="", help="GPU arch (auto-detected if empty)")
    parser.add_argument("--hf-cache", default="~/.cache/huggingface", help="HF cache path on target")
    parser.add_argument("--dry-run", action="store_true", help="Construct commands but don't launch")
    parser.add_argument("--skip-teardown", action="store_true", help="Leave containers running (debug)")
    args = parser.parse_args()

    host = args.host
    user = ""
    if "@" in host:
        user, host = host.split("@", 1)

    # Load data files
    recipes, overrides, blacklist = load_data()
    blacklisted = get_blacklisted_models(blacklist)
    docker_image = recipes.get("docker_image", "vllm/vllm-openai-rocm:latest")

    # Load model list
    with open(os.path.expanduser(args.models_file)) as f:
        models = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Loaded {len(models)} models from {args.models_file}")

    # Detect GPU
    gpu_info = None
    if args.gfx_version and args.vram_gb:
        gpu_info = {"gfx_version": args.gfx_version, "vram_gb": args.vram_gb, "gpu_count": 1}
    else:
        print("Detecting GPU...")
        gpu_info = detect_gpu(host, user, args.ssh_port)

    if not gpu_info:
        print("ERROR: Could not detect GPU. Use --gfx-version and --vram-gb to specify manually.")
        sys.exit(1)

    gfx = gpu_info["gfx_version"]
    vram = args.vram_gb if args.vram_gb else gpu_info["vram_gb"]
    gpu_count = gpu_info["gpu_count"]
    print(f"GPU: {gfx}, VRAM: {vram}GB, Count: {gpu_count}")

    # Expand HF cache
    hf_cache = args.hf_cache
    if host:
        # For remote, check for /home/amd/models
        rc, out, _ = _run("test -d /home/amd/models && echo yes || echo no", host, user, args.ssh_port)
        if out.strip() == "yes":
            hf_cache = "/home/amd/models"
            print(f"Using shared model cache: {hf_cache}")

    results = []
    counts = {"pass": 0, "fail": 0, "skip": 0}

    for i, model_id in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model_id}")

        # Check blacklist
        if model_id in blacklisted:
            reason = get_blacklist_reason(model_id, blacklist)
            print(f"  SKIP (blacklisted): {reason}")
            results.append({"model": model_id, "status": "skip", "reason": f"Blacklisted: {reason}", "source": "blacklist"})
            counts["skip"] += 1
            continue

        # Look up config
        recipe_entry = recipes.get("models", {}).get(model_id)
        source = "recipes" if recipe_entry else None

        if not source:
            if model_id in overrides.get("legacy_models", {}):
                source = "legacy"
            else:
                source = "generic"

        # Estimate VRAM
        vram_needed = None
        if source == "recipes":
            vram_needed = estimate_vram_gb(recipe_entry)
        elif source == "legacy":
            vram_needed = overrides["legacy_models"][model_id].get("vram_fp16_gb")

        if vram_needed and vram_needed > vram:
            print(f"  SKIP: Needs ~{vram_needed}GB VRAM (available: {vram}GB)")
            results.append({
                "model": model_id, "status": "skip",
                "reason": f"Needs ~{vram_needed}GB VRAM (available: {vram}GB)",
                "source": source, "vram_needed_gb": vram_needed,
            })
            counts["skip"] += 1
            continue

        # Check TP requirements
        min_tp = 1
        if source == "recipes":
            min_tp = get_min_tp(recipe_entry)
        elif source == "legacy":
            min_tp = overrides["legacy_models"][model_id].get("min_tp", 1)

        if min_tp > gpu_count:
            print(f"  SKIP: Needs TP={min_tp} (available: {gpu_count} GPUs)")
            results.append({
                "model": model_id, "status": "skip",
                "reason": f"Needs TP={min_tp} ({gpu_count} GPUs available)",
                "source": source, "min_tp": min_tp,
            })
            counts["skip"] += 1
            continue

        # Build Docker command
        container_name, docker_cmd = build_docker_cmd(
            model_id, recipe_entry, overrides, docker_image, gfx, args.port, host, user, hf_cache
        )

        print(f"  Source: {source}")
        if vram_needed:
            print(f"  VRAM estimate: ~{vram_needed}GB")

        if args.dry_run:
            print(f"  DRY RUN -- command:")
            print(f"    {docker_cmd}")
            results.append({
                "model": model_id, "status": "dry_run",
                "source": source, "docker_cmd": docker_cmd,
            })
            continue

        # Run the test
        test_result = run_test(
            model_id, docker_cmd, container_name,
            host, user, args.port, args.ssh_port, args.timeout,
        )
        test_result["source"] = source
        if vram_needed:
            test_result["vram_estimate_gb"] = vram_needed

        status = test_result["status"]
        counts[status] = counts.get(status, 0) + 1

        if status == "pass":
            print(f"  PASS (load: {test_result.get('load_time_s')}s, infer: {test_result.get('inference_time_s')}s)")
            print(f"  Response: {test_result.get('response_preview', '')[:80]}")
        else:
            print(f"  FAIL: {test_result.get('reason', 'unknown')[:100]}")

        results.append(test_result)

        # Write intermediate results after each model
        output = {
            "test_run": datetime.now(timezone.utc).isoformat(),
            "server": f"{user}@{host}" if user else (host or "local"),
            "gpu": {"gfx_version": gfx, "vram_gb": vram, "gpu_count": gpu_count},
            "docker_image": docker_image,
            "results": results,
            "summary": {"total": len(models), **counts},
        }
        with open(args.results_file, "w") as f:
            json.dump(output, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {counts.get('pass', 0)} pass, {counts.get('fail', 0)} fail, {counts.get('skip', 0)} skip")
    print(f"Results written to: {args.results_file}")


if __name__ == "__main__":
    main()
