#!/usr/bin/env python3
"""
Eval the serving-llms-on-instinct skill: run claude -p for each model,
check if the endpoint came up, save logs. Keep it simple.

Usage:
    python3 scripts/eval_skill.py \
      --gpu-host root@165.245.137.144 \
      --mode with-skill \
      --skill-dir /path/to/serve-llm

    # resume from model #10
    python3 scripts/eval_skill.py \
      --gpu-host root@165.245.137.144 \
      --start-from 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

PROMPT = (
    "Serve {model} for inference on {host}. "
    "The server has AMD Instinct GPUs. Expose the API on port {port}. "
    "After the endpoint is healthy, send a test chat completion request to "
    "verify it produces output. Do not remove the container when done."
)


def ssh(host, cmd, timeout=30):
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return -1, ""


def cleanup(host, port):
    ssh(host, "docker ps -a --filter name=vllm -q | xargs -r docker rm -f")
    ssh(host, f"docker ps -q --filter publish={port} | xargs -r docker rm -f")
    time.sleep(3)


def check_serving(host, port):
    rc, _ = ssh(host, "docker ps --filter name=vllm -q")
    if not _.strip():
        return "no_container"

    rc, _ = ssh(host, f"curl -sf http://localhost:{port}/health")
    if rc != 0:
        return "container_not_healthy"

    rc, out = ssh(host, (
        f"curl -sf -X POST http://localhost:{port}/v1/chat/completions "
        f"-H 'Content-Type: application/json' "
        f"-d '{{\"model\":\"test\",\"messages\":[{{\"role\":\"user\","
        f"\"content\":\"hi\"}}],\"max_tokens\":5}}'"
    ), timeout=30)
    if rc == 0 and out:
        try:
            content = json.loads(out)["choices"][0]["message"]["content"]
            if content.strip():
                return "serving"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        return "health_only"

    return "health_only"


def run_model(model, host, port, cwd, claude_model, max_turns, timeout):
    prompt = PROMPT.format(model=model, host=host, port=port)
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(max_turns),
        "--dangerously-skip-permissions",
        "--model", claude_model,
    ]

    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        elapsed = round(time.time() - start, 1)
        try:
            output = json.loads(r.stdout) if r.stdout.strip() else {}
        except json.JSONDecodeError:
            output = {"raw": r.stdout[:3000]}
        return elapsed, output, None
    except subprocess.TimeoutExpired:
        return round(time.time() - start, 1), {}, "timeout"
    except FileNotFoundError:
        return 0, {}, "claude_not_found"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-file", default=os.path.expanduser("~/models_to_test.txt"))
    p.add_argument("--gpu-host", required=True)
    p.add_argument("--skill-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", choices=["with-skill", "without-skill", "both"], default="both")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--claude-model", default="claude-opus-4-8")
    p.add_argument("--out-dir", default=os.path.expanduser("~/eval_logs"))
    p.add_argument("--start-from", type=int, default=0)
    args = p.parse_args()

    with open(args.models_file) as f:
        models = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if not os.path.exists(os.path.join(args.skill_dir, "SKILL.md")):
        print(f"SKILL.md not found in {args.skill_dir}", file=sys.stderr)
        sys.exit(1)

    no_skill_dir = os.path.join(os.path.dirname(args.skill_dir), "eval-no-skill")
    os.makedirs(no_skill_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # preflight
    rc, _ = ssh(args.gpu_host, "echo ok")
    if rc != 0:
        print(f"Cannot SSH to {args.gpu_host}", file=sys.stderr)
        sys.exit(1)
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        print("claude CLI not found", file=sys.stderr)
        sys.exit(1)

    modes = []
    if args.mode in ("with-skill", "both"):
        modes.append(("with_skill", args.skill_dir))
    if args.mode in ("without-skill", "both"):
        modes.append(("without_skill", no_skill_dir))

    total = len(models[args.start_from:]) * len(modes)
    run = 0
    results = []

    for mode_name, cwd in modes:
        print(f"\n{'='*60}")
        print(f"  {mode_name.upper()}")
        print(f"{'='*60}\n")

        for i, model in enumerate(models):
            if i < args.start_from:
                continue
            run += 1
            safe = model.replace("/", "--")
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] ({run}/{total}) {model} [{mode_name}]", flush=True)

            cleanup(args.gpu_host, args.port)

            elapsed, output, error = run_model(
                model, args.gpu_host, args.port, cwd,
                args.claude_model, args.max_turns, args.timeout,
            )

            if error:
                status = error
            else:
                status = check_serving(args.gpu_host, args.port)

            cost = output.get("cost_usd") or output.get("total_cost_usd")
            session_id = output.get("session_id", "")

            entry = {
                "model": model,
                "mode": mode_name,
                "status": status,
                "elapsed_s": elapsed,
                "cost_usd": cost,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "claude_output": output,
            }
            results.append(entry)

            # save per-run log
            log_path = os.path.join(args.out_dir, f"{safe}__{mode_name}.json")
            with open(log_path, "w") as f:
                json.dump(entry, f, indent=2)

            mark = "PASS" if status == "serving" else status.upper()
            print(f"  -> {mark}  ({elapsed}s, ${cost or 0:.2f})", flush=True)

            cleanup(args.gpu_host, args.port)

    # final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for mode_name, _ in modes:
        mr = [r for r in results if r["mode"] == mode_name]
        served = sum(1 for r in mr if r["status"] == "serving")
        print(f"\n  {mode_name}: {served}/{len(mr)} serving")
        for r in mr:
            mark = "PASS" if r["status"] == "serving" else r["status"]
            print(f"    {mark:25s} {r['model']}")

    # save all results
    all_path = os.path.join(args.out_dir, "all_results.json")
    with open(all_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLogs in {args.out_dir}/")
    print(f"Full results: {all_path}")


if __name__ == "__main__":
    main()
