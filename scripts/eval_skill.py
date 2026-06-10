#!/usr/bin/env python3
"""
Eval the serving-llms-on-instinct skill: run claude -p for each model,
save the JSON output.

Installs/uninstalls the skill via ~/.claude/skills/ so there's no
ambiguity about whether it's active.

Usage:
    python3 scripts/eval_skill.py \
      --gpu-host root@165.245.137.144 \
      --mode with-skill

    python3 scripts/eval_skill.py \
      --gpu-host root@165.245.137.144 \
      --mode without-skill \
      --start-from 10
"""

import argparse
import json
import os
import shutil
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

SKILL_INSTALL_DIR = os.path.expanduser("~/.claude/skills/serving-llms-on-instinct")


def install_skill(skill_dir):
    if os.path.exists(SKILL_INSTALL_DIR):
        shutil.rmtree(SKILL_INSTALL_DIR)
    shutil.copytree(skill_dir, SKILL_INSTALL_DIR)
    print(f"Skill installed -> {SKILL_INSTALL_DIR}")


def uninstall_skill():
    if os.path.exists(SKILL_INSTALL_DIR):
        shutil.rmtree(SKILL_INSTALL_DIR)
        print("Skill uninstalled")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-file", default=os.path.expanduser("~/models_to_test.txt"))
    p.add_argument("--gpu-host", required=True)
    p.add_argument("--skill-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", choices=["with-skill", "without-skill"], required=True)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--claude-model", default="claude-opus-4-8")
    p.add_argument("--out-dir", default=os.path.expanduser("~/eval_logs"))
    p.add_argument("--start-from", type=int, default=0)
    args = p.parse_args()

    with open(args.models_file) as f:
        models = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if args.mode == "with-skill":
        if not os.path.exists(os.path.join(args.skill_dir, "SKILL.md")):
            print(f"SKILL.md not found in {args.skill_dir}", file=sys.stderr)
            sys.exit(1)
        install_skill(args.skill_dir)
    else:
        uninstall_skill()

    os.makedirs(args.out_dir, exist_ok=True)
    total = len(models) - args.start_from

    for i, model in enumerate(models):
        if i < args.start_from:
            continue
        safe = model.replace("/", "--")
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] ({i+1-args.start_from}/{total}) {model}", flush=True)

        prompt = PROMPT.format(model=model, host=args.gpu_host, port=args.port)
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--max-turns", str(args.max_turns),
            "--dangerously-skip-permissions",
            "--model", args.claude_model,
        ]

        start = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            elapsed = round(time.time() - start, 1)
            try:
                output = json.loads(r.stdout) if r.stdout.strip() else {}
            except json.JSONDecodeError:
                output = {"raw": r.stdout[:3000]}
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - start, 1)
            output = {"error": "timeout"}
        except FileNotFoundError:
            print("claude CLI not found", file=sys.stderr)
            sys.exit(1)

        log_path = os.path.join(args.out_dir, f"{safe}__{args.mode}.json")
        with open(log_path, "w") as f:
            json.dump({"model": model, "mode": args.mode, "elapsed_s": elapsed,
                        "timestamp": datetime.now().isoformat(), "output": output}, f, indent=2)

        cost = output.get("cost_usd") or output.get("total_cost_usd") or 0
        print(f"  done {elapsed}s ${cost:.2f} -> {log_path}", flush=True)

    print(f"\nAll logs in {args.out_dir}/")


if __name__ == "__main__":
    main()
