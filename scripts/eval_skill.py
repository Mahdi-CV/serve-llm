#!/usr/bin/env python3
"""
Evaluate the serving-llms-on-instinct skill by comparing Claude Code's
ability to serve models WITH vs WITHOUT the skill.

Runs claude -p for each model in two modes:
  - with-skill:    CWD = skill repo (has SKILL.md, data/, scripts/)
  - without-skill: CWD = empty directory (Claude uses general knowledge)

Measures success rate, turns, tool calls, wall time, and cost.

Usage:
    # Run both modes for all models
    python3 scripts/eval_skill.py \\
      --models-file ~/models_to_test.txt \\
      --gpu-host root@165.245.137.144 \\
      --skill-dir /path/to/serve-llm \\
      --mode both

    # Run just with-skill, starting from model #10
    python3 scripts/eval_skill.py \\
      --models-file ~/models_to_test.txt \\
      --gpu-host root@165.245.137.144 \\
      --skill-dir /path/to/serve-llm \\
      --mode with-skill \\
      --start-from 10

Requires:
    - claude CLI installed and authenticated
    - SSH access to the GPU server (key-based, no password)
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_PORT = 8000
DEFAULT_MAX_TURNS = 30
DEFAULT_TIMEOUT = 1800  # 30 minutes per model
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
NO_SKILL_DIR_NAME = "eval-no-skill"

PROMPT_TEMPLATE = (
    "Serve {model_id} for inference on {gpu_host}. "
    "The server has AMD Instinct GPUs. Expose the API on port {port}. "
    "After the endpoint is healthy, send a test chat completion request to "
    "verify it produces output. Do not remove the container when done."
)

SKIP_KEYWORDS = [
    "skip", "cannot", "won't fit", "doesn't fit", "does not fit",
    "not supported", "not an llm", "embedding", "reranker", "too large",
    "insufficient", "nvfp4", "not available", "not compatible",
    "encoder-decoder", "not a language model", "cannot be served",
    "decline", "unable to serve", "won't work", "image generation",
    "audio", "diffusion", "exceeds",
]

# Files that indicate the skill was actually used (not just present).
SKILL_ARTIFACTS = [
    "SKILL.md", "reference.md",
    "detect.py", "validate.py", "estimate_vram.py", "sync_recipes.py",
    "recipes_cache.json", "gpu_overrides.json", "blacklist.json",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ssh_run(gpu_host, cmd, timeout=30):
    """Run a command on the GPU server via SSH."""
    full = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", gpu_host, cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "SSH timeout"
    except Exception as e:
        return -1, "", str(e)


def cleanup_containers(gpu_host, port):
    """Remove all vllm containers on the GPU server."""
    ssh_run(gpu_host, "docker ps -a --filter name=vllm -q | xargs -r docker rm -f", timeout=30)
    ssh_run(gpu_host, f"docker ps -q --filter publish={port} | xargs -r docker rm -f", timeout=15)
    time.sleep(5)


def check_ground_truth(gpu_host, port):
    """Check if a vLLM container is running and healthy on the GPU server."""
    result = {"container": False, "health": False, "inference": False}

    rc, out, _ = ssh_run(gpu_host, "docker ps --filter name=vllm -q", timeout=10)
    result["container"] = bool(out.strip())
    if not result["container"]:
        return result

    rc, out, _ = ssh_run(gpu_host, f"curl -sf http://localhost:{port}/health", timeout=10)
    result["health"] = rc == 0
    if not result["health"]:
        return result

    infer_cmd = (
        f"curl -sf -X POST http://localhost:{port}/v1/chat/completions "
        f"-H 'Content-Type: application/json' "
        f"-d '{{\"model\":\"test\",\"messages\":[{{\"role\":\"user\","
        f"\"content\":\"hi\"}}],\"max_tokens\":5}}'"
    )
    rc, out, _ = ssh_run(gpu_host, infer_cmd, timeout=30)
    if rc == 0 and out:
        try:
            resp = json.loads(out)
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            result["inference"] = bool(content.strip())
            result["inference_preview"] = content.strip()[:200]
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    return result


def encode_cwd(path):
    """Encode a path the way Claude Code does for project directories."""
    return re.sub(r"[^a-zA-Z0-9]", "-", path.rstrip("/"))


def find_transcript(session_id, cwd):
    """Find the transcript JSONL for a given session ID."""
    encoded = encode_cwd(cwd)
    base = os.path.expanduser(f"~/.claude/projects/{encoded}")

    exact = os.path.join(base, f"{session_id}.jsonl")
    if os.path.exists(exact):
        return exact

    pattern = os.path.join(base, "*.jsonl")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getmtime)

    return None


def parse_transcript(transcript_path):
    """Parse transcript JSONL for metrics.

    Returns counts for turns, tool calls (total and by name), failed tool
    calls, and which skill artifacts were accessed.
    """
    metrics = {
        "turns": 0,
        "tool_calls": 0,
        "failed_tool_calls": 0,
        "tool_breakdown": {},  # tool_name -> count
        "skill_files_read": [],  # which SKILL_ARTIFACTS were Read
        "skill_triggered": False,
    }
    if not transcript_path or not os.path.exists(transcript_path):
        return metrics

    seen_skill_files = set()

    try:
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = entry.get("role", "")
                if role == "assistant":
                    metrics["turns"] += 1

                content = entry.get("content", [])
                if not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue

                    # Count tool uses
                    if block.get("type") == "tool_use":
                        metrics["tool_calls"] += 1
                        name = block.get("name", "unknown")
                        metrics["tool_breakdown"][name] = (
                            metrics["tool_breakdown"].get(name, 0) + 1
                        )

                        # Detect skill file reads
                        if name == "Read":
                            inp = block.get("input", {})
                            fpath = inp.get("file_path", "")
                            for artifact in SKILL_ARTIFACTS:
                                if artifact in fpath:
                                    seen_skill_files.add(artifact)

                        # Detect skill script execution via Bash
                        if name == "Bash":
                            inp = block.get("input", {})
                            cmd = inp.get("command", "")
                            for artifact in SKILL_ARTIFACTS:
                                if artifact in cmd:
                                    seen_skill_files.add(artifact)

                    # Count failed tool results
                    if block.get("type") == "tool_result":
                        if block.get("is_error"):
                            metrics["failed_tool_calls"] += 1

    except Exception:
        pass

    metrics["skill_files_read"] = sorted(seen_skill_files)
    metrics["skill_triggered"] = bool(seen_skill_files)

    return metrics


def run_claude(prompt, cwd, max_turns, claude_model, timeout):
    """Run claude -p and return parsed output."""
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(max_turns),
        "--dangerously-skip-permissions",
        "--model", claude_model,
    ]

    start = time.time()
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        elapsed = time.time() - start

        output = {}
        if r.stdout.strip():
            try:
                output = json.loads(r.stdout)
            except json.JSONDecodeError:
                output = {"raw_stdout": r.stdout[:3000]}

        result_text = ""
        if isinstance(output, dict):
            result_text = (
                output.get("result", "")
                or output.get("content", "")
                or output.get("text", "")
                or ""
            )
        if isinstance(result_text, list):
            parts = []
            for block in result_text:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            result_text = "\n".join(parts)

        return {
            "elapsed_s": round(elapsed, 1),
            "exit_code": r.returncode,
            "session_id": output.get("session_id", ""),
            "cost_usd": output.get("total_cost_usd") or output.get("cost_usd"),
            "usage": output.get("usage", {}),
            "result_text": str(result_text)[:2000],
            "error": None,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "elapsed_s": round(elapsed, 1),
            "exit_code": -1,
            "session_id": "",
            "cost_usd": None,
            "usage": {},
            "result_text": "",
            "error": "timeout",
        }
    except FileNotFoundError:
        return {
            "elapsed_s": 0,
            "exit_code": -1,
            "session_id": "",
            "cost_usd": None,
            "usage": {},
            "result_text": "",
            "error": "claude CLI not found",
        }


def classify_outcome(ground_truth, result_text, error):
    """Classify the outcome of a run."""
    if error == "timeout":
        return "timeout"
    if error == "claude CLI not found":
        return "error"

    if ground_truth["inference"]:
        return "served"
    if ground_truth["health"]:
        return "health_only"
    if ground_truth["container"]:
        return "container_stuck"

    text_lower = result_text.lower()
    if any(kw in text_lower for kw in SKIP_KEYWORDS):
        return "correct_skip"

    return "failed"


def run_single(model_id, index, with_skill, gpu_host, port, skill_dir,
               no_skill_dir, max_turns, claude_model, timeout):
    """Full eval cycle for one model in one mode."""
    mode = "with_skill" if with_skill else "without_skill"
    cwd = skill_dir if with_skill else no_skill_dir

    log(f"[{index}] {model_id} | {mode}")
    log(f"  Cleaning up...")
    cleanup_containers(gpu_host, port)

    prompt = PROMPT_TEMPLATE.format(
        model_id=model_id, gpu_host=gpu_host, port=port,
    )

    log(f"  Running claude -p (max {max_turns} turns, {timeout}s timeout)...")
    run_result = run_claude(prompt, cwd, max_turns, claude_model, timeout)
    log(f"  Done in {run_result['elapsed_s']}s (exit={run_result['exit_code']})")

    if run_result["error"]:
        log(f"  Error: {run_result['error']}")

    log(f"  Checking ground truth on {gpu_host}...")
    ground_truth = check_ground_truth(gpu_host, port)
    log(f"  Container={ground_truth['container']} Health={ground_truth['health']} "
        f"Inference={ground_truth['inference']}")

    transcript_metrics = {}
    if run_result["session_id"]:
        tp = find_transcript(run_result["session_id"], cwd)
        if tp:
            transcript_metrics = parse_transcript(tp)
            log(f"  Transcript: {transcript_metrics['turns']} turns, "
                f"{transcript_metrics['tool_calls']} tool calls "
                f"({transcript_metrics['failed_tool_calls']} failed)")
            if transcript_metrics.get("skill_triggered"):
                log(f"  Skill triggered: {transcript_metrics['skill_files_read']}")

    outcome = classify_outcome(ground_truth, run_result["result_text"], run_result["error"])
    log(f"  Outcome: {outcome}")

    cleanup_containers(gpu_host, port)

    return {
        "model_id": model_id,
        "index": index,
        "mode": mode,
        "outcome": outcome,
        "elapsed_s": run_result["elapsed_s"],
        "cost_usd": run_result["cost_usd"],
        "turns": transcript_metrics.get("turns"),
        "tool_calls": transcript_metrics.get("tool_calls"),
        "failed_tool_calls": transcript_metrics.get("failed_tool_calls"),
        "tool_breakdown": transcript_metrics.get("tool_breakdown"),
        "skill_triggered": transcript_metrics.get("skill_triggered"),
        "skill_files_read": transcript_metrics.get("skill_files_read"),
        "exit_code": run_result["exit_code"],
        "ground_truth": ground_truth,
        "response_preview": run_result["result_text"][:500],
        "timestamp": datetime.now().isoformat(),
    }


def print_summary(results):
    """Print a summary table of results."""
    for mode in ("with_skill", "without_skill"):
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue

        served = sum(1 for r in mode_results if r["outcome"] == "served")
        skipped = sum(1 for r in mode_results if r["outcome"] == "correct_skip")
        failed = sum(1 for r in mode_results
                     if r["outcome"] in ("failed", "container_stuck", "health_only"))
        timed_out = sum(1 for r in mode_results if r["outcome"] == "timeout")

        turns_list = [r["turns"] for r in mode_results if r.get("turns")]
        avg_turns = sum(turns_list) / max(len(turns_list), 1)

        tools_list = [r["tool_calls"] for r in mode_results if r.get("tool_calls")]
        avg_tools = sum(tools_list) / max(len(tools_list), 1)

        failed_calls = [r["failed_tool_calls"] for r in mode_results
                        if r.get("failed_tool_calls") is not None]
        avg_failed = sum(failed_calls) / max(len(failed_calls), 1)

        costs = [r["cost_usd"] for r in mode_results if r.get("cost_usd")]
        total_cost = sum(costs)
        total_time = sum(r["elapsed_s"] for r in mode_results)

        skill_triggered = sum(1 for r in mode_results if r.get("skill_triggered"))

        label = "WITH SKILL" if mode == "with_skill" else "WITHOUT SKILL"
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        print(f"  Total models:      {len(mode_results)}")
        print(f"  Served:            {served}")
        print(f"  Correct skip:      {skipped}")
        print(f"  Failed:            {failed}")
        print(f"  Timeout:           {timed_out}")
        print(f"  Avg turns:         {avg_turns:.1f}")
        print(f"  Avg tool calls:    {avg_tools:.1f}")
        print(f"  Avg failed calls:  {avg_failed:.1f}")
        print(f"  Skill triggered:   {skill_triggered}/{len(mode_results)}")
        print(f"  Total cost:        ${total_cost:.2f}")
        print(f"  Total time:        {total_time/3600:.1f} hours")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models-file",
                   default=os.path.expanduser("~/models_to_test.txt"),
                   help="File with one HF model ID per line")
    p.add_argument("--gpu-host", required=True,
                   help="GPU server SSH target (e.g. root@165.245.137.144)")
    p.add_argument("--skill-dir",
                   default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="Path to the skill repo (has SKILL.md)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--mode", choices=["with-skill", "without-skill", "both"],
                   default="both")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Per-model timeout in seconds (default 1800)")
    p.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    p.add_argument("--results-file",
                   default=os.path.expanduser("~/eval_results.json"))
    p.add_argument("--start-from", type=int, default=0,
                   help="Skip first N models (for resuming)")
    args = p.parse_args()

    # Read models
    models_path = os.path.expanduser(args.models_file)
    if not os.path.exists(models_path):
        print(f"Models file not found: {models_path}", file=sys.stderr)
        sys.exit(1)
    with open(models_path) as f:
        models = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Validate skill dir
    skill_md = os.path.join(args.skill_dir, "SKILL.md")
    if not os.path.exists(skill_md):
        print(f"SKILL.md not found in {args.skill_dir}", file=sys.stderr)
        sys.exit(1)

    # Create no-skill dir
    no_skill_dir = os.path.join(os.path.dirname(args.skill_dir), NO_SKILL_DIR_NAME)
    os.makedirs(no_skill_dir, exist_ok=True)

    # Verify SSH access
    log(f"Verifying SSH to {args.gpu_host}...")
    rc, _, err = ssh_run(args.gpu_host, "echo ok", timeout=10)
    if rc != 0:
        print(f"Cannot SSH to {args.gpu_host}: {err}", file=sys.stderr)
        sys.exit(1)
    log("SSH OK")

    # Verify claude CLI
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        print("claude CLI not found. Install it first.", file=sys.stderr)
        sys.exit(1)
    log("Claude CLI OK")

    # Load existing results
    results = []
    if os.path.exists(args.results_file):
        try:
            with open(args.results_file) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    results = data.get("results", [])
                elif isinstance(data, list):
                    results = data
        except (json.JSONDecodeError, IOError):
            pass

    # Build run plan
    modes = []
    if args.mode in ("with-skill", "both"):
        modes.append(True)
    if args.mode in ("without-skill", "both"):
        modes.append(False)

    total_runs = len(models[args.start_from:]) * len(modes)
    log(f"Eval plan: {len(models)} models, "
        f"modes={[('with' if m else 'without') for m in modes]}, "
        f"start_from={args.start_from}, total_runs={total_runs}")
    log(f"GPU host: {args.gpu_host}")
    log(f"Skill dir: {args.skill_dir}")
    log(f"No-skill dir: {no_skill_dir}")
    log(f"Claude model: {args.claude_model}")
    log(f"Max turns: {args.max_turns}, Timeout: {args.timeout}s")
    log(f"Results: {args.results_file}")
    print()

    run_count = 0
    for with_skill in modes:
        mode_label = "WITH SKILL" if with_skill else "WITHOUT SKILL"
        log(f"{'='*60}")
        log(f"Starting pass: {mode_label}")
        log(f"{'='*60}")

        for i, model_id in enumerate(models):
            if i < args.start_from:
                continue

            run_count += 1
            log(f"\nRun {run_count}/{total_runs}")

            result = run_single(
                model_id=model_id,
                index=i,
                with_skill=with_skill,
                gpu_host=args.gpu_host,
                port=args.port,
                skill_dir=args.skill_dir,
                no_skill_dir=no_skill_dir,
                max_turns=args.max_turns,
                claude_model=args.claude_model,
                timeout=args.timeout,
            )
            results.append(result)

            # Save after each model for crash resilience
            output = {
                "eval_started": results[0]["timestamp"] if results else datetime.now().isoformat(),
                "eval_updated": datetime.now().isoformat(),
                "gpu_host": args.gpu_host,
                "claude_model": args.claude_model,
                "max_turns": args.max_turns,
                "timeout_s": args.timeout,
                "results": results,
            }
            with open(args.results_file, "w") as f:
                json.dump(output, f, indent=2)

    print_summary(results)
    log(f"\nResults written to {args.results_file}")


if __name__ == "__main__":
    main()
