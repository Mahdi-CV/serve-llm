# Staged demo: serving LLMs on AMD Instinct with Claude Code

This folder documents exactly how we stage the "serve any HuggingFace model on AMD
Instinct in one natural prompt" demo, so you can reproduce it for a different model
(e.g. `gpt-oss`) in your own demo folder.

It is a **staging guide**. The demo shows a real capability (Claude Code + a serving
skill brings a model up on a remote MI300X over SSH and returns a working OpenAI
endpoint). We pre-stage the box so the audience does not sit through a cold first-time
compile. Nothing is faked: the same prompt on a cold box produces the same result, it
just takes longer. Be transparent with your audience about what is pre-staged (see
"What is honest to say" below).

---

## 1. The demo story

1. On the laptop you open Claude Code in a demo folder and type one natural prompt:
   > run <MODEL> on my Instinct server at <SERVER_IP> with vLLM and give me the endpoint
2. Claude picks up the serving skill, SSHes to the GPU box, brings the model up in a
   vLLM container, polls `/health`, sends a test chat request, and prints a clean
   connection table (model, base URL, port, API key, TP, GPU).
3. You then point a client (we use the `hermes` agent) at that endpoint and chat.

Target: endpoint live in roughly two minutes, clean output, no back-and-forth.

---

## 2. Architecture (the important part)

We do **not** create a new container named after each model. Instead:

- **One general-purpose vLLM container** runs as standing infrastructure on the box,
  named after the image (e.g. `vllm-openai-rocm-ctr`). It runs an idle entrypoint
  (`sleep infinity`) and stays up.
- **Serving a model = launching `vllm serve` inside that container** via `docker exec -d`.
- **The vLLM compile cache is a host bind-mount** (`/mnt/scratch/vllm_cache` ->
  `/root/.cache/vllm`). It lives on the host disk, so compiled graphs survive process
  kills, container restarts, and even container recreation. The HuggingFace weights
  cache is bind-mounted the same way, so there is no download at demo time.
- **Reset between takes = kill the model process** (`pkill -f "vllm serve"`). The
  container stays up. The next run relaunches the model from warm disk cache and shows
  the engine warming up live.
- The container is created with **`--init`** (tini as PID 1) so killed model children
  are reaped and you do not accumulate `<defunct>` zombie processes.

### Why this design
- Generic container name looks like real infrastructure, not a per-demo prop.
- Disk-persisted cache is what keeps warm relaunch fast (~90s) vs. a cold compile
  (several minutes).
- Killing only the process (not the container) is the fastest clean reset between takes.

### Timing you can expect (MI300X, single GPU)
| Phase | Time |
|-------|------|
| Cold first-ever compile (no cache) | several minutes |
| First launch after a cache-mount change (one-time populate) | ~240s |
| Warm relaunch after a process kill (the demo path) | ~90s + orchestration |
| Hitting an already-healthy endpoint | instant |

---

## 3. Two-machine layout

- **Laptop** runs Claude Code (`claude --dangerously-skip-permissions`) and drives the
  box over SSH. The demo folder lives here and contains a `CLAUDE.md` that steers the
  agent's behavior.
- **Server** (the MI300X box) runs Docker + the vLLM container, plus a few helper
  scripts in the home folder.

`--dangerously-skip-permissions` cannot run as root, so run Claude as a normal user on
the laptop.

---

## 4. What is in a demo folder

Laptop demo folder (e.g. `~/instinct-demo/`):

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Steers the agent: target server, serving defaults, operational policy, output style. This is the file that makes the run clean and consistent. Auto-loaded by Claude Code from the working directory. |
| `stage.sh` | One command you run before each take; SSHes to the box and runs the server staging script. |
| `.demo-private.md` | Your personal cheat sheet (prompt to say, flow, timing, what is honest to say). Not shown on camera. |
| `watch.sh` (optional) | Opens a live split-screen "server view" (log + status) in a second terminal. |

Server home folder (e.g. `/root/`):

| File | Purpose |
|------|---------|
| `stage_demo.sh` | Ensures the container is up (creates it with `--init` + GPU/cache mounts if missing), frees the port, kills any running model process. |
| `check_docker_log.sh` | Tails the live vLLM log (`docker exec ... tail -F /tmp/vllm.log`). |
| `set_hermes_custom_model.sh` | Resets the `hermes` client config + sessions and points it at the vLLM endpoint, printing the values. |
| `demo_view.sh` (optional) | tmux two-pane server view (log + container/process/health), used by `watch.sh`. |

Templates for all of these are in [`templates/`](templates/) and [`server/`](server/).

---

## 5. How to build a NEW demo for your model

Replace the placeholders below throughout the template files:

| Placeholder | Meaning | Example (Qwen demo) |
|-------------|---------|---------------------|
| `<SERVER_IP>` | GPU box IP | `134.199.193.61` |
| `<SSH_USER>` | SSH user on the box | `root` |
| `<PORT>` | serving port | `8002` |
| `<CONTAINER>` | generic container name (after the image) | `vllm-openai-rocm-ctr` |
| `<IMAGE>` | vLLM ROCm image:tag | `vllm/vllm-openai-rocm:v0.22.1` |
| `<MODEL>` | full HF model id you serve | `Qwen/Qwen3.5-27B-FP8` |
| `<MODEL_SPOKEN>` | what you say out loud | `Qwen3.5 27B` |
| `<HF_CACHE>` | host HF cache dir | `/home/evaluser/.cache/huggingface` |
| `<VLLM_CACHE>` | host vLLM compile cache dir | `/mnt/scratch/vllm_cache` |
| `<VLLM_MODEL_ARGS>` | model-specific vLLM flags | `--tool-call-parser qwen3_coder --reasoning-parser qwen3` |
| `<VLLM_ENV>` | model-specific env vars | `VLLM_ROCM_USE_AITER=1 ...` |

Steps:

1. **Copy the templates into a new laptop folder**, e.g. `~/gpt-oss-demo/`:
   ```bash
   mkdir -p ~/gpt-oss-demo
   cp templates/CLAUDE.md          ~/gpt-oss-demo/CLAUDE.md
   cp templates/stage.sh           ~/gpt-oss-demo/stage.sh
   cp templates/demo-private.md.example ~/gpt-oss-demo/.demo-private.md
   chmod +x ~/gpt-oss-demo/stage.sh
   ```
2. **Fill in the placeholders** in `CLAUDE.md`, `stage.sh`, `.demo-private.md`.
3. **Copy the server scripts to the box** and fill in their placeholders:
   ```bash
   scp server/*.sh <SSH_USER>@<SERVER_IP>:/root/
   ssh <SSH_USER>@<SERVER_IP> 'chmod +x /root/*.sh'
   ```
4. **Pre-warm once** so the disk cache is populated (do this well before the demo):
   ```bash
   cd ~/gpt-oss-demo && ./stage.sh      # creates the container if missing
   claude --dangerously-skip-permissions
   # say the prompt once; let it fully come up (first time is the slow one)
   ```
5. **Verify the warm path** by running `./stage.sh` again (kills the process) and
   re-running the prompt. This second run is the demo timing.

See [`GPT-OSS.md`](GPT-OSS.md) for the model-specific values for gpt-oss.

---

## 6. Per-take flow (rehearse this)

1. Laptop: `cd ~/gpt-oss-demo && ./stage.sh`  (ensures container up, kills old process)
2. Optional second terminal: `./watch.sh`  (live server view) or SSH in and run
   `./check_docker_log.sh`.
3. Laptop: `claude --dangerously-skip-permissions`
4. Say the prompt:
   > run <MODEL_SPOKEN> on my Instinct server at <SERVER_IP> with vLLM and give me the endpoint
5. When the table prints, point the client at it:
   ```bash
   ssh <SSH_USER>@<SERVER_IP>
   ./set_hermes_custom_model.sh
   hermes chat
   ```

---

## 7. What is honest to say if asked

- The weights are cached on the box (no download at demo time).
- We keep a general-purpose vLLM container running as infra; serving a model launches
  it inside that container. The ~2 minutes is the engine warming up (loading weights
  into HBM, replaying compiled graphs, allocating the KV cache).
- A brand-new model on a cold box takes longer; we pre-stage to stay in the time
  budget. Same prompt, same result, just faster because it is warm.

---

## 8. Troubleshooting / lessons learned

| Symptom | Cause | Fix |
|---------|-------|-----|
| `<defunct>` zombie processes after kills | idle PID 1 (`sleep`) does not reap children | recreate the container with `--init` |
| Agent asks "BF16 or FP8?" | model default precision not pinned | pin the exact variant in `CLAUDE.md` ("serve `<MODEL>`, do not ask precision") |
| `SSL CERTIFICATE_VERIFY_FAILED` during VRAM estimate | box is network-restricted | tell `CLAUDE.md` to skip internet steps / VRAM estimation and ignore such errors |
| Agent shows "Ready to launch?" plan | confirmation step enabled | tell `CLAUDE.md` to proceed without confirmation |
| Agent narrates "reusing container per policy" | verbose narration | add the "Output style (keep it clean)" section to `CLAUDE.md` |
| `docker run` exits 125 | old per-model container still holds the port | stop the old container before creating/serving |
| Claude Code "hook error" outside tmux | tmux hooks return non-zero when not in tmux | wrap each hook command so it exits 0, e.g. `{ ...; }; true` |

---

## 9. How we built this (chronological, what actually happened)

This is the path we took, including the dead ends, so you understand *why* the final
design looks the way it does.

1. **First approach: one container per model.** We started by creating a container
   named after the model (e.g. `vllm-qwen3.5-27b-fp8`) and using `docker start` /
   `docker stop` to bring the model up and down between takes. It worked, but during
   the run the agent printed things like "Container `vllm-qwen3.5-27b-fp8` exists and
   is stopped", which looked scripted and prop-like, and a per-model container name
   does not look like real infrastructure.

2. **Decision: reuse ONE general-purpose container.** We switched to a single
   standing container named after the image (`vllm-openai-rocm-ctr`). It is created
   once, runs an idle entrypoint (`--entrypoint sleep ... infinity`), and stays up.
   We never recreate it per model.

3. **Serving became a process, not a container.** Instead of starting a container,
   the agent now launches `vllm serve <MODEL> ... > /tmp/vllm.log 2>&1` *inside* the
   standing container with `docker exec -d`, then polls `/health`.

4. **We made the caches host bind-mounts so warm state persists.** We mounted the HF
   weights cache and, critically, the vLLM compile cache from host disk
   (`<VLLM_CACHE>` -> `/root/.cache/vllm`). We explicitly confirmed the bind-mount
   lives on the host disk, not inside the container, so the compiled graphs survive
   not only a process kill but full container recreation. This is what makes warm
   relaunch fast.

5. **Reset after each run = kill the process, keep the container.** Between takes we
   run `pkill -f "vllm serve"` inside the container. The container stays up; the next
   run relaunches the model from the warm disk cache and shows the engine warming up.
   This is the fastest clean reset and it is what `stage_demo.sh` does.

6. **Found and fixed zombie processes.** With an idle `sleep` as PID 1, killed model
   children were not reaped and we accumulated `[vllm] <defunct>` zombies across
   takes. Fix: recreate the standing container with `--init` (tini as PID 1). After
   that, a kill leaves no defunct processes (we verified the process table was clean).

7. **Tuned `CLAUDE.md` until the run was clean and consistent.** Each rough edge in a
   real run was fixed by an instruction in `CLAUDE.md`:
   - agent asked "BF16 or FP8?" -> pin the exact variant, "do not ask precision".
   - VRAM estimate hit `SSL CERTIFICATE_VERIFY_FAILED` (box is network restricted) ->
     "skip internet/VRAM steps, ignore such errors".
   - agent showed a "Ready to launch?" plan -> "proceed without confirmation".
   - agent narrated "reusing container per policy" (looked scripted) -> added the
     "Output style (keep it clean)" section so it stays quiet and just reports the
     endpoint.

8. **How we tested it.**
   - First launch after the cache mount changed took ~243s (one-time, populating the
     disk cache).
   - We then killed the process and relaunched: ~90s warm. That is the demo path.
   - We ran the full take multiple times back to back to confirm reproducibility: no
     FP8 question, no zombies, no SSL error, no confirmation prompt, same clean table
     every time.
   - Ground truth was checked independently over SSH (container up, `/health` 200,
     a real chat completion), not by trusting the agent's text.

9. **How to do this for any model.** The design is model-agnostic. To target a new
   model you only change: the model id, the model-specific vLLM flags
   (`<VLLM_MODEL_ARGS>`), any model-specific env, and the spoken model name in
   `CLAUDE.md`. Then pre-warm once (step 4 in section 5) so the disk cache is
   populated, and the warm path is ready. Everything else (standing container,
   exec-launch, kill-to-reset, output style) stays the same. See `GPT-OSS.md`.

## 10. Files

- [`templates/CLAUDE.md`](templates/CLAUDE.md) - agent steering file (the key one)
- [`templates/stage.sh`](templates/stage.sh) - laptop staging launcher
- [`templates/demo-private.md.example`](templates/demo-private.md.example) - cheat sheet
- [`server/stage_demo.sh`](server/stage_demo.sh) - server staging (ensure container, kill model)
- [`server/check_docker_log.sh`](server/check_docker_log.sh) - tail vLLM log
- [`server/set_hermes_custom_model.sh`](server/set_hermes_custom_model.sh) - reset + point hermes
- [`server/demo_view.sh`](server/demo_view.sh) - optional tmux server view
- [`GPT-OSS.md`](GPT-OSS.md) - gpt-oss specific values and notes
