#!/usr/bin/env bash
# Live "server view" for the demo. Two panes:
#   top    = vLLM warmup/run log (the GPU actually working)
#   bottom = container status + vllm serve process + /health
# Detach with Ctrl-b d. Re-run this script to reattach.
# Run this ON the server (needs tmux + watch).
set -u
CTR=<CONTAINER>
PORT=<PORT>

# Reattach if a view session is already running.
if tmux has-session -t demoview 2>/dev/null; then
  exec tmux attach -t demoview
fi

tmux new-session -d -s demoview -n view \
  "docker exec $CTR bash -lc 'touch /tmp/vllm.log; tail -F -n +1 /tmp/vllm.log'"

tmux split-window -v -t demoview \
  "watch -n1 -t 'echo CONTAINER:; docker ps --format \"  {{.Names}}   {{.Status}}\" | grep $CTR; echo; echo VLLM PROCESS:; docker exec $CTR ps -eo etime,pid,cmd 2>/dev/null | grep \"vllm serve\" | grep -v grep || echo \"  (none - container idle)\"; echo; printf HEALTH:\" \"; curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && echo UP || echo waiting'"

tmux select-layout -t demoview even-vertical
exec tmux attach -t demoview
