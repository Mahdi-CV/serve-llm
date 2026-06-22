#!/usr/bin/env bash
# Open the live server view (run in a SECOND terminal before the Claude run).
# Top pane: vLLM warmup/run log. Bottom pane: container + process + /health.
# Detach with Ctrl-b d. Re-run to reattach.
exec ssh -t do-mi350-node 'bash /root/demo_view.sh'
