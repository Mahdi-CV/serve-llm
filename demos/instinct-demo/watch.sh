#!/usr/bin/env bash
# Open the live server view (run in a SECOND terminal before the Claude run).
# Top pane: vLLM warmup/run log. Bottom pane: container + process + /health.
# Detach with Ctrl-b d. Re-run this to reattach.
exec ssh -t root@134.199.193.61 'bash /root/demo_view.sh'
