#!/usr/bin/env bash
# Tail the live vLLM log inside the container (Ctrl-C to stop).
# Run this ON the server (it does not SSH anywhere).
exec docker exec <CONTAINER> tail -F /tmp/vllm.log
