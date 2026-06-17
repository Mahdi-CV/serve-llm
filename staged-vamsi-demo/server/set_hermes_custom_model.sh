#!/usr/bin/env bash
# Demo: reset the hermes client to a clean state, then point it at our vLLM endpoint.
# Run on the box:  ./set_hermes_custom_model.sh
set -u

PROVIDER="custom"
BASE_URL="http://<SERVER_IP>:<PORT>/v1"
API_KEY="none"
MODEL="<MODEL>"

line() { printf '%s\n' "------------------------------------------------------------"; }

echo
line
echo " Resetting Hermes for a fresh run"
line

# 1) Clear any sessions from previous runs.
echo "[1/3] Clearing previous Hermes sessions ..."
hermes sessions prune --older-than 0 --yes >/dev/null 2>&1 || true
echo "      done."

# 2) Clear previous config so we start clean (backed up first).
CFG="$(hermes config path 2>/dev/null)"
echo "[2/3] Clearing previous config: ${CFG:-<none>}"
if [ -n "${CFG:-}" ] && [ -f "$CFG" ]; then
  cp -f "$CFG" "${CFG}.bak.$(date +%s)" 2>/dev/null || true
  rm -f "$CFG"
fi
echo "      done."

# 3) Set the custom model config (echo each value as it is applied).
echo "[3/3] Setting custom model configuration:"
echo
echo "  hermes config set model.provider  $PROVIDER"
hermes config set model.provider "$PROVIDER" >/dev/null
echo "  hermes config set model.base_url  $BASE_URL"
hermes config set model.base_url "$BASE_URL" >/dev/null
echo "  hermes config set model.api_key   $API_KEY"
hermes config set model.api_key "$API_KEY" >/dev/null
echo "  hermes config set model.default   $MODEL"
hermes config set model.default "$MODEL" >/dev/null

echo
line
echo " Hermes now configured:"
line
printf "   %-12s %s\n" "provider:" "$PROVIDER"
printf "   %-12s %s\n" "base_url:" "$BASE_URL"
printf "   %-12s %s\n" "api_key:"  "$API_KEY"
printf "   %-12s %s\n" "model:"    "$MODEL"
line
echo
