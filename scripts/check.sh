#!/usr/bin/env bash
# Run unit tests and validate every SKILL.md in the catalog.
#
# Usage:
#   ./scripts/check.sh              Run tests + validate every skill.
#   ./scripts/check.sh -h|--help    Print this help.
#
# Requires `uv` (https://github.com/astral-sh/uv).
#
# Note: this repo does not publish anything yet. When we add catalog
# manifests (.cursor-plugin/plugin.json, .claude-plugin/marketplace.json,
# .mcp.json, etc.) we'll add a separate scripts/publish.sh for generation
# and check.sh will gain a `--check` mode that diffs the regenerated
# artifacts against the committed copy.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed -n '/^Usage:/,/^Requires/p'
}

case "${1:-}" in
  "")
    uv run scripts/test_validate_skills.py
    uv run scripts/validate_skills.py
    echo "All checks passed."
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac
