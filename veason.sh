#!/usr/bin/env bash
# WSL shim: run the Windows Python that owns the ML environment.
#
# VEASON_CWD carries the *shell's* working directory across the WSL/Windows
# boundary so that relative paths in `--out` behave the way the caller expects.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN_PY="${VEASON_PYTHON:-<你的 python>}"

export VEASON_CWD="$(wslpath -w "$PWD" 2>/dev/null || pwd)"

exec "$WIN_PY" "$(wslpath -w "$DIR/veason.py")" "$@"
