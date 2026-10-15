#!/usr/bin/env bash
# WSL shim: run the Windows Python that owns the ML environment.
#
# VRS_CWD carries the *shell's* working directory across the WSL/Windows
# boundary so that relative paths in `--out` behave the way the caller expects.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN_PY="${VRS_PYTHON:-/mnt/c/Users/45140/AppData/Local/Programs/Python/Python312/python.exe}"

export VRS_CWD="$(wslpath -w "$PWD" 2>/dev/null || pwd)"

exec "$WIN_PY" "$(wslpath -w "$DIR/vrs.py")" "$@"
