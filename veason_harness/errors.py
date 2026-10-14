"""Error contract: never a bare traceback, always a next step.

An LLM agent can only recover from a failure if the failure says what to do.
Every expected failure therefore leaves as a single JSON object on *stderr*
(stdout stays empty, so the agent never parses an error as data) plus a
non-zero exit code:

    {"ok": false, "error": "...", "hint": "...", "where": "veason regions"}

Exit codes: ``2`` usage, ``3`` runtime/environment, ``4`` verification failed.
"""
from __future__ import annotations

import json
import sys
from typing import NoReturn


class VrsError(Exception):
    """An expected, actionable failure."""

    def __init__(self, error: str, hint: str = "", code: int = 3) -> None:
        super().__init__(error)
        self.error = error
        self.hint = hint
        self.code = code


def emit_error(cmd: str, error: str, hint: str = "", code: int = 3) -> NoReturn:
    payload = {"ok": False, "where": cmd, "error": error}
    if hint:
        payload["hint"] = hint
    json.dump(payload, sys.stderr, ensure_ascii=False, indent=2)
    sys.stderr.write("\n")
    sys.stderr.flush()
    raise SystemExit(code)


def emit_error_from(cmd: str, exc: VrsError) -> NoReturn:
    emit_error(cmd, exc.error, exc.hint, exc.code)
