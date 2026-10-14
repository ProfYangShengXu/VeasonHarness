#!/usr/bin/env python
"""``veason`` -- VRS perception skills driven from the terminal.

Usage::

    python veason.py index   <video> [--frames 16] [--out DIR]
    python veason.py regions <index_dir> <frame_id>
    ...

The Windows and WSL shims next to this file (``veason.cmd``, ``veason.sh``) call this
script with the same arguments, so the agent can use whichever shell it has.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from veason_harness.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
