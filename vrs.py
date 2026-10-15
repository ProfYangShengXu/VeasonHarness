#!/usr/bin/env python
"""``vrs`` -- VRS perception skills driven from the terminal.

Usage::

    python vrs.py index   <video> [--frames 16] [--out DIR]
    python vrs.py regions <index_dir> <frame_id>
    ...

The Windows and WSL shims next to this file (``vrs.cmd``, ``vrs.sh``) call this
script with the same arguments, so the agent can use whichever shell it has.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vrs_skills.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
