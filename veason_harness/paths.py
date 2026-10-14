"""Path handling for a CLI that is called from WSL but runs on Windows Python.

The caller (an LLM agent) lives in a WSL shell and hands us paths like
``/mnt/c/Users/...``.  The process itself is a *Windows* interpreter, so
``cv2``/``PIL`` only understand ``C:\\Users\\...``.  This module is the single
place where the two worlds are translated, in both directions:

* :func:`resolve`      -- anything the agent typed  -> a real Windows ``Path``
* :func:`for_agent`    -- a Windows ``Path``        -> a path the agent can open

Nothing else in the package is allowed to guess at path flavours.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: Repository root (``.../veason-skills``).  Model weights and the default index
#: directory hang off this, so behaviour does not depend on the shell's cwd.
PROJECT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_YOLOE_PF = MODELS_DIR / "yoloe-11s-seg-pf.pt"
DEFAULT_YOLOE = MODELS_DIR / "yoloe-11s-seg.pt"
RELSGG_DIR = MODELS_DIR / "relsgg-vits16plus"
#: Where ``attempt_download_asset`` looks for ``mobileclip_blt.ts``.
ULTRALYTICS_WEIGHTS_DIR = MODELS_DIR / "ultralytics_weights"
DEFAULT_INDEX_DIR = PROJECT_DIR / "index"

#: Round-3 sense models (all optional: each axis degrades if its file is absent).
MEDIAPIPE_DIR = MODELS_DIR / "mediapipe"
FACE_LANDMARKER_TASK = MEDIAPIPE_DIR / "face_landmarker.task"
VIDEOMAE_DIR = MODELS_DIR / "videomae-small-kinetics"
DEPTH_DIR = MODELS_DIR / "depth-anything-v2-small"
DEPTH_ONNX = DEPTH_DIR / "onnx" / "model.onnx"
DEPTH_PREPROCESSOR = DEPTH_DIR / "preprocessor_config.json"
POSE_WEIGHTS = ULTRALYTICS_WEIGHTS_DIR / "yolo11n-pose.pt"

_MNT = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")


def to_windows(p: str | os.PathLike) -> str:
    """``/mnt/c/foo`` -> ``C:\\foo``; anything else is returned unchanged."""
    s = str(p)
    m = _MNT.match(s)
    if m:
        rest = (m.group(2) or "").replace("/", "\\")
        return f"{m.group(1).upper()}:\\{rest}"
    return s


def to_wsl(p: str | os.PathLike) -> str | None:
    """``C:\\foo`` -> ``/mnt/c/foo``; ``None`` when the drive is not /mnt-able."""
    s = str(p)
    m = re.match(r"^([a-zA-Z]):[\\/](.*)$", s)
    if not m:
        return None
    return "/mnt/" + m.group(1).lower() + "/" + m.group(2).replace("\\", "/")


def invocation_cwd() -> Path:
    """Base for relative paths.

    The shell shim exports ``VEASON_CWD`` (the *shell's* cwd, already converted to
    a Windows path) because a Windows process launched from WSL inherits a
    ``\\\\wsl.localhost\\...`` UNC cwd that is useless to ``cv2``.
    """
    env = os.environ.get("VEASON_CWD")
    if env:
        return Path(to_windows(env))
    return Path(os.getcwd())


def resolve(p: str | os.PathLike, *, base: Path | None = None) -> Path:
    """Normalise *any* user-supplied path into an absolute Windows ``Path``."""
    s = to_windows(str(p))
    q = Path(s)
    if not q.is_absolute():
        q = (base or invocation_cwd()) / q
    return q.expanduser()


def for_agent(p: str | os.PathLike) -> str:
    """The path string we hand back to the agent.

    WSL-style (``/mnt/c/...``) whenever the file sits on a mounted drive,
    because that is the form the calling shell can actually read.
    """
    ap = Path(os.path.abspath(to_windows(str(p))))
    w = to_wsl(ap)
    return w or str(ap)


def for_windows(p: str | os.PathLike) -> str:
    return os.path.abspath(to_windows(str(p)))
