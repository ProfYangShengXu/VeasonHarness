"""Video probing, hashing and frame sampling.

Frame ids are **decoded video frame indices**, not ordinals inside the
sampled selection.  That choice is deliberate: it makes ``frame_id`` stable
when ``--frames`` changes on a re-index, it makes ``t = frame_id / fps``
exact, and it lets the on-disk cache be keyed on something that only depends
on the video.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .errors import VrsError
from .store import sha256_file


def probe(video: Path) -> dict:
    """Cheap container-level facts.  Raises a guided error when undecodable."""
    if not video.is_file():
        raise VrsError(
            f"video not found: {video}",
            hint="pass an existing .mp4/.avi path; WSL paths like "
            "/mnt/c/Users/... are accepted and translated automatically.",
        )
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise VrsError(
                f"cv2 could not open {video.name}",
                hint="check the file is a real video (ffprobe it); "
                "if it is, the OpenCV build may lack that codec.",
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    finally:
        cap.release()
    if n <= 0 or w <= 0 or h <= 0:
        raise VrsError(
            f"{video.name}: container reports {n} frames at {w}x{h}",
            hint="the video does not decode far enough to index; try another file "
            "or re-encode it with `ffmpeg -i in.mp4 -c:v libx264 out.mp4`.",
        )
    if fps <= 0:
        raise VrsError(
            f"{video.name}: reported fps is {fps}",
            hint="no timestamps can be derived without fps; re-encode with a fixed "
            "frame rate (`ffmpeg -i in.mp4 -r 25 out.mp4`).",
        )
    return {
        "fps": fps,
        "n_video_frames": n,
        "width": w,
        "height": h,
        "duration_s": round(n / fps, 3),
    }


def sample_frame_ids(n_frames: int, n_video_frames: int) -> list[int]:
    """``n_frames`` ids spread evenly across the video, endpoints inclusive."""
    if n_frames <= 0:
        raise VrsError(
            f"--frames must be >= 1 (got {n_frames})",
            hint="16 is the conservative default; raise it for better keyframe recall.",
        )
    n_frames = min(n_frames, n_video_frames)
    if n_frames == 1:
        return [0]
    step = (n_video_frames - 1) / (n_frames - 1)
    return sorted({int(round(i * step)) for i in range(n_frames)})


def read_frames(video: Path, frame_ids: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested ids, seeking rather than streaming when cheap."""
    cap = cv2.VideoCapture(str(video))
    out: dict[int, np.ndarray] = {}
    try:
        if not cap.isOpened():
            raise VrsError(
                f"cv2 could not open {video}",
                hint="see `veason doctor`-style hint: re-encode the file or use another container.",
            )
        for fid in sorted(frame_ids):
            if fid < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                # Decoders occasionally drop a frame after a seek; step back once.
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid - 1))
                ok, frame = cap.read()
            if ok:
                out[fid] = frame
    finally:
        cap.release()
    if not out:
        raise VrsError(
            f"no frames could be decoded from {video.name}",
            hint="try fewer frames (`--frames 4`): some decoders fail on long seeks.",
        )
    return out


def video_fingerprint(video: Path) -> str:
    return sha256_file(video)
