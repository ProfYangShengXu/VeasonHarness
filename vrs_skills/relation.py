"""RelateAnything: regions (+ optional masks) in, scored (subject, relation, object) out.

The relation vocabulary is supplied *at inference time* -- that is the native
usage of the model, not a convenience wrapper.  ``set_vocabulary`` re-encodes
the phrases with the shipped text student and re-parameterises the head, so
``vrs relate <dir> <fid> "riding" "carrying"`` really does ask a different
question rather than filtering a fixed table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import paths
from .errors import VrsError

#: RelateAnything drops boxes beyond this many and returns [] for N < 2.
#: We apply the cap ourselves so the region idx -> triplet mapping stays exact.
MAX_REGIONS = 60


def _load_relate_anything(model_dir: Path, predicates: Optional[Sequence[str]]):
    try:
        from relsgg import RelateAnything
    except ImportError as exc:  # pragma: no cover - environment issue
        raise VrsError(
            f"the `relsgg` package is not importable ({exc})",
            hint="bash tools/setup_models.sh installs it with --no-deps so it reuses "
            "the existing torch/transformers; plain `pip install -e .` would try to "
            "downgrade numpy/Pillow/transformers.",
        ) from exc

    ckpt = model_dir / "model.pth"
    if not ckpt.is_file():
        raise VrsError(
            f"RelateAnything checkpoint not found: {ckpt}",
            hint="download maelic/relsgg-vits16plus from hf-mirror.com into "
            f"{model_dir} (see README 'Models'); huggingface.co is unreachable here.",
        )
    return RelateAnything.from_checkpoint(str(ckpt), predicates=list(predicates) if predicates else None,
                                          device="cpu")


class Relater:
    """Thin, cached wrapper around one loaded RelateAnything model."""

    _cache = None  # process-level: `index` reuses it across frames

    def __init__(self, model_dir: Optional[Path] = None, predicates: Optional[Sequence[str]] = None) -> None:
        self.model_dir = Path(model_dir or paths.RELSGG_DIR)
        key = (str(self.model_dir), tuple(predicates) if predicates else None)
        if Relater._cache is not None and Relater._cache[0] == key:
            self.model = Relater._cache[1]
        else:
            self.model = _load_relate_anything(self.model_dir, predicates)
            Relater._cache = (key, self.model)
        self.predicates = list(self.model.predicates)

    def set_vocabulary(self, predicates: Sequence[str]) -> None:
        """Swap the vocabulary on an already-loaded model (relsgg-level API).

        The commands do not call this: `Relater(predicates=...)` is the path
        they use, and `from_checkpoint` already runs the same encoder once
        during load.  A different vocabulary is therefore a *new model load*
        (~14 s here), not a free re-parameterisation -- that is why relation
        graphs are cached per vocabulary on disk rather than recomputed.
        """
        phrases = [p.strip() for p in predicates if p and p.strip()]
        if not phrases:
            raise VrsError(
                "no relation phrases given",
                hint='usage: vrs relate <index_dir> <frame_id> "riding" "holding"; '
                "any free-text relation works, they are encoded at inference time.",
                code=2,
            )
        self.model.set_vocabulary(phrases)
        self.predicates = list(self.model.predicates)

    def relate(self, frame: np.ndarray, regions: list[dict], masks: dict[int, np.ndarray],
               topk: int = 20, min_score: float = 0.0) -> tuple[list[dict], dict]:
        """Return ``(edges, info)``; ``edges`` speak in region idx, never in list positions."""
        n = len(regions)
        if n < 2:
            return [], {"n_regions": n, "n_used": n, "truncated": False,
                        "reason": "RelateAnything needs at least 2 regions to form a pair"}
        used = list(range(n))
        truncated = False
        if n > MAX_REGIONS:
            order = sorted(range(n), key=lambda i: -float(regions[i].get("score", 0.0)))
            used = sorted(order[:MAX_REGIONS])
            truncated = True

        boxes = np.asarray([regions[i]["bbox"] for i in used], dtype=np.float32)
        labels = [regions[i]["label"] for i in used]
        scores = np.asarray([float(regions[i].get("score", 1.0)) for i in used], dtype=np.float32)
        mask_list = [masks.get(i) for i in used]
        masks_arg = mask_list if all(m is not None for m in mask_list) else None

        trips = self.model.predict(frame, boxes, masks=masks_arg, box_labels=labels,
                                   box_scores=scores, topk=topk, max_boxes=MAX_REGIONS)
        edges = []
        for t in trips:
            if float(t.score) < min_score:
                continue
            edges.append(
                {
                    "subject": int(used[int(t.subject_idx)]),
                    "relation": str(t.predicate),
                    "object": int(used[int(t.object_idx)]),
                    "score": round(float(t.score), 4),
                }
            )
        info = {
            "n_regions": n,
            "n_used": len(used),
            "truncated": truncated,
            "masks_used": masks_arg is not None,
            "vocabulary": list(self.predicates),
        }
        return edges, info
