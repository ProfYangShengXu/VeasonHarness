"""VRS perception skills, packaged so an LLM agent can drive them from a shell.

Design in one line: **the agent picks ``index``es, never coordinates.**
``vrs answer`` is the only command that turns a region id into a box, so an
answer that refers to a region the caller never saw is structurally impossible.

Layering (dependencies point one way, always at the stable side)::

    cli.py        argparse + JSON out    <- changes when the interface changes
      |
      +-- indexer.py, detector.py, relation.py, render.py, answers.py
      |                                          <- model/runtime specifics
      +-- store.py, video.py, paths.py, errors.py
                                                 <- the stable core: index format,
                                                    region identity, path rules

``store.py`` owns the two invariants the rest relies on: region idx is
permanent (append-only table) and a bbox is written to disk exactly once.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
