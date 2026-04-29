"""Shared test setup.

Make ``humanize/dpo_v0/`` importable as a top-level package root and the
repo root (``videodpoWan/``) importable for ``humanize.*`` style imports
that some modules (e.g. ``eval.heldout_regen``) reach for.
"""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # humanize/dpo_v0/
_REPO_ROOT = _PKG_ROOT.parent.parent  # videodpoWan/

for _p in (_PKG_ROOT, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
