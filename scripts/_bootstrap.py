"""Shared script bootstrap: put the repo root on sys.path and fail gracefully on unfilled stubs.

INFRASTRUCTURE. Each ``run_weekN_*.py`` does ``import _bootstrap`` first, then wraps its
``main()`` in :func:`run_guarded` so an unimplemented algorithm prints a helpful message instead
of a raw traceback.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUTS = os.path.join(ROOT, "outputs")


def run_guarded(main):
    """Run ``main`` and turn a NotImplementedError from an unfilled stub into guidance."""
    try:
        return main()
    except NotImplementedError as e:
        print("\n" + "=" * 70)
        print("  This milestone needs an algorithm you haven't implemented yet:")
        print(f"    -> {e}")
        print("  Implement the TODO(owner) in that file, then re-run this script.")
        print("=" * 70 + "\n")
        sys.exit(0)
