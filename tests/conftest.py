from __future__ import annotations

import pathlib
import sys


def _ensure_src_on_path():
    """
    Prefer the in-repo source tree when running tests.

    Developers often have an older `part2pop` installed in their environment.
    Exercising the code under ./src keeps unit tests sandboxed inside the repo
    without requiring an editable install.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()
