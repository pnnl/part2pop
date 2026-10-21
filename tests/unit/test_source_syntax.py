"""Every part2pop source file must compile without warnings.

Invalid escape sequences such as ``'$\\lambda$'`` in a non-raw string warn at
compile time: ``SyntaxWarning`` on Python 3.12 and later, but
``DeprecationWarning`` on Python 3.10 and 3.11. Turning *all* warnings into
errors makes the check work on every supported version. Compiling in-process
also avoids writing ``__pycache__`` files into the source tree.
"""

import warnings
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "part2pop"


def _compile_strictly(source: str, filename: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(source, filename, "exec")


def test_strict_compile_rejects_an_invalid_escape_sequence():
    # Guards the check itself on whichever Python version runs the suite.
    with pytest.raises(SyntaxError):
        _compile_strictly("label = '$\\lambda$'\n", "<invalid-escape>")


def test_all_sources_compile_without_warnings():
    sources = sorted(SOURCE_ROOT.rglob("*.py"))
    assert sources, f"no Python sources found under {SOURCE_ROOT}"

    failures = []
    for path in sources:
        try:
            _compile_strictly(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(SOURCE_ROOT)}:{exc.lineno}: {exc.msg}")

    assert not failures, "Sources with compile-time warnings:\n" + "\n".join(failures)
