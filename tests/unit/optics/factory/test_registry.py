# tests/unit/optics/factory/test_registry.py

import os
from types import SimpleNamespace

import pytest

from part2pop.optics.factory import registry as reg
from part2pop.optics.factory.registry import discover_morphology_types


def test_discover_morphology_types_returns_callables():
    morphs = discover_morphology_types()
    assert isinstance(morphs, dict)
    assert len(morphs) > 0

    for name, builder in morphs.items():
        assert callable(builder)


def test_discover_morphology_includes_known_morphologies():
    morphs = discover_morphology_types()
    assert "homogeneous" in morphs
    assert "core_shell" in morphs


def test_safe_import_module_file_fallback(tmp_path, monkeypatch):
    temp_dir = tmp_path / "optics_fake"
    temp_dir.mkdir()
    temp_path = temp_dir / "fake_morphology.py"
    temp_path.write_text("VALUE = 42")

    module = reg._safe_import_module("nonexistent_module", file_path=str(temp_path))
    assert hasattr(module, "VALUE")


def test_discover_skips_private_and_broken_modules(monkeypatch):
    monkeypatch.setattr(
        reg,
        "pkgutil",
        SimpleNamespace(
            iter_modules=lambda paths: [
                (None, "_private", False),
                (None, "registry", False),
                (None, "broken", False),
                (None, "good", False),
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(reg, "_morphology_registry", {}, raising=False)

    def fake_safe_import(fullname, file_path=None):
        if fullname.endswith("broken"):
            raise ImportError("missing optional dep")
        if fullname.endswith("good"):
            return SimpleNamespace(build=lambda base, cfg: (base, cfg))
        return SimpleNamespace()

    monkeypatch.setattr(reg, "_safe_import_module", fake_safe_import, raising=False)

    with pytest.warns(RuntimeWarning):
        discovered = reg.discover_morphology_types()

    assert "good" in discovered
    assert "broken" not in discovered
    assert "_private" not in discovered


def test_list_morphology_types_sorted(monkeypatch):
    monkeypatch.setattr(
        reg,
        "discover_morphology_types",
        lambda: {"z": object(), "a": object()},
        raising=False,
    )
    assert reg.list_morphology_types() == ["a", "z"]
