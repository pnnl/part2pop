# tests/unit/optics/factory/test_registry.py

import os

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
