import importlib
from types import SimpleNamespace

import pytest

from part2pop.viz.factory import registry as viz_registry


def test_import_viz_factory_registry():
    importlib.import_module("part2pop.viz.factory.registry")


def test_discover_plotter_types_skips_private_and_broken_modules(monkeypatch):
    monkeypatch.setattr(
        viz_registry,
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

    def fake_import(fullname):
        if fullname.endswith("broken"):
            raise ImportError("optional dependency missing")
        if fullname.endswith("good"):
            return SimpleNamespace(build=lambda cfg: cfg)
        return SimpleNamespace()

    monkeypatch.setattr(viz_registry.importlib, "import_module", fake_import)

    with pytest.warns(RuntimeWarning):
        discovered = viz_registry.discover_plotter_types()

    assert "good" in discovered
    assert "broken" not in discovered
    assert "_private" not in discovered


def test_list_plotter_types_sorted(monkeypatch):
    monkeypatch.setattr(
        viz_registry,
        "discover_plotter_types",
        lambda: {"z": object(), "a": object()},
        raising=False,
    )
    assert viz_registry.list_plotter_types() == ["a", "z"]
