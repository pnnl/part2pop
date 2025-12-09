# tests/unit/optics/factory/test_registry.py

import importlib
import pkgutil
from types import SimpleNamespace

from part2pop.optics.factory import registry as opt_registry
from part2pop.optics.factory.registry import discover_morphology_types


def test_discover_morphology_types_returns_callables():
    morphs = discover_morphology_types()
    assert isinstance(morphs, dict)
    assert len(morphs) > 0

    for name, builder in morphs.items():
        assert callable(builder)


def test_discover_morphology_types_loads_module(monkeypatch):
    monkeypatch.setattr(opt_registry, "_morphology_registry", {})

    fake_module = SimpleNamespace()
    fake_module.build = lambda base, cfg=None: "ok"

    def fake_iter(paths):
        return [(None, "fake_morph", False)]

    monkeypatch.setattr(opt_registry.pkgutil, "iter_modules", fake_iter)

    orig_import = opt_registry.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.endswith(".fake_morph"):
            return fake_module
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(opt_registry.importlib, "import_module", fake_import)

    morphs = discover_morphology_types()
    assert "fake_morph" in morphs
