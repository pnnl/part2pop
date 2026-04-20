# tests/unit/freezing/factory/test_registry.py

import importlib
import pkgutil
from types import SimpleNamespace

import pytest

from part2pop.freezing.factory import registry as freezing_registry


def test_discover_morphology_types_returns_callables():
    morphs = freezing_registry.discover_morphology_types()
    assert isinstance(morphs, dict)
    assert len(morphs) > 0
    for name, builder in morphs.items():
        assert callable(builder)


def test_register_decorator_and_discovery(monkeypatch):
    monkeypatch.setattr(freezing_registry, "_morphology_registry", {}, raising=False)
    monkeypatch.setattr(
        freezing_registry,
        "pkgutil",
        pkgutil,
        raising=False,
    )

    @freezing_registry.register("dummy")
    def builder(base, cfg):
        return (base, cfg)

    discovered = freezing_registry.discover_morphology_types()
    assert "dummy" in discovered
    assert callable(discovered["dummy"])


def test_discover_skips_broken_module(monkeypatch):
    monkeypatch.setattr(
        freezing_registry,
        "pkgutil",
        SimpleNamespace(iter_modules=lambda paths: [(None, "missing", False)]),
        raising=False,
    )
    monkeypatch.setattr(freezing_registry, "_morphology_registry", {}, raising=False)
    def fail_import(full, file_path=None):
        raise ModuleNotFoundError("wonky")
    monkeypatch.setattr(freezing_registry, "_safe_import_module", fail_import, raising=False)
    # Should not raise even if import fails
    _ = freezing_registry.discover_morphology_types()


def test_safe_import_module_uses_loader(monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError("boom")

    class FakeLoader:
        def exec_module(self, module):
            module.value = "loaded"

    fake_spec = SimpleNamespace(loader=FakeLoader())
    fake_module = SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda fullname, file_path=None: fake_spec,
    )
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: fake_module)

    result = freezing_registry._safe_import_module("some.module", file_path="/tmp/fake.py")
    assert getattr(result, "value", None) == "loaded"


def test_discover_includes_module_build(monkeypatch):
    monkeypatch.setattr(
        freezing_registry,
        "pkgutil",
        SimpleNamespace(iter_modules=lambda paths: [(None, "dummy_build", False)]),
        raising=False,
    )
    monkeypatch.setattr(freezing_registry, "_morphology_registry", {}, raising=False)

    module = SimpleNamespace(build=lambda base, cfg=None: "ok")
    monkeypatch.setattr(
        freezing_registry,
        "_safe_import_module",
        lambda fullname, file_path=None: module,
        raising=False,
    )

    morphs = freezing_registry.discover_morphology_types()
    assert "dummy_build" in morphs
