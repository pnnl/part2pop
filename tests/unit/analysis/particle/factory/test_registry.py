import importlib
import pkgutil
from types import SimpleNamespace

import pytest

import part2pop.analysis.particle.factory.registry as reg
from part2pop.analysis.base import VariableMeta


def _reset_registry(monkeypatch):
    monkeypatch.setattr(reg, "_PARTICLE_REG", {})
    monkeypatch.setattr(reg, "_PARTICLE_ALIASES", {})
    monkeypatch.setattr(reg, "_PARTICLE_DISCOVERED", True)


def test_import_registry():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.particle.factory.registry")


def test_register_particle_variable_and_alias(monkeypatch):
    _reset_registry(monkeypatch)

    @reg.register_particle_variable("foo")
    class Dummy:
        meta = VariableMeta(
            name="foo",
            axis_names=("x",),
            description="desc",
            aliases=("bar",),
            default_cfg={"a": 1},
        )

        def __init__(self, cfg):
            self.cfg = cfg

    assert reg.resolve_particle_name("foo") == "foo"
    assert reg.resolve_particle_name("bar") == "foo"


def test_resolve_particle_suggestions(monkeypatch):
    _reset_registry(monkeypatch)
    reg._PARTICLE_REG["alpha"] = lambda cfg=None: cfg
    with pytest.raises(reg.UnknownParticleVariableError) as excinfo:
        reg.resolve_particle_name("alph")
    assert "alpha" in str(excinfo.value)


def test_get_particle_builder_unknown(monkeypatch):
    _reset_registry(monkeypatch)
    reg._PARTICLE_REG["known"] = lambda cfg=None: cfg
    with pytest.raises(KeyError):
        reg.get_particle_builder("missing")


def test_describe_particle_variable_uses_instance_meta(monkeypatch):
    _reset_registry(monkeypatch)

    class WithMeta:
        meta = VariableMeta(
            name="derived",
            axis_names=("y",),
            description="desc",
            aliases=(),
            default_cfg={"a": 2},
        )

        def __init__(self, cfg):
            self.cfg = cfg

    def factory(cfg=None):
        return WithMeta(cfg or {})

    reg._PARTICLE_REG["derived"] = factory
    info = reg.describe_particle_variable("derived")
    assert info["name"] == "derived"
    assert info["axis_keys"] == ["y"]
    assert info["defaults"] == {"a": 2}


def test_list_particle_variables_includes_aliases(monkeypatch):
    _reset_registry(monkeypatch)
    monkeypatch.setattr(reg, "_PARTICLE_REG", {"foo": lambda cfg=None: cfg})
    monkeypatch.setattr(reg, "_PARTICLE_ALIASES", {"alias": "foo"})
    monkeypatch.setattr(reg, "_PARTICLE_DISCOVERED", True)
    names = reg.list_particle_variables(include_aliases=True)
    assert "foo" in names and "alias" in names


def test_describe_particle_variable_handles_missing_meta(monkeypatch):
    _reset_registry(monkeypatch)

    def bad_builder(cfg=None):
        raise RuntimeError("boom")

    reg._PARTICLE_REG["broken"] = bad_builder
    with pytest.raises(reg.UnknownParticleVariableError):
        reg.describe_particle_variable("broken")


def test_discover_particle_factories_imports_module(monkeypatch):
    _reset_registry(monkeypatch)
    monkeypatch.setattr(reg, "_PARTICLE_DISCOVERED", False)

    fake_module = SimpleNamespace()
    fake_module.build = lambda cfg=None: "built"

    def fake_iter(paths):
        return [(None, "fakepart", False)]

    monkeypatch.setattr(pkgutil, "iter_modules", fake_iter)

    orig_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.endswith(".fakepart"):
            return fake_module
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    reg._discover_particle_factories()
    assert "fakepart" in reg._PARTICLE_REG
