import importlib
import pkgutil
from types import SimpleNamespace

import pytest

import part2pop.analysis.particle.factory.registry as reg

def test_import_registry():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.particle.factory.registry")


def _reset_registry(monkeypatch):
    monkeypatch.setattr(reg, "_PARTICLE_REG", {})
    monkeypatch.setattr(reg, "_PARTICLE_ALIASES", {})
    monkeypatch.setattr(reg, "_PARTICLE_DISCOVERED", False)


def test_register_variable_with_alias(monkeypatch):
    _reset_registry(monkeypatch)

    @reg.register_particle_variable("foo")
    class Dummy:
        meta = SimpleNamespace(
            aliases=["bar"],
            axis_names=("x",),
            description="desc",
            default_cfg={},
        )

        def __init__(self, cfg):
            self.cfg = cfg

    assert reg.resolve_particle_name("foo") == "foo"
    assert reg.resolve_particle_name("bar") == "foo"

    with pytest.raises(KeyError):
        reg.register_particle_variable("foo")(Dummy)
    with pytest.raises(KeyError):
        reg.register_particle_variable("baz")(Dummy)


def test_discover_particle_factories(monkeypatch):
    _reset_registry(monkeypatch)
    monkeypatch.setattr(pkgutil, "iter_modules", lambda paths: [(None, "dummyvar", False)])
    module = SimpleNamespace(build=lambda cfg=None: SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(importlib, "import_module", lambda name: module)

    reg._discover_particle_factories()
    assert "dummyvar" in reg._PARTICLE_REG


def test_list_particle_variables_includes_alias(monkeypatch):
    _reset_registry(monkeypatch)
    reg._PARTICLE_REG["foo"] = lambda cfg=None: cfg
    reg._PARTICLE_ALIASES["bar"] = "foo"
    reg._PARTICLE_DISCOVERED = True
    names = reg.list_particle_variables(include_aliases=True)
    assert "foo" in names and "bar" in names


def test_describe_particle_variable(monkeypatch):
    _reset_registry(monkeypatch)

    class Dummy:
        meta = SimpleNamespace(
            name="foo",
            axis_names=("x",),
            description="desc",
            aliases=["foo"],
            default_cfg={"a": 1},
            units={"x": "m"},
        )

        def __init__(self, cfg):
            self.cfg = cfg

    reg._PARTICLE_REG["foo"] = lambda cfg=None: Dummy(cfg or {})
    info = reg.describe_particle_variable("foo")
    assert info["name"] == "foo"
    assert info["axis_keys"] == ["x"]
    assert info["defaults"] == {"a": 1}

    with pytest.raises(reg.UnknownParticleVariableError):
        reg.describe_particle_variable("missing")


def test_describe_particle_variable_uses_instance_meta(monkeypatch):
    _reset_registry(monkeypatch)

    class Instance:
        meta = SimpleNamespace(
            name="inst_meta",
            axis_names=("y",),
            description="desc",
            aliases=[],
            default_cfg={"b": 2},
            units=None,
        )

    def builder(cfg=None):
        return Instance()

    reg._PARTICLE_REG["inst_meta"] = builder
    info = reg.describe_particle_variable("inst_meta")
    assert info["name"] == "inst_meta"
    assert info["defaults"] == {"b": 2}


def test_discover_particle_factories_handles_import_fail(monkeypatch):
    _reset_registry(monkeypatch)
    monkeypatch.setattr(pkgutil, "iter_modules", lambda paths: [(None, "fallback", False)])
    # Fail the normal import, force fallback via spec loader
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("boom")))

    class FakeLoader:
        def exec_module(self, module):
            module.build = lambda cfg=None: SimpleNamespace(cfg=cfg)

    fake_spec = SimpleNamespace(loader=FakeLoader())
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda fullname, file_path: fake_spec,
    )
    module_holder = SimpleNamespace()
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: module_holder)

    reg._discover_particle_factories()
    assert "fallback" in reg._PARTICLE_REG
