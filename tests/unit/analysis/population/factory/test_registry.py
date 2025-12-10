import importlib
import pkgutil
from types import SimpleNamespace

import pytest

import part2pop.analysis.population.factory.registry as reg


def _reset_registry(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {})
    monkeypatch.setattr(reg, "_ALIASES", {})
    monkeypatch.setattr(reg, "_DISCOVERED", True)


def test_register_variable_and_aliases(monkeypatch):
    _reset_registry(monkeypatch)

    @reg.register_variable("foo")
    class Dummy:
        meta = type("Meta", (), {"aliases": ["bar"]})
        def __init__(self, cfg):
            self.cfg = cfg

    assert reg.resolve_name("foo") == "foo"
    assert reg.resolve_name("bar") == "foo"

    with pytest.raises(KeyError):
        reg.register_variable("foo")(Dummy)
    with pytest.raises(KeyError):
        reg.register_variable("baz")(Dummy)  # alias collision


def test_unknown_variable_suggests(monkeypatch):
    _reset_registry(monkeypatch)
    reg._REGISTRY["abc"] = lambda cfg=None: cfg
    with pytest.raises(reg.UnknownVariableError) as excinfo:
        reg.resolve_name("abd")
    assert "abc" in str(excinfo.value)


def test_describe_variable(monkeypatch):
    _reset_registry(monkeypatch)

    class Dummy:
        meta = type(
            "Meta",
            (),
            {
                "name": "foo",
                "axis_names": ["x"],
                "description": "desc",
                "aliases": [],
                "default_cfg": {"a": 1},
                "units": {"x": "m"},
            },
        )
        def __init__(self, cfg):
            self.cfg = cfg

    reg._REGISTRY["foo"] = lambda cfg=None: Dummy(cfg or {})

    info = reg.describe_variable("foo")
    assert info["name"] == "foo"
    assert info["axis_keys"] == ["x"]
    assert info["defaults"] == {"a": 1}

    with pytest.raises(reg.UnknownVariableError):
        reg.describe_variable("missing")


def test_list_variables_includes_aliases(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {"foo": lambda cfg=None: cfg})
    monkeypatch.setattr(reg, "_ALIASES", {"bar": "foo"})
    monkeypatch.setattr(reg, "_DISCOVERED", True)
    names = reg.list_variables(include_aliases=True)
    assert "foo" in names and "bar" in names


def test_get_population_builder_unknown(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {"known": lambda cfg=None: cfg})
    monkeypatch.setattr(reg, "_ALIASES", {})
    monkeypatch.setattr(reg, "_DISCOVERED", True)
    with pytest.raises(reg.UnknownVariableError):
        reg.get_population_builder("missing")


def test_discover_registers_builders(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {})
    monkeypatch.setattr(reg, "_ALIASES", {})
    monkeypatch.setattr(reg, "_DISCOVERED", False)
    monkeypatch.setattr(pkgutil, "iter_modules", lambda paths: [(None, "dummy", False)])

    module = SimpleNamespace(build=lambda cfg=None: SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    reg._discover()

    assert "dummy" in reg._REGISTRY


def test_describe_variable_uses_instance_meta(monkeypatch):
    _reset_registry(monkeypatch)

    class Instance:
        meta = SimpleNamespace(
            name="inst",
            axis_names=("a",),
            description="desc",
            aliases=[],
            default_cfg={"c": 3},
            units={},
        )

    def builder(cfg=None):
        return Instance()

    reg._REGISTRY["inst"] = builder
    info = reg.describe_variable("inst")
    assert info["name"] == "inst"
    assert info["defaults"] == {"c": 3}


def test_discover_uses_spec_loader_on_import_failure(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {})
    monkeypatch.setattr(reg, "_ALIASES", {})
    monkeypatch.setattr(reg, "_DISCOVERED", False)
    monkeypatch.setattr(pkgutil, "iter_modules", lambda paths: [(None, "spec", False)])

    def fake_import(name):
        raise ModuleNotFoundError("boom")

    class FakeLoader:
        def exec_module(self, module):
            module.build = lambda cfg=None: SimpleNamespace(cfg=cfg)

    dummy_spec = SimpleNamespace(loader=FakeLoader())
    dummy_module = SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda fullname, file_path=None: dummy_spec,
    )
    monkeypatch.setattr(
        importlib.util,
        "module_from_spec",
        lambda spec: dummy_module,
    )

    reg._discover()
    assert "spec" in reg._REGISTRY


def test_unknown_variable_suggestions_with_discovery(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {"abc": lambda cfg=None: cfg})
    monkeypatch.setattr(reg, "_ALIASES", {})
    monkeypatch.setattr(reg, "_DISCOVERED", True)
    with pytest.raises(reg.UnknownVariableError) as excinfo:
        reg.resolve_name("abd")
    assert "abc" in str(excinfo.value)
