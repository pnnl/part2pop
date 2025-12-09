import importlib
import pkgutil
import types

import pytest

from part2pop.analysis.base import VariableMeta
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


def test_discover_populates_registry(monkeypatch):
    monkeypatch.setattr(reg, "_DISCOVERED", False)
    monkeypatch.setattr(reg, "_REGISTRY", {})
    module = types.SimpleNamespace()
    module.build = lambda cfg=None: cfg

    orig_iter = pkgutil.iter_modules
    def fake_iter(paths):
        return [(None, "fakevar", False)]
    monkeypatch.setattr(pkgutil, "iter_modules", fake_iter)

    orig_import = importlib.import_module
    def fake_import(name, *args, **kwargs):
        if name.endswith(".fakevar"):
            return module
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    reg._discover()
    assert "fakevar" in reg._REGISTRY


def test_describe_variable_with_builder_meta(monkeypatch):
    _reset_registry(monkeypatch)

    meta = VariableMeta(
        name="meta",
        axis_names=("x",),
        description="desc",
        aliases=(),
        default_cfg={"a": 3},
        units={"x": "m"},
    )

    def builder(cfg=None):
        return cfg

    builder.meta = meta
    reg._REGISTRY["meta"] = builder

    info = reg.describe_variable("meta")
    assert info["name"] == "meta"
    assert info["units"] == {"x": "m"}


def test_describe_variable_handles_failed_instantiation(monkeypatch):
    _reset_registry(monkeypatch)

    def bad_builder(cfg=None):
        raise RuntimeError("boom")

    reg._REGISTRY["broken"] = bad_builder
    with pytest.raises(reg.UnknownVariableError):
        reg.describe_variable("broken")


def test_list_variables_default_excludes_aliases(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {"foo": lambda cfg=None: cfg})
    monkeypatch.setattr(reg, "_ALIASES", {"alias": "foo"})
    monkeypatch.setattr(reg, "_DISCOVERED", True)
    names = reg.list_variables()
    assert "foo" in names
    assert "alias" not in names
