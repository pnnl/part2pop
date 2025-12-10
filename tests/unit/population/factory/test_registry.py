# tests/unit/population/factory/test_registry.py

import pkgutil
from types import SimpleNamespace

import pytest

from part2pop.analysis.base import VariableMeta
from part2pop.analysis.population.factory import registry as analysis_factory_registry
from part2pop.population import factory as factory_pkg
from part2pop.population.factory import registry as factory_registry


def test_discover_population_types_covers_all_factory_modules():
    """
    Ensure that discover_population_types finds a build function for every
    population factory module under part2pop.population.factory except
    the registry module itself.

    This guarantees that adding a new <module>.py with a build() function
    automatically makes it discoverable by the builder.
    """
    types = factory_registry.discover_population_types()
    assert isinstance(types, dict)
    for name, fn in types.items():
        assert callable(fn)

    # All .py modules under part2pop.population.factory except "registry"
    module_names = {
        name
        for _, name, _ in pkgutil.iter_modules(factory_pkg.__path__)
        if not name.startswith("_") and name != "registry"
    }

    # discover_population_types returns keys that should match module names
    # (binned_lognormals, sampled_lognormals, monodisperse, mam4, partmc, ...)
    discovered = set(types.keys())

    # If you ever add a factory module without a build() function, this test
    # will fail and remind you to either add build() or explicitly ignore it.
    assert module_names == discovered


def test_describe_variable_returns_meta(monkeypatch):
    fake_meta = VariableMeta(
        name="tempvar",
        axis_names=("d",),
        description="temporary",
        aliases=("alias",),
        default_cfg={"foo": "bar"},
    )

    def builder(cfg=None):
        return SimpleNamespace(meta=fake_meta)

    builder.meta = fake_meta
    monkeypatch.setattr(
        analysis_factory_registry,
        "get_population_builder",
        lambda name: builder,
        raising=False,
    )

    described = analysis_factory_registry.describe_variable("tempvar")
    assert described["name"] == fake_meta.name
    assert described["defaults"] == {"foo": "bar"}
    assert "alias" in described["aliases"]


def test_describe_variable_raises_without_meta(monkeypatch):
    def builder(cfg=None):
        class Dummy:
            meta = None

        return Dummy()

    monkeypatch.setattr(
        analysis_factory_registry,
        "get_population_builder",
        lambda name: builder,
        raising=False,
    )
    with pytest.raises(analysis_factory_registry.UnknownVariableError):
        analysis_factory_registry.describe_variable("tempvar")
