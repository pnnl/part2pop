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
        if not name.startswith("_") and name not in {"registry", "helpers"}
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
        return SimpleNamespace(meta=None)

    monkeypatch.setattr(
        analysis_factory_registry,
        "get_population_builder",
        lambda name: builder,
        raising=False,
    )
    with pytest.raises(analysis_factory_registry.UnknownVariableError):
        analysis_factory_registry.describe_variable("tempvar")


def test_discover_population_types_skips_private_and_broken_modules(monkeypatch):
    monkeypatch.setattr(factory_registry, "_DISCOVERED", False)
    monkeypatch.setattr(factory_registry, "_registry", {})
    monkeypatch.setattr(
        factory_registry,
        "pkgutil",
        SimpleNamespace(
            iter_modules=lambda paths: [
                (None, "_private", False),
                (None, "helpers", True),
                (None, "registry", False),
                (None, "broken", False),
                (None, "good", False),
            ]
        ),
        raising=False,
    )

    def fake_import(fullname):
        if fullname.endswith("broken"):
            raise ImportError("missing optional dep")
        if fullname.endswith("good"):
            return SimpleNamespace(build=lambda cfg: cfg)
        return SimpleNamespace()

    monkeypatch.setattr(factory_registry.importlib, "import_module", fake_import)

    with pytest.warns(RuntimeWarning):
        discovered = factory_registry.discover_population_types()

    assert "good" in discovered
    assert "broken" not in discovered
    assert "_private" not in discovered
    assert "helpers" not in discovered


def test_list_population_types_sorted(monkeypatch):
    monkeypatch.setattr(
        factory_registry,
        "discover_population_types",
        lambda: {"z": object(), "a": object()},
        raising=False,
    )
    assert factory_registry.list_population_types() == ["a", "z"]


def test_describe_population_type(monkeypatch):
    def builder(cfg):
        """Example population builder."""
        return cfg

    monkeypatch.setattr(
        factory_registry,
        "discover_population_types",
        lambda: {"demo": builder},
        raising=False,
    )
    info = factory_registry.describe_population_type("demo")
    assert info["name"] == "demo"
    assert info["type"] == "builder"
    assert "Example population builder" in info["description"]


def test_describe_population_type_unknown(monkeypatch):
    monkeypatch.setattr(factory_registry, "discover_population_types", lambda: {}, raising=False)
    with pytest.raises(ValueError, match="Unknown population type"):
        factory_registry.describe_population_type("missing")


def test_list_population_types_includes_observation_builders():
    types = factory_registry.list_population_types()
    assert "edx_observations" in types
    assert "hiscale_observations" in types


def test_describe_population_type_for_observation_builders():
    edx = factory_registry.describe_population_type("edx_observations")
    hiscale = factory_registry.describe_population_type("hiscale_observations")
    assert edx["name"] == "edx_observations"
    assert hiscale["name"] == "hiscale_observations"
    assert edx["module"] is not None
    assert hiscale["module"] is not None
