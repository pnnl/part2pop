from types import SimpleNamespace

import pytest

from part2pop.analysis import global_registry as gr
from part2pop.analysis.particle.factory import registry as particle_registry
from part2pop.analysis.population.factory import registry as population_registry


def test_family_to_suffix_and_full_name():
    assert gr.family_to_suffix("PopulationVariable") == ".population"
    assert gr.build_full_variable_name("foo", "PopulationVariable") == "foo.population"
    assert gr.build_full_variable_name("bar.population", "PopulationVariable") == "bar.population"

    with pytest.raises(ValueError):
        gr.family_to_suffix("BadFamily")


def test_get_variable_builder_resolves_alias(monkeypatch):
    builder = lambda cfg: SimpleNamespace(cfg=cfg)
    builder.meta = SimpleNamespace(default_cfg={})
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {"foo.population": builder}},
        raising=False,
    )
    monkeypatch.setattr(
        gr,
        "DEFAULT_VARIABLE_FAMILIES",
        {"foo": "PopulationVariable"},
        raising=False,
    )
    got = gr.get_variable_builder("foo")
    assert got is builder


def test_get_variable_builder_missing_family(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {}, raising=False)
    with pytest.raises(ValueError):
        gr.get_variable_builder("nope")


def test_get_variable_class_bubbles_instantiation_error(monkeypatch):
    def fail_builder(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {"bad.population": fail_builder}},
        raising=False,
    )
    monkeypatch.setattr(
        gr,
        "DEFAULT_VARIABLE_FAMILIES",
        {"bad": "PopulationVariable"},
        raising=False,
    )

    with pytest.raises(RuntimeError):
        gr.get_variable_class("bad")


def test_list_registered_variables_produces_mapping(monkeypatch):
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {
            "PopulationVariable": lambda: {"foo.population": lambda cfg=None: cfg},
            "ParticleVariable": lambda: {"bar.particle": lambda cfg=None: cfg},
        },
        raising=False,
    )
    mapping = gr.list_registered_variables()
    assert mapping["foo.population"] == "PopulationVariable"
    assert mapping["bar.particle"] == "ParticleVariable"


def test_family_suffix_inference_from_full_name(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {}, raising=False)
    builder = lambda cfg=None: SimpleNamespace(cfg=cfg)
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {"foo.population": builder}},
        raising=False,
    )
    result = gr.get_variable_builder("foo.population")
    assert result is builder


def test_build_full_variable_name_with_existing_suffix():
    assert gr.build_full_variable_name("foo.population", "PopulationVariable") == "foo.population"


def test_get_variable_builder_unknown_family_raises(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {}, raising=False)
    with pytest.raises(ValueError):
        gr.get_variable_builder("missing")


def test_discover_population_handles_discovery_exceptions(monkeypatch):
    monkeypatch.setattr(
        population_registry,
        "_discover",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(population_registry, "list_variables", lambda: ["foo"], raising=False)
    monkeypatch.setattr(
        population_registry,
        "get_builder",
        lambda name: lambda cfg=None: SimpleNamespace(name=name),
        raising=False,
    )
    mapping = gr._discover_population_variable_types()
    assert "foo.population" in mapping


def test_discover_particle_handles_discovery_exceptions(monkeypatch):
    monkeypatch.setattr(
        particle_registry,
        "_discover_particle_factories",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(particle_registry, "list_particle_variables", lambda: ["bar"], raising=False)
    monkeypatch.setattr(
        particle_registry,
        "get_particle_builder",
        lambda name: lambda cfg=None: SimpleNamespace(name=name),
        raising=False,
    )
    mapping = gr._discover_particle_variable_types()
    assert "bar.particle" in mapping


def test_discover_population_success(monkeypatch):
    builder = lambda cfg=None: SimpleNamespace(name="foo")
    monkeypatch.setattr(population_registry, "_discover", lambda: None, raising=False)
    monkeypatch.setattr(population_registry, "list_variables", lambda: ["foo"], raising=False)
    monkeypatch.setattr(population_registry, "get_builder", lambda name: builder, raising=False)
    mapping = gr._discover_population_variable_types()
    assert mapping["foo.population"] is builder


def test_discover_particle_success(monkeypatch):
    builder = lambda cfg=None: SimpleNamespace(name="bar")
    monkeypatch.setattr(particle_registry, "_discover_particle_factories", lambda: None, raising=False)
    monkeypatch.setattr(particle_registry, "list_particle_variables", lambda: ["bar"], raising=False)
    monkeypatch.setattr(
        particle_registry,
        "get_particle_builder",
        lambda name: builder,
        raising=False,
    )
    mapping = gr._discover_particle_variable_types()
    assert mapping["bar.particle"] is builder


def test_get_variable_builder_with_config_family(monkeypatch):
    builder = lambda cfg=None: SimpleNamespace(cfg=cfg)
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {"foo.population": builder}},
        raising=False,
    )
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {}, raising=False)
    result = gr.get_variable_builder("foo", config={"family": "PopulationVariable"})
    assert result is builder


def test_get_variable_builder_unknown_family(monkeypatch):
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {}},
        raising=False,
    )
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {}, raising=False)
    with pytest.raises(ValueError, match="Unknown family"):
        gr.get_variable_builder("foo", config={"family": "UnknownFamily"})


def test_get_variable_builder_missing_variable(monkeypatch):
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {}},
        raising=False,
    )
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {"foo": "PopulationVariable"}, raising=False)
    with pytest.raises(ValueError, match="not found"):
        gr.get_variable_builder("foo")
