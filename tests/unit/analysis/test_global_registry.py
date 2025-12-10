import sys
import types

import pytest

import part2pop.analysis.global_registry as gr


def test_family_to_suffix_and_build_full_variable_name():
    assert gr.family_to_suffix("PopulationVariable") == ".population"
    assert gr.build_full_variable_name("foo", "PopulationVariable") == "foo.population"
    assert gr.build_full_variable_name("already.suffix", "PopulationVariable") == "already.suffix"
    with pytest.raises(ValueError):
        gr.family_to_suffix("BadFamily")


def test_get_variable_builder_uses_discovery(monkeypatch):
    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {})
    with pytest.raises(ValueError):
        gr.get_variable_builder("unknown", {})

    def fake_discover():
        return {"foo.population": lambda cfg=None: "built"}

    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {"PopulationVariable": fake_discover})
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {"foo": "PopulationVariable"})

    builder = gr.get_variable_builder("foo", {})
    assert builder() == "built"

    with pytest.raises(ValueError):
        gr.get_variable_builder("bar", {"family": "MissingFamily"})


def test_get_variable_class_wraps_errors(monkeypatch):
    def bad_builder(cfg=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {"PopulationVariable": lambda: {"bad.population": bad_builder}})
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {"bad": "PopulationVariable"})

    with pytest.raises(RuntimeError):
        gr.get_variable_class("bad")


def test_list_registered_variables(monkeypatch):
    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {
        "PopulationVariable": lambda: {"a.population": lambda cfg=None: cfg},
        "ParticleVariable": lambda: {"b.particle": lambda cfg=None: cfg},
    })
    vars_map = gr.list_registered_variables()
    assert vars_map == {
        "a.population": "PopulationVariable",
        "b.particle": "ParticleVariable",
    }


def test_family_inferred_from_suffix(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {})
    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {
        "PopulationVariable": lambda: {"foo.population": lambda cfg=None: "foo"},
    })

    builder = gr.get_variable_builder("foo.population", {})
    assert builder() == "foo"


def test_list_registered_variables_empty(monkeypatch):
    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {})
    assert gr.list_registered_variables() == {}


def test_build_full_variable_name_handles_dot():
    assert gr.build_full_variable_name("foo.bar", "PopulationVariable") == "foo.bar"


def test_discover_population_and_particle_types(monkeypatch):
    fake_pop = types.SimpleNamespace()
    fake_pop._discover = lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    fake_pop.list_variables = lambda: ["popvar"]
    fake_pop.get_builder = lambda name: (lambda cfg=None: f"pop_{name}")

    fake_par = types.SimpleNamespace()
    fake_par._discover_particle_factories = lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    fake_par.list_particle_variables = lambda: ["partvar"]
    fake_par.get_particle_builder = lambda name: (lambda cfg=None: f"part_{name}")

    monkeypatch.setitem(sys.modules, "part2pop.analysis.population.factory.registry", fake_pop)
    fake_pop_pkg = types.ModuleType("part2pop.analysis.population.factory")
    fake_pop_pkg.registry = fake_pop
    fake_par_pkg = types.ModuleType("part2pop.analysis.particle.factory")
    fake_par_pkg.registry = fake_par
    monkeypatch.setitem(sys.modules, "part2pop.analysis.population.factory", fake_pop_pkg)
    monkeypatch.setitem(sys.modules, "part2pop.analysis.particle.factory", fake_par_pkg)
    monkeypatch.setitem(sys.modules, "part2pop.analysis.particle.factory.registry", fake_par)

    pop_builders = gr._discover_population_variable_types()
    part_builders = gr._discover_particle_variable_types()

    assert "popvar.population" in pop_builders
    assert pop_builders["popvar.population"]() == "pop_popvar"
    assert "partvar.particle" in part_builders
    assert part_builders["partvar.particle"]() == "part_partvar"


def test_get_variable_builder_finds_family_from_suffix(monkeypatch):
    fake_registry = lambda: {"alpha.population": lambda cfg=None: "alpha"}
    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {"PopulationVariable": fake_registry})
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {})

    builder = gr.get_variable_builder("alpha.population", {})
    assert builder({}) == "alpha"


def test_get_variable_builder_lazy_population(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {})
    def discover():
        return {"lazy.population": lambda cfg=None: "lazy"}

    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {"PopulationVariable": discover})
    builder = gr.get_variable_builder("lazy", {})
    assert builder() == "lazy"


def test_family_to_suffix_requires_variable():
    with pytest.raises(ValueError):
        gr.family_to_suffix("BadFamily")


def test_get_variable_class_runtime_error(monkeypatch):
    def bad_builder(cfg=None):
        raise RuntimeError("no class")

    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {"PopulationVariable": lambda: {"bad.population": bad_builder}})
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {"bad": "PopulationVariable"})

    with pytest.raises(RuntimeError):
        gr.get_variable_class("bad")


def test_get_variable_builder_requires_family(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {})
    def exploding():
        raise RuntimeError("boom")
    monkeypatch.setattr(gr, "FAMILY_DISCOVERY_FUNCS", {"PopulationVariable": exploding})

    with pytest.raises(ValueError, match="No family specified"):
        gr.get_variable_builder("unknown", {})


def test_get_variable_builder_missing_variable(monkeypatch):
    monkeypatch.setattr(gr, "DEFAULT_VARIABLE_FAMILIES", {"foo": "PopulationVariable"})
    monkeypatch.setattr(
        gr,
        "FAMILY_DISCOVERY_FUNCS",
        {"PopulationVariable": lambda: {"other.population": lambda cfg=None: "x"}},
    )

    with pytest.raises(ValueError, match="not found"):
        gr.get_variable_builder("foo", {})
