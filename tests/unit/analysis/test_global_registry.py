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
