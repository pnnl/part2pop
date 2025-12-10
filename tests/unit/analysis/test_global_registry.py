from types import SimpleNamespace

import pytest

from part2pop.analysis import global_registry as gr


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
