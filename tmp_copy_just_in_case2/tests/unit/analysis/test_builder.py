from types import SimpleNamespace

import pytest

from part2pop.analysis import builder as builder_mod
from part2pop.analysis.base import VariableMeta


def test_get_registry_builder_rejects_unknown_scope():
    with pytest.raises(ValueError):
        builder_mod._get_registry_builder("unknown")


def test_variable_builder_merges_configs_and_warns_for_alias(monkeypatch):
    class FakeBuilder(SimpleNamespace):
        pass

    fake_builder = lambda cfg: FakeBuilder(cfg=cfg)
    fake_builder.meta = VariableMeta(
        name="canon",
        axis_names=("x",),
        description="desc",
        default_cfg={"meta_value": 2},
    )

    monkeypatch.setattr(builder_mod, "_ALIASES", {"alias": "canon"}, raising=False)
    monkeypatch.setattr(builder_mod, "resolve_name", lambda name: "canon")
    monkeypatch.setattr(builder_mod, "resolve_particle_name", lambda name: name)
    monkeypatch.setattr(builder_mod, "_get_defaults_for_var", lambda name: {"global_value": 1})
    monkeypatch.setattr(
        builder_mod,
        "_get_registry_builder",
        lambda scope: (lambda name: fake_builder),
        raising=False,
    )

    with pytest.warns(DeprecationWarning, match="deprecated"):
        vb = builder_mod.VariableBuilder("alias", cfg={"user_value": 3}, scope="population")
    vb.modify(runtime_value=4)

    built = vb.build()
    assert hasattr(built, "cfg")
    assert built.cfg["global_value"] == 1
    assert built.cfg["meta_value"] == 2
    assert built.cfg["user_value"] == 3
    assert built.cfg["runtime_value"] == 4


def test_build_variable_for_particle_scope(monkeypatch):
    class Dummy:
        meta = VariableMeta(
            name="particle",
            axis_names=("x",),
            description="particle variable",
            default_cfg={"base": "value"},
        )

        def __init__(self, cfg):
            self.cfg = dict(cfg)

    monkeypatch.setattr(
        builder_mod,
        "_get_registry_builder",
        lambda scope: (lambda name: Dummy),
        raising=False,
    )
    monkeypatch.setattr(builder_mod, "resolve_particle_name", lambda name: "particle")
    built = builder_mod.build_variable("particle", scope="particle", var_cfg={"extra": 1})
    assert built.cfg["base"] == "value"
    assert built.cfg["extra"] == 1


def test_build_variable_invalid_scope_raises():
    with pytest.raises(ValueError):
        builder_mod._get_registry_builder("invalid")


def test_registry_builder_returns_population_and_particle_callables():
    pop_builder = builder_mod._get_registry_builder("population")
    assert callable(pop_builder)
    part_builder = builder_mod._get_registry_builder("particle")
    assert callable(part_builder)


def test_variable_builder_handles_missing_meta_and_default_fallback(monkeypatch):
    class ExplodingBuilder:
        def __call__(self, cfg):
            raise RuntimeError("boom")

    fake_builder = ExplodingBuilder()
    monkeypatch.setattr(
        builder_mod,
        "_get_registry_builder",
        lambda scope: (lambda name: fake_builder),
        raising=False,
    )
    monkeypatch.setattr(builder_mod, "resolve_name", lambda name: name, raising=False)
    monkeypatch.setattr(builder_mod, "_get_defaults_for_var", lambda name: (_ for _ in ()).throw(RuntimeError("no defaults")))

    with pytest.raises(RuntimeError):
        builder_mod.VariableBuilder("foo").build()
