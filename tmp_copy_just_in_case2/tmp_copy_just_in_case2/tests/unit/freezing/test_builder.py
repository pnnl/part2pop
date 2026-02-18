# tests/unit/freezing/test_builder.py

import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.freezing.factory import registry as freezing_registry
from part2pop.freezing.factory.utils import calculate_Psat
from part2pop.population.base import ParticlePopulation
from part2pop.population.builder import build_population
import part2pop.freezing.builder as fb
from part2pop.freezing.builder import (
    build_freezing_particle,
    build_freezing_population,
)


def _make_monodisperse_population():
    """
    Minimal population for freezing tests: SO4 + H2O droplet.
    """
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "H2O"]],
        "N": [1.0e4],
        "D": [0.5e-6],
        "aero_spec_fracs": [[0.2, 0.8]],
    }
    return build_population(cfg)


def test_build_freezing_particle_requires_morphology_key():
    """
    FreezingParticleBuilder should raise if 'morphology' is missing from config.
    """
    pop = _make_monodisperse_population()
    base_particle = pop.get_particle(pop.ids[0])

    with pytest.raises(ValueError, match="morphology"):
        build_freezing_particle(base_particle, {})


def test_build_freezing_particle_homogeneous():
    """
    With morphology='homogeneous', build_freezing_particle should return
    a usable freezing particle that can compute Jhet.
    """
    pop = _make_monodisperse_population()
    base_particle = pop.get_particle(pop.ids[0])

    cfg = {"morphology": "homogeneous"}
    fp = build_freezing_particle(base_particle, cfg)

    assert hasattr(fp, "get_Jhet")
    J = fp.get_Jhet(T=235.0)
    assert np.isfinite(J)
    assert J >= 0.0


def test_build_freezing_population_T_in_C():
    """
    build_freezing_population should accept temperature in °C via T_grid in
    the config and T_units='C', and return a FreezingPopulation with frozen
    fractions between 0 and 1.

    This matches the current implementation, which expects T to come from
    config['T_grid'] when T is not passed explicitly.
    """
    pop = _make_monodisperse_population()

    cfg = {
        "T_units": "C",
        "T_grid": [-30.0],          # °C; builder will convert to K
        "morphology": "homogeneous", # not used here but future-proof
        "cooling_rate": -1.0, 
        "T_units": "C",
    }

    # Do NOT pass T argument; builder will pull from T_grid and make it an array
    frz_pop = build_freezing_population(pop, cfg)

    assert hasattr(frz_pop, "get_frozen_fraction")
    ff = frz_pop.get_frozen_fraction(cfg["cooling_rate"])

    # One frozen fraction per particle
    assert ff.shape[0] == len(pop.ids)
    assert np.all(ff >= 0.0)
    assert np.all(ff <= 1.0)


def test_freezing_particle_builder_validates_type(monkeypatch):
    builder = fb.FreezingParticleBuilder({"morphology": None})
    with pytest.raises(ValueError):
        builder.build(base_particle=object())

    monkeypatch.setattr(fb, "discover_morphology_types", lambda: {"good": lambda p, c: ("ok", p, c)})
    with pytest.raises(ValueError):
        fb.FreezingParticleBuilder({"morphology": "bad"}).build(base_particle=object())

    result = fb.FreezingParticleBuilder({"morphology": "good"}).build(base_particle="p")
    assert result[1] == "p"


def test_build_freezing_population_unknown_units(monkeypatch):
    class _StubFreezePop:
        def __init__(self, base, T): self.base = base; self.T=T
        def add_freezing_particle(self, fp, pid, T): pass
    monkeypatch.setattr(fb, "FreezingPopulation", _StubFreezePop)
    monkeypatch.setattr(fb, "build_freezing_particle", lambda base_particle, cfg: ("fp", cfg))

    base = type("P", (), {"ids": [1], "get_particle": lambda self, pid: "p"})()
    with pytest.raises(ValueError):
        fb.build_freezing_population(base, {"T_units": "X"})


def test_calculate_psat_helpers_increase_with_temperature():
    low_wv, low_ice = calculate_Psat(260.0)
    high_wv, high_ice = calculate_Psat(280.0)
    assert high_wv > low_wv
    assert high_ice > low_ice


def test_calculate_psat_returns_positive_values():
    psat_wv, psat_ice = calculate_Psat(270.0)
    assert psat_wv > 0.0
    assert psat_ice > 0.0


class _FakeFreezingParticle:
    def __init__(self, *args, **kwargs):
        self._jhet = np.array([1.0])
        self.INSA = np.array([1.0])

    def get_Jhet(self, T):
        return self._jhet


def _build_base_population():
    particle = make_particle(1e-6, ["SO4"], [1.0])
    return ParticlePopulation(
        species=particle.species,
        spec_masses=np.asarray([particle.masses]),
        num_concs=np.asarray([1.0]),
        ids=[3],
    )


def test_freezing_particle_builder_requires_morphology():
    with pytest.raises(ValueError):
        fb.FreezingParticleBuilder({}).build(_FakeFreezingParticle())


def test_freezing_particle_builder_unknown_type(monkeypatch):
    monkeypatch.setattr(freezing_registry, "discover_morphology_types", lambda: {})
    builder = fb.FreezingParticleBuilder({"morphology": "missing"})
    with pytest.raises(ValueError):
        builder.build(_FakeFreezingParticle())


def test_build_freezing_population_runs_for_C_and_K(monkeypatch):
    monkeypatch.setattr(
        fb,
        "discover_morphology_types",
        lambda: {"stub": lambda base, cfg: _FakeFreezingParticle()},
    )
    base = _build_base_population()
    cfg = {"morphology": "stub", "T_grid": [270.0], "T_units": "C"}
    pop_c = fb.build_freezing_population(base, cfg)
    assert hasattr(pop_c, "Jhet")

    cfg["T_units"] = "K"
    pop_k = fb.build_freezing_population(base, cfg)
    assert hasattr(pop_k, "Jhet")


def test_build_freezing_population_bad_units():
    base = _build_base_population()
    with pytest.raises(ValueError):
        fb.build_freezing_population(base, {"morphology": "stub", "T_units": "M"})
