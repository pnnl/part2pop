# tests/unit/freezing/test_builder.py

import numpy as np
import pytest

from part2pop.population.builder import build_population
import part2pop.freezing.builder as fb
from part2pop.freezing.base import FreezingPopulation, retrieve_Jhet_val
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


def test_build_freezing_population_with_kelvin():
    pop = _make_monodisperse_population()
    cfg = {
        "T_units": "K",
        "T_grid": [250.0],
        "morphology": "homogeneous",
        "cooling_rate": -1.0,
    }

    frz_pop = build_freezing_population(pop, cfg)
    ff = frz_pop.get_frozen_fraction(cfg["cooling_rate"])
    assert ff.shape == frz_pop.T_grid.shape
    assert np.all(ff >= 0.0)


class _StubFreezingParticle:
    def __init__(self, jhet, insa):
        self._jhet = np.array(jhet, dtype=float)
        self.INSA = np.array(insa, dtype=float)

    def get_Jhet(self, T):
        return self._jhet


def test_freezing_population_math_helpers():
    base = _make_monodisperse_population()
    T_grid = np.array([230.0, 240.0])
    frz_pop = FreezingPopulation(base, T_grid=T_grid)

    particle = _StubFreezingParticle(jhet=[1.0, 2.0], insa=[0.1, 0.2])
    frz_pop.add_freezing_particle(particle, part_id=base.ids[0], T=T_grid)

    assert np.allclose(frz_pop.get_avg_Jhet(), particle._jhet)
    ns = frz_pop.get_nucleating_sites(dT_dt=1.0)
    assert ns.shape == T_grid.shape
    assert np.all(np.isfinite(ns))

    ff = frz_pop.get_frozen_fraction(dT_dt=1.0)
    assert ff.shape == T_grid.shape
    assert np.all(np.isfinite(ff))

    probs = frz_pop.get_freezing_probs()
    assert probs.shape == frz_pop.Jhet.shape
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_freezing_population_descending_T_branch():
    base = _make_monodisperse_population()
    T_grid = np.array([250.0, 240.0])
    frz_pop = FreezingPopulation(base, T_grid=T_grid)

    particle = _StubFreezingParticle(jhet=[0.5, 0.5], insa=[0.3, 0.3])
    frz_pop.add_freezing_particle(particle, part_id=base.ids[0], T=T_grid)

    ns = frz_pop.get_nucleating_sites(dT_dt=0.5)
    ff = frz_pop.get_frozen_fraction(dT_dt=0.5)
    assert ns.shape == T_grid.shape
    assert ff.shape == T_grid.shape


def test_add_freezing_particle_invalid_id_raises():
    base = _make_monodisperse_population()
    frz_pop = FreezingPopulation(base, T_grid=[260.0])
    particle = _StubFreezingParticle(jhet=[1.0], insa=[0.1])

    with pytest.raises(ValueError):
        frz_pop.add_freezing_particle(particle, part_id=9999, T=[260.0])


def test_retrieve_Jhet_val_with_overrides():
    default_m, default_b = retrieve_Jhet_val("SO4")
    assert default_m == "26.61"
    assert default_b == "-3.93"

    override_m, override_b = retrieve_Jhet_val(
        "SO4",
        spec_modifications={"m_log10Jhet": "1.23", "b_log10Jhet": "-4.56"},
    )
    assert override_m == "1.23"
    assert override_b == "-4.56"
