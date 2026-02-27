# tests/unit/freezing/test_builder.py

import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.freezing.factory.homogeneous import HomogeneousParticle
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
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e4],
        "D": [0.5e-6],
        "aero_spec_fracs": [[0.1, 0.2, 0.7]],
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
    J = fp.get_Jhet(T=np.array([235.0]))
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


def test_freezing_particle_builder_validates_type():
    builder = fb.FreezingParticleBuilder({"morphology": None})
    with pytest.raises(ValueError):
        builder.build(base_particle=object())

    builder = fb.FreezingParticleBuilder({"morphology": 'random'})
    with pytest.raises(ValueError):
        builder.build(base_particle=object())

    base_population = _make_monodisperse_population()
    base_particle = base_population.get_particle(base_population.ids[0])
    result = fb.FreezingParticleBuilder({"morphology": "homogeneous"}).build(base_particle=base_particle)
    assert type(result)==HomogeneousParticle


def test_build_freezing_population_unknown_units():
    base_population = _make_monodisperse_population()
    with pytest.raises(ValueError):
        fb.build_freezing_population(base_population, {"T_units": "X"})


def test_calculate_psat_helpers_increase_with_temperature():
    low_wv, low_ice = calculate_Psat(np.array([260.0]))
    high_wv, high_ice = calculate_Psat(np.array([280.0]))
    assert high_wv > low_wv
    assert high_ice > low_ice


def test_calculate_psat_returns_positive_values():
    psat_wv, psat_ice = calculate_Psat(np.array([270.0]))
    assert psat_wv > 0.0
    assert psat_ice > 0.0


def test_build_freezing_population_runs_for_C_and_K(monkeypatch):
    base_population = _make_monodisperse_population()
    cfg = {"morphology": "homogeneous", "T_grid": np.array([270.0]), "T_units": "C"}
    pop_c = fb.build_freezing_population(base_population, cfg)
    assert hasattr(pop_c, "Jhet")
    assert hasattr(pop_c, "Jhom")
    assert hasattr(pop_c, "INSA")

    cfg["T_units"] = "K"
    pop_k = fb.build_freezing_population(base_population, cfg)
    assert hasattr(pop_k, "Jhet")
    assert hasattr(pop_k, "Jhom")
    assert hasattr(pop_k, "INSA")
