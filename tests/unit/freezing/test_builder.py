# tests/unit/freezing/test_builder.py

import numpy as np
import pytest

from pyparticle.population.builder import build_population
from pyparticle.freezing.builder import (
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
