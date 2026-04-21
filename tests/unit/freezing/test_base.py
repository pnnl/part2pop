# tests/unit/freezing/test_base.py

import io

import numpy as np
import pytest
import part2pop.freezing.builder as fb
from part2pop.aerosol_particle import make_particle
from part2pop.freezing.base import FreezingPopulation, retrieve_Jhet_val
from part2pop.population.base import ParticlePopulation
from part2pop.freezing import build_freezing_population


def test_retrieve_Jhet_val_for_known_species():
    m, b = retrieve_Jhet_val("BC", spec_modifications={})
    float(m)
    float(b)


def _base_population():
    particle = make_particle(1e-6, ["SO4","BC","H2O"], [0.1, 0.1, 0.8])
    return ParticlePopulation(
        species=particle.species,
        spec_masses=np.asarray([particle.masses]),
        num_concs=np.asarray([1.0]),
        ids=[1],
    )

def test_freezing_population_updates_and_aggregates():
    base_population = _base_population()
    T_grid = np.array([270.0, 280.0])
    config = {"T_units": "K",
              "T_grid": T_grid,
              "morphology": "homogeneous"}
    freezing_pop = FreezingPop=build_freezing_population(base_population, config)
    
    avg = freezing_pop.get_avg_Jhet()
    assert np.isfinite(avg.all())
    assert avg.shape == T_grid.shape
    
    sites = freezing_pop.get_nucleating_sites(dT_dt=1.0)
    assert not np.any(np.isnan(sites))

    frozen = freezing_pop.get_frozen_fraction(dT_dt=1.0)
    assert np.all(np.isfinite(frozen))
    
    probs = freezing_pop.get_freezing_probs(dt=1.0)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_freezing_population_with_decreasing_T_grid():
    base_population = _base_population()
    T_grid = np.array([280.0, 230.0])
    config = {"T_units": "K",
              "T_grid": T_grid,
              "morphology": "homogeneous"}
    freezing_pop = build_freezing_population(base_population, config)
    sites = freezing_pop.get_nucleating_sites(dT_dt=2.0)
    frozen = freezing_pop.get_frozen_fraction(dT_dt=2.0)
    assert np.all(sites >= 0.0)
    assert np.all((frozen >= 0.0) & (frozen <= 1.0))


def test_retrieve_jhet_val_uses_modifications():
    base_population = _base_population()
    T_grid = np.array([270.0])
    config = {"T_units": "K",
              "T_grid": T_grid,
              "morphology": "homogeneous"}
    freezing_pop = build_freezing_population(base_population, config)
    Jhet1 = freezing_pop.Jhet[0][0]
    config = {"T_units": "K",
              "T_grid": T_grid,
              "morphology": "homogeneous",
              "species_modifications": {"BC": {"m_log10Jhet": 0.0}}}
    freezing_pop = build_freezing_population(base_population, config)
    Jhet2 = freezing_pop.Jhet[0][0]
    assert Jhet1 != Jhet2

def particle_not_in_ids():
    base_population = _base_population()
    base_particle = base_population.get_particle(base_population.ids[0])
    result = fb.FreezingParticleBuilder({"morphology": "homogeneous"}).build(base_particle)
    T_grid = np.array([270.0])
    freezing_pop = FreezingPopulation(base_population, T_grid)
    with pytest.raises(ValueError):
        freezing_pop.add_freezing_particle(result,999,T_grid)
