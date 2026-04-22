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
    T = 270.0
    config = {"morphology": "homogeneous"}
    freezing_pop = build_freezing_population(base_population, config)
    avg = freezing_pop.get_avg_Jhet(T, config)
    assert np.isfinite(avg)

    probs = freezing_pop.get_freezing_probs(T, config,dt=1.0)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

def test_freezing_population_with_decreasing_T_grid():
    base_population = _base_population()
    T_grid = np.array([230.0, 280.0])
    config = {"morphology": "homogeneous"}
    freezing_pop = build_freezing_population(base_population, config)
    avg_Jhet_cold = freezing_pop.get_avg_Jhet(T_grid[0], config)
    avg_Jhet_warm = freezing_pop.get_avg_Jhet(T_grid[1], config)
    freezing_probs_cold = freezing_pop.get_freezing_probs(T_grid[0], config, dt=1.0)
    freezing_probs_warm = freezing_pop.get_freezing_probs(T_grid[1], config, dt=1.0)
    assert avg_Jhet_cold >= 0.0 and avg_Jhet_warm >= 0.0
    assert avg_Jhet_cold >= avg_Jhet_warm
    assert freezing_probs_cold >= 0.0 and freezing_probs_warm >= 0.0
    assert freezing_probs_cold >= freezing_probs_warm


def test_retrieve_jhet_val_uses_modifications():
    base_population = _base_population()
    T = 270.0
    config = {"morphology": "homogeneous"}
    freezing_pop = build_freezing_population(base_population, config)
    Jhet1 = freezing_pop.compute_Jhets(T, config)
    config = {"morphology": "homogeneous",
              "species_modifications": {"BC": {"m_log10_Jhet": 0.0}}}
    freezing_pop = build_freezing_population(base_population, config)
    Jhet2 = freezing_pop.compute_Jhets(T, config)
    assert Jhet1 != Jhet2

def particle_not_in_ids():
    base_population = _base_population()
    base_particle = base_population.get_particle(base_population.ids[0])
    result = fb.FreezingParticleBuilder({"morphology": "homogeneous"}).build(base_particle)
    T_grid = np.array([270.0])
    freezing_pop = FreezingPopulation(base_population, T_grid)
    with pytest.raises(ValueError):
        freezing_pop.add_freezing_particle(result,999,T_grid)
