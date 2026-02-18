# tests/unit/freezing/test_base.py

import io

import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.freezing.base import FreezingPopulation, retrieve_Jhet_val
from part2pop.population.base import ParticlePopulation


def test_retrieve_Jhet_val_for_known_species():
    m, b = retrieve_Jhet_val("SO4", spec_modifications={})
    float(m)
    float(b)


def _base_population():
    particle = make_particle(1e-6, ["SO4"], [1.0])
    return ParticlePopulation(
        species=particle.species,
        spec_masses=np.asarray([particle.masses]),
        num_concs=np.asarray([1.0]),
        ids=[1],
    )


class _FakeFreezingParticle:
    def __init__(self, jhet, insa):
        self._jhet = np.asarray(jhet)
        self.INSA = np.asarray(insa)

    def get_Jhet(self, T):
        return self._jhet


def test_freezing_population_updates_and_aggregates():
    base = _base_population()
    T_grid = np.array([270.0, 280.0])
    pop = FreezingPopulation(base, T_grid)
    particle = _FakeFreezingParticle([1.0, 2.0], [1.0, 1.0])
    pop.add_freezing_particle(particle, part_id=1, T=T_grid)

    avg = pop.get_avg_Jhet()
    assert np.allclose(avg, [1.0, 2.0])
    sites = pop.get_nucleating_sites(dT_dt=1.0)
    assert not np.any(np.isnan(sites))
    frozen = pop.get_frozen_fraction(dT_dt=1.0)
    assert np.all(np.isfinite(frozen))
    probs = pop.get_freezing_probs()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_freezing_population_with_decreasing_T_grid():
    base = _base_population()
    T_grid = np.array([280.0, 270.0])
    pop = FreezingPopulation(base, T_grid)
    particle = _FakeFreezingParticle([0.5, 0.75], [1.0, 1.0])
    pop.add_freezing_particle(particle, part_id=1, T=T_grid)

    sites = pop.get_nucleating_sites(dT_dt=2.0)
    frozen = pop.get_frozen_fraction(dT_dt=2.0)
    assert np.all(sites >= 0.0)
    assert np.all((frozen >= 0.0) & (frozen <= 1.0))


def test_add_freezing_particle_missing_id_raises():
    base = _base_population()
    pop = FreezingPopulation(base, np.array([270.0]))
    with pytest.raises(ValueError):
        pop.add_freezing_particle(_FakeFreezingParticle([1.0], [1.0]), part_id=99, T=np.array([270.0]))


def test_retrieve_jhet_val_uses_modifications(monkeypatch):
    fake_data = "SO4 1.0 2.0\n"
    monkeypatch.setattr("part2pop.freezing.base.open_dataset", lambda _: io.StringIO(fake_data))
    mval, bval = retrieve_Jhet_val("SO4", spec_modifications={"m_log10Jhet": "3.5"})
    assert mval == "3.5"
    assert bval == "2.0"
