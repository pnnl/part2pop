# tests/unit/population/test_base.py

import numpy as np
import pytest
from types import SimpleNamespace

from part2pop.population.base import ParticlePopulation
from part2pop.aerosol_particle import make_particle
from part2pop.species.registry import get_species


def _empty_population():
    """Create a truly empty ParticlePopulation with the right shapes."""
    # Start with a dummy species entry; will be overwritten on first add_particle
    dummy_species = ()
    spec_masses = np.zeros((0, 0))
    num_concs = np.zeros((0,))
    ids = []
    return ParticlePopulation(
        species=dummy_species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=ids,
    )


def _make_simple_particle():
    sulfate = get_species("SO4")
    D_wet = 0.1e-6
    return make_particle(
        D=D_wet,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )


def test_particle_population_add_and_get_particle():
    """
    Start from an empty ParticlePopulation and use add_particle via set_particle
    to populate it, then recover the particle and its number concentration.
    """
    p = _make_simple_particle()
    pop = _empty_population()

    # set_particle will call add_particle for new id
    pop.set_particle(particle=p, part_id=5, num_conc=1e8, suppress_warning=False)

    assert 5 in pop.ids
    idx = list(pop.ids).index(5)
    assert np.isclose(pop.num_concs[idx], 1e8)

    p_out = pop.get_particle(5)
    assert np.isclose(p_out.get_Dwet(), p.get_Dwet())
    assert np.isclose(p_out.get_mass_tot(), p.get_mass_tot())


def test_particle_population_set_particle_overwrites_existing():
    """
    When we call set_particle with an existing id, the masses and num_conc
    should be overwritten in place (not duplicated).
    """
    p1 = _make_simple_particle()
    pop = _empty_population()

    pop.set_particle(particle=p1, part_id=1, num_conc=1e8)
    assert len(pop.ids) == 1

    p2 = _make_simple_particle()
    # Change the number concentration
    pop.set_particle(particle=p2, part_id=1, num_conc=3e8)

    assert len(pop.ids) == 1  # still one entry
    idx = list(pop.ids).index(1)
    assert np.isclose(pop.num_concs[idx], 3e8)


def test_particle_population_total_number_concentration():
    """
    get_Ntot should sum num_concs over all stored particles.
    """
    p = _make_simple_particle()
    pop = _empty_population()

    pop.set_particle(particle=p, part_id=1, num_conc=1e8)
    pop.set_particle(particle=p, part_id=2, num_conc=2e8)

    N_tot = pop.get_Ntot()
    assert np.isclose(N_tot, 3e8)


# ---------------------------------------------------------------------------
# Extended tests using a stubbed Particle to isolate population logic
# ---------------------------------------------------------------------------


class _FakeParticle:
    def __init__(self, species, masses):
        self.species = species
        self.masses = np.array(masses, dtype=float)

    def get_Dwet(self, *args, **kwargs):
        # diameter in meters; here just proportional to total mass for simplicity
        return float(self.masses.sum())

    def idx_h2o(self):
        for i, spec in enumerate(self.species):
            if spec.name == "H2O":
                return i
        return -1

    def get_variable(self, varname, *args, **kwargs):
        # return sum of masses for any variable name
        return float(self.masses.sum())


def _make_stub_population(monkeypatch):
    # Two species including H2O
    species = (SimpleNamespace(name="H2O"), SimpleNamespace(name="SO4"))
    spec_masses = np.array([[1.0, 2.0]])
    num_concs = np.array([5.0])
    ids = [1]
    monkeypatch.setattr("part2pop.population.base.Particle", _FakeParticle)
    return ParticlePopulation(species, spec_masses, num_concs, ids)


def test_find_and_get_particle_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    assert pop.find_particle(1) == 0
    assert pop.find_particle(99) == len(pop.ids)

    particle = pop.get_particle(1)
    assert isinstance(particle, _FakeParticle)
    with pytest.raises(ValueError):
        pop.get_particle(99)


def test_set_and_add_particle_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    new_particle = _FakeParticle(pop.species, [3.0, 4.0])
    pop.set_particle(new_particle, part_id=1, num_conc=2.5)
    assert pop.num_concs[0] == 2.5
    assert pop.spec_masses[0, 0] == 3.0

    # Add a brand new particle id
    another = _FakeParticle(pop.species, [1.0, 1.0])
    pop.set_particle(another, part_id=2, num_conc=1.0)
    assert len(pop.ids) == 2
    assert pop.num_concs[-1] == 1.0


def test_population_mass_and_radius_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    # effective radius uses get_Dwet / 2 averaged by num_concs
    eff_r = pop.get_effective_radius()
    expected = (pop.get_particle(1).get_Dwet() / 2.0)
    assert eff_r == expected

    assert pop.get_tot_mass() == np.sum(pop.num_concs * np.sum(pop.spec_masses, axis=1))
    assert pop.get_tot_dry_mass() == pop.get_tot_mass() - pop.num_concs[0] * pop.spec_masses[0, 0]
    assert pop.get_mass_conc("SO4") == pop.num_concs[0] * pop.spec_masses[0, 1]


def test_get_particle_var_and_hist_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    vals = pop.get_particle_var("any")
    assert vals.shape == (len(pop.ids),)

    hist, edges = pop.get_num_dist_1d(N_bins=2, density=False)[0:2]
    assert hist.shape == (2,)

    with pytest.raises(NotImplementedError):
        pop.get_num_dist_1d(method="kde")
