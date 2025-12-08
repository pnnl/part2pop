# tests/unit/population/test_base.py

import numpy as np
import pytest

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
