import numpy as np
from part2pop.aerosol_particle import make_particle
from part2pop.population.base import ParticlePopulation


def test_clone_detaches_ids_species_modifications_and_arrays():
    part = make_particle(100e-9, ['SO4'], [1.])
    original = ParticlePopulation(species=tuple(part.species),
        spec_masses=np.array([part.masses]), num_concs=np.array([10.]),
        ids=[1], classes=['dust'], species_modifications={'SO4': {'kappa':.3}})
    other = original.clone_detached()
    other.ids[0] = 2
    other.spec_masses[0, 0] = 0.
    other.num_concs[0] = 20.
    other.species_modifications['SO4']['kappa'] = 0.9
    other.species[0].density = 999.
    assert original.ids == [1]
    assert original.spec_masses[0, 0] != 0.
    assert original.num_concs[0] == 10.
    assert original.species_modifications['SO4']['kappa'] == .3
    assert original.species[0].density != 999.
