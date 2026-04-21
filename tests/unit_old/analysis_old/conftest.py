import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.population.base import ParticlePopulation
from part2pop.species.registry import get_species


def _empty_population():
    """Create an empty ParticlePopulation shell with the right shapes."""
    return ParticlePopulation(
        species=(),
        spec_masses=np.zeros((0, 0)),
        num_concs=np.zeros((0,)),
        ids=[],
        species_modifications={},
    )


def make_simple_population(
    diameters=(1.0e-6, 2.0e-6),
    num_concs=(100.0, 200.0),
    s_crit=(0.15, 0.05),
    kappas=(0.3, 0.5),
):
    """
    Build a small ParticlePopulation with real Particle instances.

    Supersaturation thresholds and kappa values are overridden to keep tests
    fast and deterministic.
    """
    species_list = [get_species("SO4"), get_species("BC")]
    base_pop = _empty_population()

    for idx, D in enumerate(diameters):
        particle = make_particle(
            D=D,
            aero_spec_names=species_list,
            aero_spec_frac=np.array([0.6, 0.4]),
            species_modifications={},
            D_is_wet=True,
        )
        # Override heavy computations with deterministic values
        sc_val = s_crit[idx]
        particle.get_critical_supersaturation = lambda T, return_D_crit=False, val=sc_val, part=particle: (
            (val, part.get_Dwet()) if return_D_crit else val
        )
        particle.get_tkappa = lambda val=kappas[idx]: val

        base_pop.set_particle(particle=particle, part_id=idx, num_conc=num_concs[idx])

    # Provide the convenience method expected by some variables
    def get_species_mass_conc(self, name):
        try:
            return self.get_mass_conc(name)
        except Exception:
            return 0.0

    base_pop.get_species_mass_conc = get_species_mass_conc.__get__(base_pop, ParticlePopulation)
    return base_pop


@pytest.fixture
def simple_population():
    return make_simple_population()
