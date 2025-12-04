# tests/unit/species/test_species_modifications.py

import numpy as np

from pyparticle.aerosol_particle import make_particle
from pyparticle.species.registry import get_species


def test_species_modifications_override_density_and_kappa_in_particle():
    """
    species_modifications in make_particle config should override base species
    properties (e.g., density, kappa) for that particle.
    """

    base = get_species("SO4")
    assert base.density is not None
    assert base.kappa is not None

    # Construct base particle
    D = 0.1e-6
    names = [base.name]
    fracs = np.array([1.0])

    p_base = make_particle(
        D=D,
        aero_spec_names=names,
        aero_spec_frac=fracs,
        species_modifications={},
        D_is_wet=True,
    )

    rho_base = p_base.get_trho()
    kappa_base = p_base.get_tkappa()

    # Now override density and kappa via species_modifications
    new_density = base.density * 2.0
    new_kappa = base.kappa * 0.5

    mods = {
        base.name: {
            "density": new_density,
            "kappa": new_kappa,
        }
    }

    p_mod = make_particle(
        D=D,
        aero_spec_names=names,
        aero_spec_frac=fracs,
        species_modifications=mods,
        D_is_wet=True,
    )

    rho_mod = p_mod.get_trho()
    kappa_mod = p_mod.get_tkappa()

    # Density and kappa of modified particle should change accordingly
    assert rho_mod > rho_base  # since we doubled density
    assert kappa_mod < kappa_base  # since we halved kappa
    assert rho_mod == new_density
    assert kappa_mod == new_kappa
