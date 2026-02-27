# tests/unit/freezing/factory/test_homogeneous.py

import numpy as np
from part2pop.freezing.factory.utils import calculate_Psat, calculate_dPsat_dT
import pytest

# def _make_droplet_with_inp():
#     """
#     Construct a physically reasonable wet particle using the public
#     `make_particle` helper, rather than manually instantiating Particle.

#     Returns
#     -------
#     base_particle : part2pop.aerosol_particle.Particle
#     cfg : dict
#         Configuration dict passed to HomogeneousParticle. Currently only
#         `species_modifications` is used by the model.
#     """
#     # 0.5 µm wet diameter, 10% insoluble OC, 90% water by mass.
#     D_wet = 0.5e-6
#     aero_spec_names = ["BC", "H2O"]
#     aero_spec_frac = [0.1, 0.9]

#     base_particle = make_particle(
#         D=D_wet,
#         aero_spec_names=aero_spec_names,
#         aero_spec_frac=aero_spec_frac,
#         D_is_wet=True,
#     )

#     # Minimal config that is consistent with the implementation:
#     cfg = {
#         "species_modifications": {},
#     }
#     return base_particle, cfg

# def _make_droplet_without_inp():
#     """
#     Construct a physically reasonable wet particle using the public
#     `make_particle` helper, rather than manually instantiating Particle.

#     Returns
#     -------
#     base_particle : part2pop.aerosol_particle.Particle
#     cfg : dict
#         Configuration dict passed to HomogeneousParticle. Currently only
#         `species_modifications` is used by the model.
#     """
#     # 0.5 µm wet diameter, 10% soluble SO4, 90% water by mass.
#     D_wet = 0.5e-6
#     aero_spec_names = ["SO4", "H2O"]
#     aero_spec_frac = [0.05, 0.95]

#     base_particle = make_particle(
#         D=D_wet,
#         aero_spec_names=aero_spec_names,
#         aero_spec_frac=aero_spec_frac,
#         D_is_wet=True,
#     )

#     # Minimal config that is consistent with the implementation:
#     cfg = {
#         "species_modifications": {},
#     }
#     return base_particle, cfg


def test_Psat():
    """
    Temperatures less than zero K should give an error.
    """
    Psat_liq, Psat_ice = calculate_Psat(np.array([250]))
    assert np.isfinite(Psat_liq)
    assert np.isfinite(Psat_ice)

    with pytest.raises(ValueError):
        calculate_Psat(np.array([-10.0]))

def test_dPsat_dT():
    """
    Test operation of calculate_dPsat_dT. 
    """
    dp_dT_liq, dp_dT_ice = calculate_dPsat_dT(np.array([250]))
    assert np.isfinite(dp_dT_liq)
    assert np.isfinite(dp_dT_ice)

    with pytest.raises(ValueError):
        calculate_dPsat_dT(np.array([-10.0]))




