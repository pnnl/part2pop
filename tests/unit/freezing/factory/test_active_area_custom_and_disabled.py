"""Custom and disabled freezing materials in one particle.

#127 counts complete custom coefficient pairs in the ice-active area; #136
drops tabulated materials disabled with NaN coefficients. Both PRs carry the
same get_insoluble_radius() and this same test file, so each is complete on
its own and the second to merge applies without conflict.
"""

import numpy as np
import pytest

# Areas here are ~1e-13 m^2: pytest.approx's default absolute tolerance of
# 1e-12 would accept any of them, so every comparison sets abs=0.

from part2pop import AerosolSpecies, Particle, get_species
from part2pop.freezing.builder import build_freezing_particle

NAN_PAIR = {"m_log10_Jhet": np.nan, "b_log10_Jhet": np.nan}
CUSTOM = AerosolSpecies("custom_ice_nucleus", density=2000, kappa=0.0, molar_mass=0.1)


def _insa(species, volumes, modifications):
    masses = np.array(volumes) * np.array([s.density for s in species])
    particle = Particle(tuple(species), masses)
    cfg = {"morphology": "homogeneous", "species_modifications": modifications}
    return build_freezing_particle(particle, cfg).INSA


def _sphere_area(volume):
    return 4 * np.pi * (3 * volume / (4 * np.pi)) ** (2 / 3)


def test_custom_pair_counts_and_disabled_table_material_does_not():
    species = (CUSTOM, get_species("OC", None), get_species("BC", None), get_species("H2O", None))
    modifications = {
        CUSTOM.name: {"m_log10_Jhet": 20.0, "b_log10_Jhet": -2.0},
        "OC": NAN_PAIR,
    }
    # Distinct volumes, so a missing custom volume cannot be offset by a
    # wrongly counted OC volume.
    area = _insa(species, [1e-21, 1.5e-21, 2e-21, 1e-20], modifications)
    assert area == pytest.approx(_sphere_area(3e-21), rel=1e-12, abs=0.0)


def test_incomplete_custom_pair_does_not_count():
    species = (CUSTOM, get_species("BC", None), get_species("H2O", None))
    with pytest.warns(UserWarning, match="missing b_log10_Jhet"):
        area = _insa(species, [1e-21, 2e-21, 1e-20], {CUSTOM.name: {"m_log10_Jhet": 20.0}})
    assert area == pytest.approx(_sphere_area(2e-21), rel=1e-12, abs=0.0)


def test_zero_zero_pair_is_active_but_nan_pair_is_not():
    species = (get_species("OC", None), get_species("BC", None), get_species("H2O", None))
    zero_pair = {"OC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}}
    active = _insa(species, [1e-21, 2e-21, 1e-20], zero_pair)
    disabled = _insa(species, [1e-21, 2e-21, 1e-20], {"OC": NAN_PAIR})
    assert active == pytest.approx(_sphere_area(3e-21), rel=1e-12, abs=0.0)
    assert disabled == pytest.approx(_sphere_area(2e-21), rel=1e-12, abs=0.0)


def test_string_and_numeric_coefficients_are_both_accepted():
    # Table values come back as strings; overrides may be numpy scalars.
    species = (get_species("BC", None), get_species("H2O", None))
    numpy_pair = {"BC": {"m_log10_Jhet": np.float32(18.06), "b_log10_Jhet": np.float64(-2.09)}}
    assert _insa(species, [2e-21, 1e-20], numpy_pair) == pytest.approx(
        _insa(species, [2e-21, 1e-20], {}), rel=1e-12
    )
