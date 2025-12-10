import numpy as np
import pytest

from part2pop.aerosol_particle import (
    compute_Dwet,
    compute_mass_h2o,
    effective_density,
    make_particle,
)
from types import SimpleNamespace


def test_make_particle_requires_fraction_sum():
    with pytest.raises(ValueError):
        make_particle(1e-6, ["SO4"], [0.5])


def test_make_particle_always_includes_h2o():
    particle = make_particle(1e-6, ["SO4"], [1.0])
    names = [spec.name.upper() for spec in particle.species]
    assert "H2O" in names
    assert np.isclose(np.sum(particle.masses), particle.get_mass_tot())


def test_compute_dwet_skips_when_zero_conditions():
    d = 1e-6
    assert np.isclose(
        compute_Dwet(Ddry=d, kappa=0.0, RH=0.5, T=300.0), d
    )
    assert np.isclose(
        compute_Dwet(Ddry=d, kappa=0.2, RH=0.0, T=300.0), d
    )


def test_compute_mass_h2o_matches_volume_difference():
    Ddry = 1e-6
    Dwet = 1.1e-6
    rho = 997.0
    expected = np.pi / 6.0 * (Dwet ** 3 - Ddry ** 3) * rho
    assert np.isclose(compute_mass_h2o(Ddry, Dwet, rho_h2o=rho), expected)


def test_effective_density_computes_inverse_sum():
    specs = [SimpleNamespace(density=1000.0), SimpleNamespace(density=2000.0)]
    fracs = [0.25, 0.75]
    expected = 1.0 / (fracs[0] / 1000.0 + fracs[1] / 2000.0)
    assert np.isclose(effective_density(fracs, specs), expected)
