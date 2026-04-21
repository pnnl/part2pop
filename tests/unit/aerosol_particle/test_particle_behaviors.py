import numpy as np
import pytest

from part2pop.aerosol_particle import (
    AerosolSpecies,
    Particle,
    compute_Dwet,
    compute_mass_h2o,
    effective_density,
    make_particle,
    make_particle_from_masses,
)


def _make_species(name, density, kappa, molar_mass):
    return AerosolSpecies(name=name, density=density, kappa=kappa, molar_mass=molar_mass)


def _build_particle():
    specs = (
        _make_species("H2O", 1000.0, 0.0, 18e-3),
        _make_species("BC", 1800.0, 0.0, 12e-3),
        _make_species("SO4", 1770.0, 0.5, 98e-3),
    )
    masses = (1.0e-18, 2.0e-18, 3.0e-18)
    return Particle(species=specs, masses=masses)


def test_particle_mass_and_volume_helpers():
    particle = _build_particle()
    assert np.isclose(particle.get_mass_dry(), 5.0e-18)
    assert np.isclose(particle.get_mass_tot(), 6.0e-18)
    assert particle.get_vol_tot() > particle.get_vol_dry()
    assert particle.get_vol_core() <= particle.get_vol_dry()
    assert particle.get_tkappa() >= 0.0
    assert np.isclose(particle.get_Ddry(), (particle.get_vol_dry() * 6.0 / np.pi) ** (1.0 / 3.0))


def test_compute_Dwet_returns_no_change_at_zero_RH():
    particle = _build_particle()
    ddry = particle.get_Ddry()
    assert np.isclose(compute_Dwet(ddry, particle.get_tkappa(), RH=0.0, T=298.0), ddry)


def test_compute_mass_h2o_difference_is_correct():
    ddry = 1e-6
    dwet = 1.1e-6
    mass = compute_mass_h2o(ddry, dwet)
    assert mass == pytest.approx(np.pi / 6.0 * (dwet**3 - ddry**3) * 1000.0)


def test_effective_density_uses_inverse_sum():
    specs = (_make_species("SO4", 1500.0, 0.5, 98e-3), _make_species("BC", 1800.0, 0.0, 12e-3))
    fracs = np.array([0.4, 0.6])
    dens = effective_density(fracs, specs)
    inverse_sum = sum(fracs[i] / specs[i].density for i in range(len(specs)))
    assert np.isclose(dens, 1.0 / inverse_sum)


def test_make_particle_appends_missing_water():
    specs = (_make_species("BC", 1800.0, 0.0, 12e-3),)
    particle = make_particle(D=200e-9, aero_spec_names=[specs[0]], aero_spec_frac=[1.0])
    assert any(spec.name.upper() == "H2O" for spec in particle.species)


def test_make_particle_raises_if_fractions_not_normalized():
    spec_names = ["H2O", "BC"]
    fracs = [0.5, 0.6]
    with pytest.raises(ValueError):
        make_particle(D=100e-9, aero_spec_names=spec_names, aero_spec_frac=fracs)


def test_make_particle_from_masses_returns_particle_copy():
    spec_names = ["H2O", "BC"]
    masses = np.array([1e-18, 2e-18])
    particle = make_particle_from_masses(spec_names, masses)
    assert np.isclose(particle.get_mass_tot(), np.sum(masses))