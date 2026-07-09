import importlib

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
from part2pop.species.registry import get_species


def test_import_aerosol_particle():
    importlib.import_module("part2pop.aerosol_particle")


def test_make_particle_basic_properties():
    """Create a simple SO4-only particle and check derived properties."""
    specdata_path = None
    sulfate = get_species("SO4", specdata_path)

    D_wet = 0.1e-6  # 100 nm
    aero_spec_names = [sulfate.name]
    aero_spec_frac = np.array([1.0])

    p = make_particle(
        D=D_wet,
        aero_spec_names=aero_spec_names,
        aero_spec_frac=aero_spec_frac,
        species_modifications={},
        D_is_wet=True,
    )

    assert isinstance(p, Particle)

    # Wet diameter should be what we passed (within float tolerance)
    assert np.isclose(p.get_Dwet(), D_wet)

    # Dry diameter should be <= wet diameter
    D_dry = p.get_Ddry()
    assert D_dry > 0
    assert D_dry <= p.get_Dwet()

    # Mass and density should be positive and finite
    mass_tot = p.get_mass_tot()
    assert mass_tot > 0
    assert np.isfinite(mass_tot)

    rho = p.get_trho()
    assert rho > 0
    assert np.isfinite(rho)

    # Species list and composition
    assert len(p.species) == 2 # includes water
    assert p.species[0].name.upper() == "SO4"
    assert np.isclose(np.sum(p.masses), mass_tot)


def test_particle_kappa_and_critical_supersaturation_monotonic():
    """
    Check that higher κ -> lower critical supersaturation for fixed dry size.

    Uses two real species from the registry: SO4 (more hygroscopic) and OPOA.
    """
    specdata_path = None
    sulfate = get_species("SO4", specdata_path)
    organics = get_species("OC", specdata_path)

    D_dry = 0.1e-6  # 100 nm (treated as dry)
    T = 298.15

    p_hi = make_particle(
        D=D_dry,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=False,
    )

    p_lo = make_particle(
        D=D_dry,
        aero_spec_names=[organics.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=False,
    )

    k_hi = p_hi.get_tkappa()
    k_lo = p_lo.get_tkappa()

    # We expect sulfate κ > organic κ
    assert k_hi > k_lo

    s_crit_hi = p_hi.get_critical_supersaturation(T=T)
    s_crit_lo = p_lo.get_critical_supersaturation(T=T)

    # Higher κ should activate at lower supersaturation
    assert s_crit_hi < s_crit_lo
    assert s_crit_hi > 0
    assert s_crit_lo > 0


def test_make_particle_from_masses_round_trip():
    """Check that make_particle_from_masses reproduces total mass."""
    specdata_path = None
    sulfate = get_species("SO4", specdata_path)
    D_wet = 0.1e-6

    p = make_particle(
        D=D_wet,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )

    p2 = make_particle_from_masses(
        aero_spec_names=[spec.name for spec in p.species],
        spec_masses=p.masses.copy(),
    )

    assert np.isclose(p.get_Dwet(), p2.get_Dwet())
    assert np.isclose(p.get_mass_tot(), p2.get_mass_tot())


def _make_species(name, density, kappa, molar_mass):
    return AerosolSpecies(name=name, density=density, kappa=kappa, molar_mass=molar_mass)


def test_particle_mass_and_volume_helpers():
    specs = (
        _make_species("H2O", 1000.0, 0.0, 18e-3),
        _make_species("BC", 1800.0, 0.0, 12e-3),
        _make_species("SO4", 1770.0, 0.5, 98e-3),
    )
    particle = Particle(species=specs, masses=(1.0e-18, 2.0e-18, 3.0e-18))

    assert np.isclose(particle.get_mass_dry(), 5.0e-18)
    assert np.isclose(particle.get_mass_tot(), 6.0e-18)
    assert particle.get_vol_tot() > particle.get_vol_dry()
    assert particle.get_vol_core() <= particle.get_vol_dry()
    assert particle.get_tkappa() >= 0.0
    assert np.isclose(particle.get_Ddry(), (particle.get_vol_dry() * 6.0 / np.pi) ** (1.0 / 3.0))


def test_compute_Dwet_returns_no_change_at_zero_rh():
    specs = (
        _make_species("H2O", 1000.0, 0.0, 18e-3),
        _make_species("SO4", 1770.0, 0.5, 98e-3),
    )
    particle = Particle(species=specs, masses=(1.0e-18, 3.0e-18))
    ddry = particle.get_Ddry()

    assert np.isclose(compute_Dwet(ddry, particle.get_tkappa(), RH=0.0, T=298.0), ddry)


def test_compute_mass_h2o_difference_is_correct():
    ddry = 1e-6
    dwet = 1.1e-6

    assert compute_mass_h2o(ddry, dwet) == pytest.approx(np.pi / 6.0 * (dwet**3 - ddry**3) * 1000.0)


def test_effective_density_uses_inverse_sum():
    specs = (_make_species("SO4", 1500.0, 0.5, 98e-3), _make_species("BC", 1800.0, 0.0, 12e-3))
    fracs = np.array([0.4, 0.6])
    inverse_sum = sum(fracs[i] / specs[i].density for i in range(len(specs)))

    assert np.isclose(effective_density(fracs, specs), 1.0 / inverse_sum)


def test_make_particle_appends_missing_water():
    particle = make_particle(D=200e-9, aero_spec_names=["BC"], aero_spec_frac=[1.0])

    assert any(spec.name.upper() == "H2O" for spec in particle.species)


def test_make_particle_raises_if_fractions_not_normalized():
    with pytest.raises(ValueError):
        make_particle(D=100e-9, aero_spec_names=["H2O", "BC"], aero_spec_frac=[0.5, 0.6])


def test_make_particle_applies_species_modifications():
    base = get_species("SO4", None)
    particle_base = make_particle(
        D=0.1e-6,
        aero_spec_names=[base.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )

    new_density = base.density * 2.0
    new_kappa = base.kappa * 0.5
    particle_mod = make_particle(
        D=0.1e-6,
        aero_spec_names=[base.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={base.name: {"density": new_density, "kappa": new_kappa}},
        D_is_wet=True,
    )

    assert particle_mod.get_trho() > particle_base.get_trho()
    assert particle_mod.get_tkappa() < particle_base.get_tkappa()
    assert particle_mod.get_trho() == new_density
    assert particle_mod.get_tkappa() == new_kappa
