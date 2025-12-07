# tests/unit/test_aerosol_particle.py

import numpy as np

from part2pop.aerosol_particle import make_particle, make_particle_from_masses, Particle
from part2pop.species.registry import get_species


def test_make_particle_basic_properties():
    """Create a simple SO4-only particle and check derived properties."""

    sulfate = get_species("SO4")

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

    sulfate = get_species("SO4")
    organics = get_species("OC")

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

    sulfate = get_species("SO4")
    D_wet = 0.1e-6

    p = make_particle(
        D=D_wet,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )

    p2 = make_particle_from_masses(
        aero_spec_names=[sulfate.name],
        spec_masses=p.masses.copy(),
    )

    assert np.isclose(p.get_Dwet(), p2.get_Dwet())
    assert np.isclose(p.get_mass_tot(), p2.get_mass_tot())
