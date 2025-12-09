# tests/unit/test_aerosol_particle.py

import numpy as np
import pytest
from scipy.constants import R

import part2pop.aerosol_particle as ap
from part2pop.aerosol_particle import (
    Particle,
    compute_Dwet,
    compute_Sc_funsixdeg,
    compute_mass_h2o,
    effective_density,
    make_particle,
    make_particle_from_masses,
)
from part2pop.species.base import AerosolSpecies
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


def test_particle_variable_helpers_and_accessors(monkeypatch):
    sulfate = get_species("SO4")
    p = make_particle(
        D=0.2e-6,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )

    monkeypatch.setattr(ap, "T", 298.15, raising=False)
    monkeypatch.setattr(
        Particle,
        "get_critical_supersaturation",
        lambda self, T, return_D_crit=False, sigma_h2o=None: 42.0,
    )

    assert np.isclose(p.get_variable("wet_diameter"), p.get_Dwet())
    assert np.isclose(p.get_variable("dry_diameter"), p.get_Ddry())
    assert np.isclose(p.get_variable("tkappa"), p.get_tkappa())
    sc = p.get_variable("critical_supersaturation", 298.15)
    assert sc > 0.0

    original_mass = p.get_spec_mass("SO4")
    new_mass = original_mass * 1.1
    p.set_spec_mass("SO4", new_mass)
    assert np.isclose(p.get_spec_mass("SO4"), new_mass)
    assert p.get_spec_vol("SO4") > 0.0
    assert p.get_spec_moles("SO4") > 0.0
    assert p.get_spec_rho("SO4") > 0.0


def test_particle_equilibrate_and_density():
    sulfate = get_species("SO4")
    p = make_particle(
        D=0.3e-6,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )

    before = p.get_mass_h2o()
    p._equilibrate_h2o(RH=1.2, T=298.15)
    assert p.get_mass_h2o() >= before
    assert np.isclose(p.get_trho(), np.sum(p.masses) / np.sum(p.get_vks()))

    with pytest.warns(UserWarning):
        assert p.get_surface_tension() == 0.072


def test_tkappa_shell_and_root_solvers():
    sulfate = get_species("SO4")
    bc = get_species("BC")
    p = make_particle(
        D=0.2e-6,
        aero_spec_names=[bc.name, sulfate.name],
        aero_spec_frac=np.array([0.4, 0.6]),
        species_modifications={},
        D_is_wet=True,
    )

    tkappa_full = p.get_tkappa()
    tkappa_shell = p.get_shell_tkappa()
    assert tkappa_shell != tkappa_full
    assert p.get_critical_supersaturation(298.15) > 0.0


def test_compute_functions_behavior():
    Ddry = 1e-7
    Dwet = 1.2e-7
    mass_h2o = compute_mass_h2o(Ddry, Dwet, rho_h2o=1000.0)
    assert mass_h2o == pytest.approx(np.pi / 6.0 * (Dwet**3 - Ddry**3) * 1000.0)


def test_make_particle_requires_unit_mass_fraction():
    sulfate = get_species("SO4")
    with pytest.raises(ValueError):
        make_particle(
            D=0.1e-6,
            aero_spec_names=[sulfate.name],
            aero_spec_frac=np.array([0.5]),
            species_modifications={},
            D_is_wet=True,
        )


def test_make_particle_appends_water_when_missing():
    sulfate = get_species("SO4")
    particle = make_particle(
        D=0.1e-6,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )

    assert particle.species[-1].name.upper() == "H2O"
    assert np.isclose(float(particle.get_spec_mass("H2O")), 0.0, atol=1e-12)


def test_make_particle_dry_mode_assigns_water_mass():
    sulfate = get_species("SO4")
    water = get_species("H2O")
    particle = make_particle(
        D=0.2e-6,
        aero_spec_names=[sulfate.name, water.name],
        aero_spec_frac=np.array([0.6, 0.4]),
        species_modifications={},
        D_is_wet=False,
    )

    assert particle.species[-1].name.upper() == "H2O"
    assert float(particle.get_spec_mass("H2O")) > 0.0


def test_compute_Dwet_sigma_priority(monkeypatch):
    captured_As = []

    def _fake_brentq(func, a, b):
        closure = func.__closure__ or ()
        kw = dict(
            zip(
                func.__code__.co_freevars,
                [cell.cell_contents for cell in closure],
            )
        )
        captured_As.append(kw.get("A"))
        return 1.3

    monkeypatch.setattr(ap.opt, "brentq", _fake_brentq)
    Ddry = 1e-7
    compute_Dwet(Ddry, 0.5, 0.6, 298.15, sigma_sa=0.02, sigma_h2o=0.01)
    compute_Dwet(Ddry, 0.5, 0.6, 298.15, sigma_sa=0.04, sigma_h2o=None)

    expected_A = lambda sigma: 4.0 * sigma * 18e-3 / (R * 298.15 * 1000.0)
    assert captured_As[0] == pytest.approx(expected_A(0.02))
    assert captured_As[1] == pytest.approx(expected_A(0.04))


def test_compute_Dwet_returns_ddry_for_nonpositive_inputs():
    Ddry = 1e-7
    assert compute_Dwet(Ddry, 0.0, 0.5, 298.15) == Ddry
    assert compute_Dwet(Ddry, 0.5, 0.0, 298.15) == Ddry


def test_particle_mass_and_volume_helpers():
    sulfate = get_species("SO4")
    bc = get_species("BC")
    particle = make_particle(
        D=0.2e-6,
        aero_spec_names=[bc.name, sulfate.name],
        aero_spec_frac=np.array([0.3, 0.7]),
        species_modifications={},
        D_is_wet=True,
    )

    assert np.isclose(particle.get_mass_tot(), np.sum(particle.masses))
    assert particle.get_mass_dry() <= particle.get_mass_tot()
    assert particle.get_vol_dry() <= particle.get_vol_tot()
    assert particle.get_vol_core() <= particle.get_vol_dry()
    assert particle.get_vol_dry_shell() >= 0
    assert particle.get_spec_rhos().shape[0] == len(particle.species)
    assert particle.get_spec_kappas().shape[0] == len(particle.species)
    assert particle.get_spec_MWs().shape[0] == len(particle.species)

    bc_index = particle.idx_core()[0]
    assert bc_index == 0
    assert np.isfinite(particle.get_vol_core())
    assert np.isfinite(particle.get_mass_h2o())
    assert particle.get_spec_moles("BC") > 0
    assert particle.get_spec_rho("BC") > 0


def test_get_critical_supersaturation_handles_idx_errors(monkeypatch):
    sulfate = get_species("SO4")
    oc = get_species("OC")
    particle = make_particle(
        D=0.3e-6,
        aero_spec_names=[sulfate.name, oc.name],
        aero_spec_frac=np.array([0.5, 0.5]),
        species_modifications={},
        D_is_wet=True,
    )

    called = {"raised": False}

    def _raise_idx(_self):
        called["raised"] = True
        raise IndexError

    monkeypatch.setattr(Particle, "get_Ddry", lambda self: 1e-7)
    monkeypatch.setattr(Particle, "get_tkappa", lambda self: 0.3)
    monkeypatch.setattr(Particle, "idx_h2o", _raise_idx)
    sc = particle.get_critical_supersaturation(298.15)
    assert called["raised"]
    assert sc > 0.0


def test_effective_density_from_explicit_species():
    specs = [
        AerosolSpecies(name="X1", density=2.0, kappa=0.0, molar_mass=1.0),
        AerosolSpecies(name="X2", density=4.0, kappa=0.0, molar_mass=1.0),
    ]
    fracs = np.array([0.5, 0.5])
    dens = effective_density(fracs, specs)
    assert dens == pytest.approx(1.0 / (0.5 / 2.0 + 0.5 / 4.0))


def test_make_particle_validation_and_h2o_insertion():
    sulfate = get_species("SO4")
    with pytest.raises(ValueError):
        make_particle(
            D=0.1e-6,
            aero_spec_names=[sulfate.name],
            aero_spec_frac=np.array([0.5]),
            species_modifications={},
            D_is_wet=True,
        )

    p = make_particle(
        D=0.1e-6,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )
    names = [spec.name for spec in p.species]
    assert "H2O" in names


def test_particle_without_h2o_critical_supersaturation():
    sulfate = get_species("SO4")
    # Construct Particle manually without H2O
    part = Particle(
        species=(sulfate,),
        masses=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="No H2O"):
        part.get_critical_supersaturation(298.15)


def test_particle_idx_helpers_and_properties():
    sulfate = get_species("SO4")
    bc = get_species("BC")
    water = get_species("H2O")
    part = Particle(
        species=(bc, sulfate, water),
        masses=np.array([1.0, 0.5, 0.2]),
    )

    assert part.idx_h2o() == 2
    assert bc in [part.species[idx] for idx in part.idx_core()]
    assert sulfate in [part.species[idx] for idx in part.idx_dry_shell()]
    assert isinstance(part.idx_spec("SO4")[0], np.ndarray)
    assert part.get_spec_rhos().shape[0] == 3
    assert part.get_mass_dry() == pytest.approx(1.5)
    assert part.get_mass_tot() == pytest.approx(1.7)
    assert part.get_vol_dry() > 0.0
    part.set_spec_mass("SO4", 0.7)
    assert part.get_spec_mass("SO4") == pytest.approx(0.7)
