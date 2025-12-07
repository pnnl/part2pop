# tests/unit/species/test_base.py

import pytest

from part2pop.species.base import AerosolSpecies


def test_aerosol_species_defaults_and_basic_fields():
    """Check that AerosolSpecies stores basic fields and uses defaults."""
    s = AerosolSpecies(name="TESTSPEC")

    assert s.name == "TESTSPEC"
    # Defaults from __post_init__ / dataclass (depending on your implementation)
    assert s.surface_tension > 0.0

    # Optional values may be None if not provided
    assert hasattr(s, "density")
    assert hasattr(s, "kappa")
    assert hasattr(s, "molar_mass")


def test_aerosol_species_full_constructor():
    """Construct a fully-specified species and ensure values are preserved."""
    s = AerosolSpecies(
        name="TESTSPEC",
        density=1500.0,
        kappa=0.3,
        molar_mass=1234.,
        surface_tension=0.072,
    )

    assert s.name == "TESTSPEC"
    assert s.density == 1500.0
    assert s.kappa == 0.3
    assert s.molar_mass == 1234.
    assert s.surface_tension == 0.072


def test_aerosol_species_invalid_parameters_raise():
    """
    If __post_init__ enforces any constraints (e.g. non-negative density),
    test that obviously unphysical values are rejected.
    """
    # If you don't currently raise anything, you can skip or xfail this test.
    with pytest.raises(ValueError):
        AerosolSpecies(
            name="BADSPEC",
            density=-100.0,  # clearly unphysical
            kappa=0.3,
            molar_mass=1234.,
        )
