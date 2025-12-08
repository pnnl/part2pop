# tests/unit/species/test_base.py

import pytest

from part2pop.species import AerosolSpecies


def test_aerosol_species_defaults_and_basic_fields():
    """
    When only the name is provided, AerosolSpecies should look up the
    remaining properties from the packaged species data file.
    """
    s = AerosolSpecies(name="SO4")  # known to exist in species_data/aero_data.dat

    # Basic identity
    assert s.name.upper() == "SO4"

    # All key physical properties should be populated
    assert s.density is not None
    assert s.kappa is not None
    assert s.molar_mass is not None
    assert s.surface_tension is not None

    # And they should be physically sensible
    assert s.density > 0.0
    assert s.molar_mass > 0.0
    assert s.kappa >= 0.0


def test_aerosol_species_allows_explicit_parameters_without_lookup():
    """
    If all key parameters are supplied explicitly, the implementation
    currently *does not* perform any validation. This test documents that
    behavior and protects against accidental changes.
    """
    s = AerosolSpecies(
        name="BADSPEC",
        density=-100.0,    # intentionally unphysical
        kappa=0.5,
        molar_mass=50.0,
        surface_tension=0.072,
    )

    # No error should be raised, and the attributes should be stored verbatim.
    assert s.name == "BADSPEC"
    assert s.density == -100.0
    assert s.kappa == 0.5
    assert s.molar_mass == 50.0
    assert s.surface_tension == 0.072
