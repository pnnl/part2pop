# tests/unit/species/test_registry.py

from pyparticle.species.base import AerosolSpecies
from pyparticle.species.registry import (
    get_species,
    register_species,
    list_species,
)


def test_get_species_loads_default_data_case_insensitive():
    """Check that SO4 parameters are loaded from aero_data and name handling is case-insensitive."""

    so4_upper = get_species("SO4")
    so4_lower = get_species("so4")

    assert isinstance(so4_upper, AerosolSpecies)
    assert isinstance(so4_lower, AerosolSpecies)

    # Same object or at least same values
    assert so4_upper.name.upper() == "SO4"
    assert so4_lower.name.upper() == "SO4"
    assert so4_upper.density == so4_lower.density
    assert so4_upper.kappa == so4_lower.kappa
    assert so4_upper.molar_mass == so4_lower.molar_mass


def test_register_species_and_list_species_round_trip():
    """Register a custom species and ensure it can be retrieved and listed."""
    
    new_name = "TESTSPEC"

    spec = AerosolSpecies(
        name=new_name,
        density=1500.0,
        kappa=0.3,
        molar_mass=1234.,
        surface_tension=0.072,
    )

    register_species(spec)

    names = list_species()
    assert new_name in names

    retrieved = get_species(new_name)
    assert isinstance(retrieved, AerosolSpecies)
    assert retrieved.name == new_name
    assert retrieved.density == 1500.0
    assert retrieved.kappa == 0.3
    assert retrieved.molar_mass == 1234.
