# tests/unit/species/test_registry.py

import os
import tempfile

from part2pop.species.base import AerosolSpecies
from part2pop.species.registry import (
    get_species,
    register_species,
    list_species,
    describe_species,
)


def test_get_species_loads_default_data_case_insensitive():
    """Check that SO4 parameters are loaded from aero_data and name handling is case-insensitive."""

    specdata_path = None
    so4_upper = get_species("SO4", specdata_path)
    so4_lower = get_species("so4",specdata_path)

    assert isinstance(so4_upper, AerosolSpecies)
    assert isinstance(so4_lower, AerosolSpecies)

    # Same object or at least same values
    assert so4_upper.name.upper() == "SO4"
    assert so4_lower.name.upper() == "SO4"
    assert so4_upper.density == so4_lower.density
    assert so4_upper.kappa == so4_lower.kappa
    assert so4_upper.molar_mass == so4_lower.molar_mass


def test_get_species_resolves_dust_alias_to_oin():
    specdata_path = None
    dust = get_species("Dust", specdata_path)

    assert isinstance(dust, AerosolSpecies)
    assert dust.name.upper() == "OIN"
    assert dust.density is not None
    assert dust.kappa is not None
    assert dust.molar_mass is not None


def test_get_species_resolves_soot_alias_to_bc():
    specdata_path = None
    soot = get_species("soot", specdata_path)

    assert isinstance(soot, AerosolSpecies)
    assert soot.name.upper() == "BC"
    assert soot.density is not None
    assert soot.kappa is not None
    assert soot.molar_mass is not None


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
    specdata_path = None
    retrieved = get_species(new_name, specdata_path)
    assert isinstance(retrieved, AerosolSpecies)
    assert retrieved.name == new_name
    assert retrieved.density == 1500.0
    assert retrieved.kappa == 0.3
    assert retrieved.molar_mass == 1234.


def test_describe_species_custom_registered():
    new_name = "DESCSPEC"
    spec = AerosolSpecies(
        name=new_name,
        density=1111.0,
        kappa=0.11,
        molar_mass=11.0,
        surface_tension=0.07,
    )
    register_species(spec)

    info = describe_species(new_name)
    assert info["name"] == new_name
    assert info["type"] == "AerosolSpecies"
    assert info["defaults"]["density"] == 1111.0


def test_describe_species_default_from_data():
    info = describe_species("SO4")
    assert info["name"].upper() == "SO4"
    assert info["type"] == "AerosolSpecies"
    assert info["defaults"]["density"] is not None


def test_describe_species_unknown_raises():
    import uuid
    missing = f"MISSING_{uuid.uuid4().hex}"
    try:
        describe_species(missing)
    except ValueError as exc:
        assert "Unknown species" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown species")


def test_describe_species_custom_specdata_path():
    """describe_species should use a custom specdata_path when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a minimal aero_data.dat with a unique species
        dat_path = os.path.join(tmpdir, "aero_data.dat")
        with open(dat_path, "w") as fh:
            fh.write("# density  ions  molar_mass  kappa\n")
            fh.write("CUSTSPEC  1234.0  0  99d-3  0.42\n")

        info = describe_species("CUSTSPEC", specdata_path=tmpdir)

    assert info["name"].upper() == "CUSTSPEC"
    assert info["defaults"]["density"] == 1234.0
    assert info["defaults"]["kappa"] == 0.42


def test_get_species_unknown_alias_fails_clearly():
    missing = "definitely_unknown_species_label"

    try:
        get_species(missing, None)
    except ValueError as exc:
        message = str(exc)
        assert missing in message
        assert "not found" in message or "Unknown species" in message
    else:
        raise AssertionError("Expected ValueError for unknown species")


def test_registry_get_resolves_alias_to_custom_registered_species():
    from part2pop.species.registry import AerosolSpeciesRegistry

    registry = AerosolSpeciesRegistry()
    registry.register(
        AerosolSpecies(
            name="OIN",
            density=999.0,
            kappa=0.99,
            molar_mass=99.0,
            surface_tension=0.09,
        )
    )

    resolved = registry.get("dust", None, kappa=0.5)

    assert resolved.name == "OIN"
    assert resolved.density == 999.0
    assert resolved.kappa == 0.5
    assert resolved.molar_mass == 99.0
    assert resolved.surface_tension == 0.09


def test_describe_species_resolves_alias_to_custom_registered_species(monkeypatch):
    import part2pop.species.registry as species_registry

    registry = species_registry.AerosolSpeciesRegistry()
    registry.register(
        AerosolSpecies(
            name="OC",
            density=888.0,
            kappa=0.88,
            molar_mass=88.0,
            surface_tension=0.08,
        )
    )
    monkeypatch.setattr(species_registry, "_registry", registry)

    info = species_registry.describe_species("POA")

    assert info["name"] == "OC"
    assert info["type"] == "AerosolSpecies"
    assert info["defaults"]["density"] == 888.0
    assert info["defaults"]["kappa"] == 0.88
    assert info["defaults"]["molar_mass"] == 88.0
    assert info["defaults"]["surface_tension"] == 0.08


def test_get_species_alias_resolved_name_missing_from_custom_specdata():
    """Error message must include both the original alias and the resolved canonical name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal dataset that does NOT contain OIN (the target of 'dust')
        dat_path = os.path.join(tmpdir, "aero_data.dat")
        with open(dat_path, "w") as fh:
            fh.write("# minimal dataset without OIN\n")
            fh.write("SO4  1840.0  3  96d-3  1.2\n")

        try:
            get_species("dust", tmpdir)
        except ValueError as exc:
            message = str(exc)
            assert "dust" in message
            assert "OIN" in message
        else:
            raise AssertionError(
                "Expected ValueError when resolved species is missing from custom specdata_path"
            )
