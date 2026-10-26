import numpy as np
import pytest
from part2pop.species import registry

from part2pop import (
    AerosolSpecies,
    get_species,
    make_particle_from_masses,
    register_species,
)


def test_legacy_module_species_lookup_remains_importable():
    from part2pop.aerosol_particle import retrieve_one_species

    assert retrieve_one_species("SO4").name == "SO4"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry._registry, "_custom", dict(registry._registry._custom))


def test_registered_species_can_be_constructed_from_masses():
    s = AerosolSpecies("custom_mass", density=2000.0, kappa=0.1, molar_mass=0.1)
    register_species(s)
    p = make_particle_from_masses([s.name, "H2O"], [1e-18, 0.0])
    assert p.species[0].density == 2000.0
    np.testing.assert_array_equal(p.masses, [1e-18, 0.0])


def test_overrides_do_not_mutate_registered_template():
    s = AerosolSpecies("custom_override", density=2000.0, kappa=0.1, molar_mass=0.1)
    register_species(s)
    p = make_particle_from_masses(
        [s.name], [1e-18], species_modifications={s.name: {"density": 1800.0}}
    )
    assert p.species[0].density == 1800.0
    assert get_species(s.name, None).density == 2000.0


def test_packaged_species_and_unknown_names():
    p = make_particle_from_masses(["SO4", "BC"], [1e-18, 2e-18])
    assert [s.name for s in p.species] == ["SO4", "BC"]
    np.testing.assert_array_equal(p.masses, [1e-18, 2e-18])
    with pytest.raises(ValueError):
        make_particle_from_masses(["definitely_unknown"], [1e-18])


def test_alias_lookup_uses_registry_and_keeps_constructor_inputs_detached():
    register_species(AerosolSpecies("BC", density=1900.0, kappa=0.0, molar_mass=0.001))
    names = ["black carbon", "H2O"]
    masses = np.array([1e-18, 0.0])
    p = make_particle_from_masses(names, masses)
    assert p.species[0].name == "BC"
    assert p.species[0].density == 1900.0
    p.masses[0] = 2e-18
    np.testing.assert_array_equal(masses, [1e-18, 0.0])
    assert names == ["black carbon", "H2O"]
    p.species[0].density = 1.0
    assert get_species("BC", None).density == 1900.0


@pytest.mark.parametrize("alias", ["soot", "black carbon", "Black-Carbon"])
@pytest.mark.parametrize("custom", [False, True])
def test_alias_uses_canonical_override_without_mutation(alias, custom):
    if custom:
        register_species(
            AerosolSpecies("BC", density=1900.0, kappa=0.0, molar_mass=0.001)
        )
    original_density = get_species("BC", None).density
    modifications = {"BC": {"density": 1234.0}}
    particle = make_particle_from_masses(
        [alias], [1e-18], species_modifications=modifications
    )
    assert particle.species[0].name == "BC"
    assert particle.species[0].density == 1234.0
    assert modifications == {"BC": {"density": 1234.0}}
    assert get_species("BC", None).density == original_density


@pytest.mark.parametrize("raw_override", [{}, {"density": 1456.0}])
def test_exact_input_override_wins_as_a_whole_mapping(raw_override):
    template = get_species("BC", None)
    modifications = {
        "BC": {"density": 1234.0, "kappa": 0.5},
        "black carbon": raw_override.copy(),
    }
    particle = make_particle_from_masses(
        ["black carbon"], [1e-18], species_modifications=modifications
    )
    assert particle.species[0].density == raw_override.get("density", template.density)
    assert particle.species[0].kappa == template.kappa
    assert modifications["black carbon"] == raw_override


def test_registered_literal_alias_retains_priority_over_builtin_resolution():
    register_species(AerosolSpecies("soot", density=1900.0, kappa=0.1, molar_mass=0.1))
    particle = make_particle_from_masses(
        ["soot"], [1e-18], species_modifications={"BC": {"density": 1234.0}}
    )
    assert particle.species[0].name == "soot"
    assert particle.species[0].density == 1900.0
