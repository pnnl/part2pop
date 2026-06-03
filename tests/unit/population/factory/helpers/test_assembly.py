import numpy as np
import pytest

from part2pop.population.base import ParticlePopulation
from part2pop.population.factory.helpers.assembly import assemble_population_from_mass_fractions


def test_assembly_helper_basic_population_creation():
    pop = assemble_population_from_mass_fractions(
        diameters=[1e-7],
        number_concentrations=[2.0],
        species_names=[["SO4"]],
        mass_fractions=[[1.0]],
    )
    assert isinstance(pop, ParticlePopulation)
    assert len(pop.ids) == 1
    assert np.isclose(pop.get_Ntot(), 2.0)


def test_assembly_helper_species_alignment_across_particles():
    pop = assemble_population_from_mass_fractions(
        diameters=[1e-7, 2e-7],
        number_concentrations=[1.0, 1.0],
        species_names=[["SO4"], ["OC", "SO4"]],
        mass_fractions=[[1.0], [0.25, 0.75]],
    )
    names = [s.name for s in pop.species]
    assert names[:2] == ["SO4", "OC"]
    assert "H2O" in names
    assert pop.spec_masses.shape[0] == 2

    so4_idx = names.index("SO4")
    oc_idx = names.index("OC")
    assert pop.spec_masses[0, so4_idx] > 0
    assert pop.spec_masses[0, oc_idx] == 0
    assert pop.spec_masses[1, so4_idx] > 0
    assert pop.spec_masses[1, oc_idx] > 0


def test_assembly_helper_preserves_classes_and_ids():
    pop = assemble_population_from_mass_fractions(
        diameters=[1e-7, 2e-7],
        number_concentrations=[1.0, 3.0],
        species_names=[["SO4"], ["SO4"]],
        mass_fractions=[[1.0], [1.0]],
        ids=[10, 20],
        classes=["a", "b"],
    )
    assert pop.ids == [10, 20]
    assert pop.classes == ["a", "b"]


def test_assembly_helper_invalid_shapes_raise():
    with pytest.raises(ValueError):
        assemble_population_from_mass_fractions(
            diameters=[1e-7],
            number_concentrations=[1.0, 2.0],
            species_names=[["SO4"]],
            mass_fractions=[[1.0]],
        )

    with pytest.raises(ValueError):
        assemble_population_from_mass_fractions(
            diameters=[1e-7],
            number_concentrations=[1.0],
            species_names=[["SO4", "OC"]],
            mass_fractions=[[1.0]],
        )


def test_assembly_helper_spec_masses_is_2d_regression():
    pop = assemble_population_from_mass_fractions(
        diameters=[1e-7, 2e-7, 3e-7],
        number_concentrations=[1.0, 2.0, 3.0],
        species_names=[["BC"], ["OIN"], ["SO4", "OC"]],
        mass_fractions=[[1.0], [1.0], [0.6, 0.4]],
    )
    assert pop.spec_masses.ndim == 2
    assert pop.spec_masses.shape[0] == 3
    assert pop.spec_masses.shape[1] == len(pop.species)


def test_assembly_helper_nested_fraction_row_raises():
    with pytest.raises(ValueError):
        assemble_population_from_mass_fractions(
            diameters=[1e-7],
            number_concentrations=[1.0],
            species_names=[["SO4", "OC"]],
            mass_fractions=[[[0.5, 0.5]]],
        )
