import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from part2pop import Particle, ParticlePopulation, get_species


# Test/documentation gap: the long-double cases use SimpleNamespace to test
# population storage without Particle.__init__'s float64 conversion. They do
# not establish end-to-end precision through the public Particle constructor.
# A constructor precision policy and real-Particle coverage are separate work.


def population():
    species = tuple(get_species(n, None) for n in ["SO4", "BC", "H2O"])
    return ParticlePopulation(
        species,
        np.array([[1e-18, 2e-18, 0.0], [3e-18, 4e-18, 0.0]]),
        np.array([1.0, 2.0]),
        [0, 1],
    )


@pytest.mark.parametrize("backing_dtype", [int, np.float64])
@pytest.mark.parametrize("reordered", [False, True])
def test_replacement_promotes_to_incoming_extended_precision(backing_dtype, reordered):
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("long double has no additional precision on this platform")
    specs = tuple(get_species(n, None) for n in ("SO4", "BC"))
    precise = np.longdouble(1) + np.longdouble(2) ** -60
    pop = ParticlePopulation(
        specs,
        np.array([[1, 2], [3, 4]], dtype=backing_dtype),
        np.array([1, 2], dtype=backing_dtype),
        [0, 1],
    )
    incoming = SimpleNamespace(
        species=specs[::-1] if reordered else specs,
        masses=np.array(
            [3, precise] if reordered else [precise, 3], dtype=np.longdouble
        ),
    )
    pop.set_particle(incoming, 0, precise)
    assert pop.spec_masses.dtype == np.dtype(np.longdouble)
    assert pop.num_concs.dtype == np.dtype(np.longdouble)
    assert pop.spec_masses[0, 0] == precise
    assert pop.num_concs[0] == precise
    np.testing.assert_array_equal(pop.spec_masses[1], [3, 4])
    assert pop.num_concs[1] == 2


@pytest.mark.parametrize("operation", ["add", "set"])
def test_reordered_incoming_particle_preserves_mass_identity(operation):
    pop = population()
    original = pop.spec_masses[1].copy()
    incoming = Particle(
        (pop.species[1], pop.species[0], pop.species[2]), [6e-18, 5e-18, 0.0]
    )
    if operation == "add":
        pop.add_particle(incoming, 2, 3.0)
        row = pop.spec_masses[2]
    else:
        pop.set_particle(incoming, 0, 3.0)
        row = pop.spec_masses[0]
    assert [s.name for s in pop.species] == ["SO4", "BC", "H2O"]
    np.testing.assert_array_equal(row, [5e-18, 6e-18, 0.0])
    np.testing.assert_array_equal(pop.spec_masses[1], original)


@pytest.mark.parametrize("operation", ["add", "set"])
@pytest.mark.parametrize("problem", ["set", "duplicate", "shape", "density"])
def test_invalid_species_update_is_rejected_before_mutation(operation, problem):
    pop = population()
    before = copy.deepcopy(pop)
    specs = list(pop.species)
    masses = [1e-18, 2e-18, 0.0]
    if problem == "set":
        specs[0] = get_species("OC", None)
    if problem == "duplicate":
        specs[0] = specs[1]
    if problem == "density":
        specs[0] = replace(specs[0], density=999.0)
    if problem == "shape":
        masses = [1e-18]
    incoming = (
        SimpleNamespace(species=specs, masses=np.array(masses))
        if problem == "shape"
        else Particle(specs, masses)
    )
    with pytest.raises(ValueError):
        if operation == "add":
            pop.add_particle(incoming, 2, 3.0)
        else:
            pop.set_particle(incoming, 0, 3.0)
    assert pop.species == before.species
    assert pop.ids == before.ids
    np.testing.assert_array_equal(pop.spec_masses, before.spec_masses)
    np.testing.assert_array_equal(pop.num_concs, before.num_concs)


@pytest.mark.parametrize("operation", ["add", "set"])
def test_tuple_ids_accept_aligned_updates(operation):
    pop = population()
    pop.ids = tuple(pop.ids)  # The public dataclass annotates ids as a tuple.
    incoming = Particle(tuple(reversed(pop.species)), [0.0, 6e-18, 5e-18])
    if operation == "add":
        pop.add_particle(incoming, 2, 3.5)
        index = 2
    else:
        pop.set_particle(incoming, 0, 3.5)
        index = 0
    assert list(pop.ids) == ([0, 1, 2] if operation == "add" else [0, 1])
    np.testing.assert_array_equal(pop.spec_masses[index], [5e-18, 6e-18, 0.0])
    assert pop.num_concs[index] == 3.5


def test_integer_backing_arrays_do_not_truncate_fractional_replacement():
    specs = tuple(get_species(n, None) for n in ["SO4", "BC"])
    pop = ParticlePopulation(
        specs, np.array([[1, 2], [3, 4]]), np.array([1, 2]), [0, 1]
    )
    incoming = Particle(tuple(reversed(specs)), [0.25, 0.5])
    pop.set_particle(incoming, 0, 0.75)
    np.testing.assert_array_equal(pop.spec_masses, [[0.5, 0.25], [3.0, 4.0]])
    np.testing.assert_array_equal(pop.num_concs, [0.75, 2.0])


def test_replacement_keeps_extended_precision_in_existing_rows_and_incoming_mass():
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("long double has no additional precision on this platform")
    specs = tuple(get_species(n, None) for n in ("SO4", "BC"))
    precise = np.longdouble(1) + np.longdouble(2) ** -60
    pop = ParticlePopulation(
        specs,
        np.array([[1, 2], [precise, precise]], dtype=np.longdouble),
        np.array([1, precise], dtype=np.longdouble),
        [0, 1],
    )
    incoming = SimpleNamespace(
        species=specs, masses=np.array([precise, 3], dtype=np.longdouble)
    )
    pop.set_particle(incoming, 0, precise)
    assert pop.spec_masses.dtype == np.dtype(np.longdouble)
    assert pop.num_concs.dtype == np.dtype(np.longdouble)
    assert pop.spec_masses[0, 0] == precise
    assert pop.spec_masses[1, 0] == precise
    assert pop.num_concs[0] == precise
    assert pop.num_concs[1] == precise


def test_appending_extended_precision_mass_keeps_incoming_value():
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("long double has no additional precision on this platform")
    specs = tuple(get_species(n, None) for n in ("SO4", "BC"))
    precise = np.longdouble(1) + np.longdouble(2) ** -60
    pop = ParticlePopulation(
        specs,
        np.array([[1, 2]], dtype=np.longdouble),
        np.array([1], dtype=np.longdouble),
        [0],
    )
    incoming = SimpleNamespace(
        species=specs, masses=np.array([precise, 3], dtype=np.longdouble)
    )
    pop.add_particle(incoming, 1, precise)
    assert pop.spec_masses[1, 0] == precise
    assert pop.num_concs[1] == precise


def test_empty_population_adopts_extended_precision_mass():
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("long double has no additional precision on this platform")
    specs = tuple(get_species(n, None) for n in ("SO4", "BC"))
    precise = np.longdouble(1) + np.longdouble(2) ** -60
    pop = ParticlePopulation((), np.empty((0, 0)), np.array([]), ())
    incoming = SimpleNamespace(
        species=specs, masses=np.array([precise, 3], dtype=np.longdouble)
    )
    pop.add_particle(incoming, 0, precise)
    assert pop.spec_masses.dtype == np.dtype(np.longdouble)
    assert pop.spec_masses[0, 0] == precise
    assert pop.num_concs[0] == precise


@pytest.mark.parametrize(
    "field,value", [("kappa", 0.9), ("molar_mass", 0.2), ("surface_tension", 0.05)]
)
@pytest.mark.parametrize("operation", ["add", "set"])
def test_all_physical_identity_fields_are_checked_before_mutation(
    field, value, operation
):
    pop = population()
    before = copy.deepcopy(pop)
    specs = list(pop.species)
    specs[0] = replace(specs[0], **{field: value})
    incoming = Particle(specs, [1e-18, 2e-18, 0.0])
    with pytest.raises(ValueError, match=field):
        getattr(pop, operation + "_particle")(
            incoming, 2 if operation == "add" else 0, 3.0
        )
    np.testing.assert_array_equal(pop.spec_masses, before.spec_masses)
    np.testing.assert_array_equal(pop.num_concs, before.num_concs)
    assert pop.ids == before.ids


def test_equivalent_species_copies_and_name_only_objects_remain_compatible():
    pop = population()
    specs = tuple(replace(s) for s in reversed(pop.species))
    # Incidental runtime caches do not change physical species identity.
    specs[0].refractive_index_cache = object()
    pop.add_particle(Particle(specs, [0.0, 6e-18, 5e-18]), 2, 1.0)
    np.testing.assert_array_equal(pop.spec_masses[2], [5e-18, 6e-18, 0.0])
    names = (SimpleNamespace(name="A"), SimpleNamespace(name="B"))
    light = ParticlePopulation(names, np.array([[1.0, 2.0]]), np.array([1.0]), [0])
    light.add_particle(
        SimpleNamespace(species=tuple(reversed(names)), masses=[4.0, 3.0]), 1, 2.0
    )
    np.testing.assert_array_equal(light.spec_masses[1], [3.0, 4.0])


def test_empty_population_adopts_initial_order_and_detaches_incoming_masses():
    pop = ParticlePopulation((), np.empty((0, 0)), np.array([]), ())
    incoming = Particle(
        (get_species("BC", None), get_species("SO4", None)), [2e-18, 1e-18]
    )
    pop.add_particle(incoming, 7, 1.5)
    assert [s.name for s in pop.species] == ["BC", "SO4"]
    incoming.masses[:] = 0.0
    np.testing.assert_array_equal(pop.spec_masses, [[2e-18, 1e-18]])
    np.testing.assert_array_equal(pop.num_concs, [1.5])


def test_independently_created_name_only_species_remain_compatible():
    names = (SimpleNamespace(name="A"), SimpleNamespace(name="B"))
    pop = ParticlePopulation(names, np.array([[1.0, 2.0]]), np.array([1.0]), [0])
    incoming = SimpleNamespace(
        species=(SimpleNamespace(name="B"), SimpleNamespace(name="A")),
        masses=[4.0, 3.0],
    )
    pop.add_particle(incoming, 1, 2.0)
    np.testing.assert_array_equal(pop.spec_masses[1], [3.0, 4.0])


@pytest.mark.parametrize("operation", ["add", "set"])
@pytest.mark.parametrize("problem", ["missing_metadata", "matrix_masses"])
def test_missing_metadata_and_matrix_masses_fail_before_mutation(operation, problem):
    pop = population()
    before = copy.deepcopy(pop)
    specs = list(pop.species)
    masses = np.array([1e-18, 2e-18, 0.0])
    if problem == "missing_metadata":
        specs[0] = SimpleNamespace(name=specs[0].name)
    else:
        masses = masses.reshape(1, -1)
    incoming = SimpleNamespace(species=specs, masses=masses)
    with pytest.raises(ValueError, match="density|1D"):
        getattr(pop, operation + "_particle")(
            incoming, 2 if operation == "add" else 0, 3.0
        )
    np.testing.assert_array_equal(pop.spec_masses, before.spec_masses)
    np.testing.assert_array_equal(pop.num_concs, before.num_concs)
    assert pop.ids == before.ids


@pytest.mark.parametrize("operation", ["add", "set"])
def test_identical_species_with_nan_property_are_compatible(operation):
    # NaN != NaN, so a NaN-valued property used to reject an identical copy.
    base = tuple(get_species(n, None) for n in ["SO4", "BC"])
    pop_species = (replace(base[0], surface_tension=float("nan")), base[1])
    incoming_species = (replace(base[0], surface_tension=float("nan")), replace(base[1]))
    pop = ParticlePopulation(
        pop_species, np.array([[1e-18, 2e-18]]), np.array([1.0]), [0]
    )
    incoming = Particle(incoming_species, [3e-18, 4e-18])

    if operation == "add":
        pop.add_particle(incoming, 1, 2.0)
        np.testing.assert_array_equal(pop.spec_masses[1], [3e-18, 4e-18])
    else:
        pop.set_particle(incoming, 0, 2.0)
        np.testing.assert_array_equal(pop.spec_masses[0], [3e-18, 4e-18])


def test_nan_property_still_differs_from_a_number():
    base = tuple(get_species(n, None) for n in ["SO4", "BC"])
    pop = ParticlePopulation(
        (replace(base[0], kappa=float("nan")), base[1]),
        np.array([[1e-18, 2e-18]]), np.array([1.0]), [0],
    )
    with pytest.raises(ValueError, match="kappa"):
        pop.add_particle(Particle(base, [3e-18, 4e-18]), 1, 2.0)
