# tests/unit/population/test_base.py

import copy
from dataclasses import replace

import numpy as np
import pytest
from types import SimpleNamespace

from part2pop.population.base import ParticlePopulation
from part2pop.aerosol_particle import (
    Particle,
    compute_Dwet,
    compute_Sc_funsixdeg,
    compute_mass_h2o,
    effective_density,
    make_particle,
    make_particle_from_masses,
)
from part2pop.species.registry import get_species


def _empty_population():
    """Create a truly empty ParticlePopulation with the right shapes."""
    # Start with a dummy species entry; will be overwritten on first add_particle
    dummy_species = ()
    spec_masses = np.zeros((0, 0))
    num_concs = np.zeros((0,))
    ids = []
    return ParticlePopulation(
        species=dummy_species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=ids,
    )


def _make_simple_particle():
    specdata_path = None
    sulfate = get_species("SO4", specdata_path)
    D_wet = 0.1e-6
    return make_particle(
        D=D_wet,
        aero_spec_names=[sulfate.name],
        aero_spec_frac=np.array([1.0]),
        species_modifications={},
        D_is_wet=True,
    )


def test_particle_population_add_and_get_particle():
    """
    Start from an empty ParticlePopulation and use add_particle via set_particle
    to populate it, then recover the particle and its number concentration.
    """
    p = _make_simple_particle()
    pop = _empty_population()

    # set_particle will call add_particle for new id
    pop.set_particle(particle=p, part_id=5, num_conc=1e8, suppress_warning=False)

    assert 5 in pop.ids
    idx = list(pop.ids).index(5)
    assert np.isclose(pop.num_concs[idx], 1e8)

    p_out = pop.get_particle(5)
    assert np.isclose(p_out.get_Dwet(), p.get_Dwet())
    assert np.isclose(p_out.get_mass_tot(), p.get_mass_tot())


def test_particle_population_set_particle_overwrites_existing():
    """
    When we call set_particle with an existing id, the masses and num_conc
    should be overwritten in place (not duplicated).
    """
    p1 = _make_simple_particle()
    pop = _empty_population()

    pop.set_particle(particle=p1, part_id=1, num_conc=1e8)
    assert len(pop.ids) == 1

    p2 = _make_simple_particle()
    # Change the number concentration
    pop.set_particle(particle=p2, part_id=1, num_conc=3e8)

    assert len(pop.ids) == 1  # still one entry
    idx = list(pop.ids).index(1)
    assert np.isclose(pop.num_concs[idx], 3e8)


def test_particle_population_total_number_concentration():
    """
    get_Ntot should sum num_concs over all stored particles.
    """
    p = _make_simple_particle()
    pop = _empty_population()

    pop.set_particle(particle=p, part_id=1, num_conc=1e8)
    pop.set_particle(particle=p, part_id=2, num_conc=2e8)

    N_tot = pop.get_Ntot()
    assert np.isclose(N_tot, 3e8)


def test_find_particle_returns_next_index_for_unknown():
    pop = _empty_population()
    particle = _make_simple_particle()
    pop.set_particle(particle, part_id=1, num_conc=1e8)
    assert pop.find_particle(2) == len(pop.ids)


def test_find_particle_with_duplicate_ids_hits_duplicate_branch():
    particle = _make_simple_particle()
    spec_masses = np.vstack([particle.masses, particle.masses])
    pop = ParticlePopulation(
        species=particle.species,
        spec_masses=spec_masses,
        num_concs=np.array([1.0, 2.0]),
        ids=[1, 1],
    )
    idx = pop.find_particle(1)
    assert np.array_equal(idx, np.array([0, 1]))


def test_get_particle_raises_for_missing_id():
    pop = _empty_population()
    with pytest.raises(ValueError):
        pop.get_particle(99)


def test_set_particle_warns_when_adding_new_id():
    pop = _empty_population()
    particle = _make_simple_particle()
    with pytest.warns(UserWarning, match="adding"):
        pop.set_particle(particle, part_id=5, num_conc=2e8, suppress_warning=False)
    assert 5 in pop.ids
    assert np.isclose(pop.num_concs[-1], 2e8)


def test_add_particle_appends_to_population():
    pop = _empty_population()
    p1 = _make_simple_particle()
    pop.add_particle(p1, part_id=1, num_conc=1e8)
    p2 = _make_simple_particle()
    pop.add_particle(p2, part_id=2, num_conc=2e8)
    assert len(pop.ids) == 2
    assert np.isclose(pop.num_concs[-1], 2e8)
    assert pop.spec_masses.shape[0] == 2


# Test/documentation gap: the long-double cases use SimpleNamespace to test
# population storage without Particle.__init__'s float64 conversion. They do
# not establish end-to-end precision through the public Particle constructor.
# A constructor precision policy and real-Particle coverage are separate work.


def _make_three_species_population():
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
    pop = _make_three_species_population()
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
    pop = _make_three_species_population()
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
    pop = _make_three_species_population()
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
    pop = _make_three_species_population()
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
    pop = _make_three_species_population()
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
    pop = _make_three_species_population()
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


def test_get_tot_dry_mass_without_h2o_equals_total():
    specdata_path = None
    dry_species = (get_species("BC", specdata_path), get_species("SO4", specdata_path))
    spec_masses = np.array([[1e-15, 2e-15]])
    num_concs = np.array([1e6])
    pop = ParticlePopulation(
        species=dry_species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=[1],
    )
    assert np.isclose(pop.get_tot_dry_mass(), pop.get_tot_mass())


def test_make_particle_requires_fraction_sum():
    with pytest.raises(ValueError):
        make_particle(1e-6, ["SO4"], [0.5])


def test_make_particle_always_includes_h2o():
    particle = make_particle(1e-6, ["SO4"], [1.0])
    names = [spec.name.upper() for spec in particle.species]
    assert "H2O" in names
    assert np.isclose(np.sum(particle.masses), particle.get_mass_tot())


def test_compute_dwet_skips_when_zero_conditions():
    d = 1e-6
    assert np.isclose(compute_Dwet(Ddry=d, kappa=0.0, RH=0.5, T=300.0), d)
    assert np.isclose(compute_Dwet(Ddry=d, kappa=0.2, RH=0.0, T=300.0), d)


def test_compute_mass_h2o_matches_volume_difference():
    Ddry = 1e-6
    Dwet = 1.1e-6
    rho = 997.0
    expected = np.pi / 6.0 * (Dwet ** 3 - Ddry ** 3) * rho
    assert np.isclose(compute_mass_h2o(Ddry, Dwet, rho_h2o=rho), expected)


def test_effective_density_computes_inverse_sum():
    specs = [
        SimpleNamespace(density=1000.0),
        SimpleNamespace(density=2000.0),
    ]
    fracs = [0.25, 0.75]
    expected = 1.0 / (fracs[0] / 1000.0 + fracs[1] / 2000.0)
    assert np.isclose(effective_density(fracs, specs), expected)


def _build_population_from_particle(particle):
    spec_masses = np.atleast_2d(particle.masses)
    return ParticlePopulation(
        species=particle.species,
        spec_masses=spec_masses.copy(),
        num_concs=np.array([1.0]),
        ids=[1],
    )


def test_population_species_idx_and_mass_conc():
    particle = make_particle(1e-6, ["BC"], [1.0])
    pop = _build_population_from_particle(particle)
    idx = pop.get_species_idx("BC")
    assert int(idx) == idx
    assert pop.get_mass_conc("BC") > 0.0


def test_get_species_idx_missing_returns_none():
    particle = _make_simple_particle()
    pop = _build_population_from_particle(particle)
    assert pop.get_species_idx("NOT_PRESENT") is None


def test_get_mass_conc_missing_species_raises():
    particle = _make_simple_particle()
    pop = _build_population_from_particle(particle)
    with pytest.raises(IndexError):
        pop.get_mass_conc("UNKNOWN")


def test_equilibrate_h2o_updates_mass():
    particle = _make_simple_particle()
    pop = _build_population_from_particle(particle)
    idx_h2o = pop.get_species_idx("H2O")
    assert pop.spec_masses[0, idx_h2o] == 0.0
    pop._equilibrate_h2o(0.8, 298.15)
    assert pop.spec_masses[0, idx_h2o] > 0.0


def test_equilibrate_h2o_handles_empty_population():
    pop = _empty_population()
    pop._equilibrate_h2o(0.5, 290.0)


def _build_multi_particle_population():
    particle = make_particle(1e-6, ["BC", "SO4"], [0.5, 0.5])
    spec_masses = np.vstack([particle.masses, particle.masses * 1.5])
    num_concs = np.array([1.0, 2.0])
    ids = [1, 2]
    return ParticlePopulation(
        species=particle.species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=ids,
    )


def test_reduce_mixing_state_part_res_is_noop():
    pop = _build_multi_particle_population()
    with pytest.raises(NotImplementedError, match="intentionally unsupported"):
        pop.reduce_mixing_state(mixing_state="part_res", RH=0.5, T=298.0)


def test_reduce_mixing_state_mam4_branch(monkeypatch):
    pop = _build_multi_particle_population()
    with pytest.raises(NotImplementedError, match="intentionally unsupported"):
        pop.reduce_mixing_state(mixing_state="MAM4sameDryMass", RH=0.5, T=298.0)


def test_reduce_mixing_state_mam5_branch(monkeypatch):
    pop = _build_multi_particle_population()
    with pytest.raises(NotImplementedError, match="intentionally unsupported"):
        pop.reduce_mixing_state(mixing_state="MAM5sameBC", RH=0.5, T=298.0)


def test_clone_detached_copies_arrays():
    particle = _make_simple_particle()
    pop = _build_population_from_particle(particle)
    clone = pop.clone_detached()
    pop.spec_masses[0, 0] += 1.0
    pop.num_concs[0] += 1.0
    assert not np.isclose(clone.spec_masses[0, 0], pop.spec_masses[0, 0])
    assert not np.isclose(clone.num_concs[0], pop.num_concs[0])


def test_get_num_dist_1d_rejects_unknown_methods():
    particle = _make_simple_particle()
    pop = _build_population_from_particle(particle)
    with pytest.raises(NotImplementedError):
        pop.get_num_dist_1d(method="unknown")


def test_tot_dry_mass_and_reduce_mixing_state():
    particle = make_particle(1e-6, ["BC", "SO4"], [0.5, 0.5])
    pop = _build_population_from_particle(particle)
    total_mass = pop.get_tot_mass()
    dry_mass = pop.get_tot_dry_mass()
    assert np.isclose(dry_mass, total_mass - pop.spec_masses[0, pop.get_species_idx("H2O")])
    spec_masses = np.vstack([particle.masses, particle.masses])
    multi = ParticlePopulation(
        species=particle.species,
        spec_masses=spec_masses.copy(),
        num_concs=np.array([1.0, 1.0]),
        ids=[1, 2],
    )
    with pytest.raises(NotImplementedError, match="intentionally unsupported"):
        multi.reduce_mixing_state(mixing_state="MAM4sameDryMass", RH=0.5, T=290.0)

def test_particle_equilibration_updates_h2o_mass():
    particle = _make_simple_particle()
    assert np.isclose(particle.get_mass_h2o(), 0.0)
    particle._equilibrate_h2o(RH=0.8, T=298.15)
    assert particle.get_mass_h2o() > 0.0


def test_particle_variable_helpers_return_expected_values():
    particle = _make_simple_particle()
    wet = particle.get_variable("wet_diameter", 0.5, 300.0)
    assert wet >= particle.get_Ddry()
    assert np.isclose(particle.get_variable("dry_diameter"), particle.get_Ddry())
    assert np.isclose(particle.get_variable("tkappa"), particle.get_tkappa())


def test_particle_density_and_surface_properties(monkeypatch):
    particle = make_particle(1e-6, ["BC", "SO4"], [0.5, 0.5])
    assert particle.get_rho_w() == 1000.0

    with pytest.warns(UserWarning):
        sigma = particle.get_surface_tension()
    assert np.isclose(sigma, 0.072)

    s_crit, dcrit = particle.get_critical_supersaturation(T=298.15, return_D_crit=True)
    assert s_crit >= 0.0
    assert dcrit > 0.0

    # ensure effective kappa getters return finite values
    assert particle.get_tkappa() >= 0.0
    assert particle.get_shell_tkappa() >= 0.0

    # effective density helper returns positive value
    assert particle.get_trho() > 0.0


def test_particle_spec_accessors_and_moles():
    particle = _make_simple_particle()
    def _get_mass(spec):
        return np.asarray(particle.get_spec_mass(spec)).ravel()[0]

    original = float(_get_mass("SO4"))
    particle.set_spec_mass("SO4", original * 2)
    assert np.isclose(float(_get_mass("SO4")), original * 2)
    assert particle.get_spec_moles("SO4") > 0.0
    assert particle.get_spec_vol("SO4") > 0.0
    assert particle.get_spec_rho("SO4") > 0.0


def test_particle_tkappa_shell_and_density():
    particle = make_particle(1e-6, ["BC", "SO4"], [0.5, 0.5])
    assert particle.get_tkappa() > 0.0
    assert particle.get_shell_tkappa() > 0.0
    assert particle.get_trho() > 0.0


def test_critical_supersaturation_branches_for_high_and_low_kappa():
    stiff = _make_simple_particle()
    s_high = stiff.get_critical_supersaturation(T=295.0)
    assert s_high > 0.0

    hydrophobic = make_particle(1e-6, ["OC"], [1.0])
    s_low, dcrit = hydrophobic.get_critical_supersaturation(T=280.0, return_D_crit=True)
    assert s_low >= 0.0
    assert dcrit > hydrophobic.get_Ddry()


def test_compute_dwet_returns_larger_diameter_for_hygroscopic_particle():
    dwet = compute_Dwet(1e-6, kappa=0.4, RH=0.8, T=298.15)
    assert dwet > 1e-6


def test_compute_sc_function_matches_formula():
    value = compute_Sc_funsixdeg(1e-6, 1.0, 0.2, 0.8e-6)
    c6 = 1.0
    c4 = -(3.0 * (0.8e-6**3) * 0.2 / 1.0)
    c3 = -(2.0 - 0.2) * (0.8e-6**3)
    c0 = (0.8e-6**6) * (1.0 - 0.2)
    expected = c6 * (1e-6**6) + c4 * (1e-6**4) + c3 * (1e-6**3) + c0
    assert np.isclose(value, expected)


def test_make_particle_from_masses_preserves_species_order():
    particle = make_particle_from_masses(["SO4", "BC"], [1e-15, 2e-15])
    names = [spec.name for spec in particle.species]
    assert names == ["SO4", "BC"]


def test_particle_index_and_volume_helpers():
    particle = make_particle(1e-6, ["BC", "SO4"], [0.7, 0.3])
    # Mass helpers
    assert particle.get_mass_dry() <= particle.get_mass_tot()
    assert particle.get_mass_tot() > 0.0

    # Index helpers
    assert particle.idx_core().size >= 1
    assert particle.idx_dry().size >= 1
    assert particle.idx_dry_shell().size >= 1
    assert particle.idx_h2o() >= 0

    # Volume helpers
    assert particle.get_vol_tot() >= particle.get_vol_dry()
    assert particle.get_vol_core() >= 0.0
    assert particle.get_vol_dry_shell() >= 0.0

    # Species property helpers return sensible values
    assert np.all(particle.get_spec_rhos() > 0.0)
    assert np.all(particle.get_spec_kappas() >= 0.0)
    assert np.all(particle.get_spec_MWs() > 0.0)
    assert np.all(particle.get_vks() >= 0.0)
    assert np.all(particle.get_moles() >= 0.0)

    # Mass / density helpers
    assert particle.get_rho_h2o() > 0.0
    assert particle.get_mass_h2o() >= 0.0

    # Access specific species components
    spec_mass = np.asarray(particle.get_spec_mass("BC")).ravel()[0]
    particle.set_spec_mass("BC", float(spec_mass) * 0.5)
    assert np.isclose(
        float(np.asarray(particle.get_spec_mass("BC")).ravel()[0]),
        float(spec_mass) * 0.5,
    )


# ---------------------------------------------------------------------------
# Extended tests using a stubbed Particle to isolate population logic
# ---------------------------------------------------------------------------


class _FakeParticle:
    def __init__(self, species, masses):
        self.species = species
        self.masses = np.array(masses, dtype=float)

    def get_Dwet(self, *args, **kwargs):
        # diameter in meters; here just proportional to total mass for simplicity
        return float(self.masses.sum())

    def idx_h2o(self):
        for i, spec in enumerate(self.species):
            if spec.name == "H2O":
                return i
        return -1

    def get_variable(self, varname, *args, **kwargs):
        # return sum of masses for any variable name
        return float(self.masses.sum())


def _make_stub_population(monkeypatch):
    # Two species including H2O
    species = (SimpleNamespace(name="H2O"), SimpleNamespace(name="SO4"))
    spec_masses = np.array([[1.0, 2.0]])
    num_concs = np.array([5.0])
    ids = [1]
    monkeypatch.setattr("part2pop.population.base.Particle", _FakeParticle)
    return ParticlePopulation(species, spec_masses, num_concs, ids)


def test_find_and_get_particle_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    assert pop.find_particle(1) == 0
    assert pop.find_particle(99) == len(pop.ids)

    particle = pop.get_particle(1)
    assert isinstance(particle, _FakeParticle)
    with pytest.raises(ValueError):
        pop.get_particle(99)


def test_set_and_add_particle_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    new_particle = _FakeParticle(pop.species, [3.0, 4.0])
    pop.set_particle(new_particle, part_id=1, num_conc=2.5)
    assert pop.num_concs[0] == 2.5
    assert pop.spec_masses[0, 0] == 3.0

    # Add a brand new particle id
    another = _FakeParticle(pop.species, [1.0, 1.0])
    pop.set_particle(another, part_id=2, num_conc=1.0)
    assert len(pop.ids) == 2
    assert pop.num_concs[-1] == 1.0


@pytest.mark.parametrize(
    "varname",
    ["critical_supersaturation", "s_c"],
)
def test_get_variable_forwards_critical_supersaturation_arguments(
    monkeypatch, varname
):
    particle = _make_simple_particle()
    calls = []

    def fake_get_critical_supersaturation(T, return_D_crit=False):
        calls.append((T, return_D_crit))
        if return_D_crit:
            return 0.5, 1.2e-6
        return 0.5

    monkeypatch.setattr(
        particle,
        "get_critical_supersaturation",
        fake_get_critical_supersaturation,
    )

    result = particle.get_variable(varname, 298.15, True)

    assert result == (0.5, 1.2e-6)
    assert calls == [(298.15, True)]


@pytest.mark.parametrize(
    "varname",
    ["critical_supersaturation", "s_c"],
)
def test_get_variable_critical_supersaturation_matches_direct_call(
    monkeypatch, varname
):
    particle = _make_simple_particle()

    # Avoid the unrelated default-surface-tension warning in this test.
    monkeypatch.setattr(
        particle,
        "get_surface_tension",
        lambda: 0.072,
    )

    expected = particle.get_critical_supersaturation(298.15)
    actual = particle.get_variable(varname, 298.15)

    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "varname",
    ["critical_supersaturation", "s_c"],
)
def test_get_variable_critical_supersaturation_can_return_critical_diameter(
    monkeypatch, varname
):
    particle = _make_simple_particle()

    monkeypatch.setattr(
        particle,
        "get_surface_tension",
        lambda: 0.072,
    )

    s_crit, d_crit = particle.get_variable(
        varname,
        298.15,
        True,
    )

    assert s_crit >= 0.0
    assert d_crit > particle.get_Ddry()

def test_population_mass_and_radius_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    with pytest.warns(DeprecationWarning, match="effective_radius"):
        eff_r = pop.get_effective_radius()
    expected = (pop.get_particle(1).get_Dwet() / 2.0)
    assert eff_r == expected

    assert pop.get_tot_mass() == np.sum(pop.num_concs * np.sum(pop.spec_masses, axis=1))
    assert pop.get_tot_dry_mass() == pop.get_tot_mass() - pop.num_concs[0] * pop.spec_masses[0, 0]
    assert pop.get_mass_conc("SO4") == pop.num_concs[0] * pop.spec_masses[0, 1]


def test_get_particle_var_and_hist_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    vals = pop.get_particle_var("any")
    assert vals.shape == (len(pop.ids),)

    hist, edges = pop.get_num_dist_1d(N_bins=2, density=False)[0:2]
    assert hist.shape == (2,)

    with pytest.raises(NotImplementedError):
        pop.get_num_dist_1d(method="kde")
