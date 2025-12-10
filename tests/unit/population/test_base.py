# tests/unit/population/test_base.py

import numpy as np
import pytest
from numpy.exceptions import AxisError
from types import SimpleNamespace

from part2pop.population.base import ParticlePopulation
from part2pop.aerosol_particle import (
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
    sulfate = get_species("SO4")
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


def test_equilibrate_h2o_updates_mass():
    particle = _make_simple_particle()
    pop = _build_population_from_particle(particle)
    idx_h2o = pop.get_species_idx("H2O")
    assert pop.spec_masses[0, idx_h2o] == 0.0
    pop._equilibrate_h2o(0.8, 298.15)
    assert pop.spec_masses[0, idx_h2o] > 0.0


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
    with pytest.raises(AxisError):
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


def test_particle_spec_accessors_and_moles():
    particle = _make_simple_particle()
    original = float(particle.get_spec_mass("SO4"))
    particle.set_spec_mass("SO4", original * 2)
    assert np.isclose(float(particle.get_spec_mass("SO4")[0]), original * 2)
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
    spec_mass = particle.get_spec_mass("BC")
    particle.set_spec_mass("BC", float(spec_mass[0]) * 0.5)
    assert np.isclose(float(particle.get_spec_mass("BC")[0]), float(spec_mass[0]) * 0.5)


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


def test_population_mass_and_radius_stub(monkeypatch):
    pop = _make_stub_population(monkeypatch)
    # effective radius uses get_Dwet / 2 averaged by num_concs
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
