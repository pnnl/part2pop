# tests/unit/optics/test_base.py

import numpy as np
import pytest

from part2pop.aerosol_particle import make_particle
from part2pop.optics.base import OpticalParticle, OpticalPopulation
from part2pop.population.builder import build_population
from part2pop.population.base import ParticlePopulation


def _make_monodisperse_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "BC", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.7, 0.2, 0.1]],
    }
    pop = build_population(cfg)
    assert isinstance(pop, ParticlePopulation)
    return pop


class DummyOpticalParticle(OpticalParticle):
    """Minimal concrete implementation to exercise base class logic."""

    def compute_optics(self):
        nR = len(self.rh_grid)
        nW = len(self.wvl_grid)
        for rr in range(nR):
            for ww in range(nW):
                val = float(rr + ww + 1)
                self.Cext[rr, ww] = val
                self.Csca[rr, ww] = val / 2.0
                self.Cabs[rr, ww] = val / 4.0
                self.g[rr, ww] = 0.5


def test_optical_population_aggregates_particle_coeffs():
    pop = _make_monodisperse_population()
    rh_grid = [0.0, 0.5]
    wvl_grid = [400e-9, 550e-9]

    optical_pop = OpticalPopulation(pop, rh_grid, wvl_grid)

    base_particle = pop.get_particle(pop.ids[0])

    # IMPORTANT: config must be a dict, matching OpticalParticle.__init__
    cfg = {"rh_grid": rh_grid, "wvl_grid": wvl_grid}
    opt_part = DummyOpticalParticle(base_particle, cfg)
    opt_part.compute_optics()

    # Attach to population
    optical_pop.add_optical_particle(opt_part, pop.ids[0])

    # Aggregated extinction should be finite and positive
    b_ext = optical_pop.get_optical_coeff("b_ext", rh=0, wvl=550e-9)
    assert np.isfinite(b_ext)
    assert b_ext > 0.0

    # Unknown key should raise
    with pytest.raises(ValueError):
        optical_pop.get_optical_coeff("not_a_real_optics_type", rh=0, wvl=0)


def test_optical_helpers_indexing_and_refractive_indices():
    pop = _make_monodisperse_population()
    rh_grid = [0.0, 0.5]
    wvl_grid = [400e-9, 550e-9]
    cfg = {"rh_grid": rh_grid, "wvl_grid": wvl_grid}
    base_particle = pop.get_particle(pop.ids[0])
    opt_part = DummyOpticalParticle(base_particle, cfg)
    opt_part.compute_optics()

    # Refractive indices available and callable
    ri = opt_part.get_refractive_indices()
    assert len(ri) == len(base_particle.species)

    # Per-particle getter with slices/indices
    assert opt_part.get_cross_section("ext", rh_idx=0).shape == (len(wvl_grid),)

    optical_pop = OpticalPopulation(pop, rh_grid, wvl_grid)
    optical_pop.add_optical_particle(opt_part, pop.ids[0])

    # Internal helper handles list indexing
    vals = optical_pop._safe_index_2d(np.array([[1, 2], [3, 4]]), [0, 1], [0, 1])
    assert vals.shape == (2, 2)

    # Missing RH/wavelength should raise
    with pytest.raises(ValueError):
        optical_pop.get_optical_coeff("b_ext", rh=0.25)


def _simple_population():
    particle = make_particle(1e-6, ["SO4"], [1.0])
    spec_masses = np.asarray([particle.masses])
    num_concs = np.asarray([1e6])
    return ParticlePopulation(
        species=particle.species,
        spec_masses=spec_masses,
        num_concs=num_concs,
        ids=[1],
    )


class _FakeOpticalParticle:
    def __init__(self, rh_grid, wvl_grid):
        shape = (len(rh_grid), len(wvl_grid))
        self.Cabs = np.ones(shape) * 1.0
        self.Csca = np.ones(shape) * 2.0
        self.Cext = np.ones(shape) * 3.0
        self.g = np.ones(shape) * 0.5
        self.rh_grid = rh_grid
        self.wvl_grid = wvl_grid

    def compute_optics(self):
        self.Cabs *= 1.0


def test_optical_population_select_indices_and_safe_indexing():
    base_pop = _simple_population()
    pop = OpticalPopulation(base_pop, rh_grid=[0.0, 0.5], wvl_grid=[400e-9, 700e-9])
    with pytest.raises(ValueError):
        pop._select_indices(rh=1.0, wvl=400e-9)
    rh_idx, wvl_idx = pop._select_indices(rh=0.5, wvl=None)
    assert isinstance(rh_idx, int)
    assert wvl_idx == slice(None)

    arr = np.arange(4).reshape(2, 2)
    assert np.array_equal(pop._safe_index_2d(arr, slice(None), slice(None)), arr)
    assert np.array_equal(pop._safe_index_2d(arr, [0], [1]), arr[[0]][:, [1]])


def test_optical_population_add_and_coefficients():
    base_pop = _simple_population()
    pop = OpticalPopulation(base_pop, rh_grid=[0.0, 0.5], wvl_grid=[400e-9, 700e-9])
    particle = _FakeOpticalParticle(pop.rh_grid, pop.wvl_grid)
    pop.add_optical_particle(particle, part_id=1)

    val = pop.get_optical_coeff("b_abs", rh=0.0, wvl=400e-9)
    assert val > 0.0
    assert np.allclose(pop.get_optical_coeff("g"), pop.g[0])

    with pytest.raises(ValueError):
        pop.get_optical_coeff("unknown")


def test_compute_effective_kappas_runs():
    base_pop = _simple_population()
    pop = OpticalPopulation(base_pop, rh_grid=[0.0], wvl_grid=[550e-9])
    pop.compute_effective_kappas()
    assert pop.tkappas.shape == (len(pop.ids),)
