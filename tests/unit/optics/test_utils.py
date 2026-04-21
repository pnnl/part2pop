# tests/unit/optics/test_utils.py

import numpy as np
import pytest

from part2pop.optics.utils import (
    OPTICS_TYPE_MAP,
    m_to_nm,
    get_cross_section_array_from_population,
)
from part2pop.population.builder import build_population
from part2pop.optics.builder import build_optical_population


class DummyPopulation:
    def __init__(self):
        self.Cabs = np.ones((2, 2, 2))
        self.Csca = np.full((2, 2, 2), 2.0)
        self.Cext = np.full((2, 2, 2), 3.0)
        self.Cabs_bc = np.zeros((2, 2, 2))
        self.Csca_bc = np.zeros((2, 2, 2))
        self.Cext_bc = np.zeros((2, 2, 2))


def _make_monodisperse_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["SO4", "BC", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.7, 0.2, 0.1]],
    }
    return build_population(cfg)


def _make_optical_population():
    pop = _make_monodisperse_population()
    cfg = {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]}
    return build_optical_population(pop, cfg)


def test_m_to_nm_basic():
    vals_m = np.array([1e-9, 1e-6, 1e-3])
    vals_nm = m_to_nm(vals_m)
    assert np.allclose(vals_nm, vals_m * 1e9)


def test_get_cross_section_array_total_ext():
    optical_pop = _make_optical_population()
    arr = get_cross_section_array_from_population(
        optical_pop, "total_ext", idx_rh=0, idx_wvl=0
    )
    assert arr.shape == (len(optical_pop.ids),)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)


def test_get_cross_section_array_invalid_type_raises():
    optical_pop = _make_optical_population()
    with pytest.raises(ValueError, match="Unknown optics_type"):
        get_cross_section_array_from_population(optical_pop, "not_a_real_type")


def test_get_cross_section_array_returns_full_when_no_indices():
    optical_pop = _make_optical_population()
    arr = get_cross_section_array_from_population(optical_pop, "total_ext")
    assert arr.shape == optical_pop.Cext.shape


def test_get_cross_section_array_with_dummy_population_and_slice():
    pop = DummyPopulation()
    arr = get_cross_section_array_from_population(pop, "total_abs")
    assert arr.shape == pop.Cabs.shape
    assert np.all(arr == 1.0)

    arr = get_cross_section_array_from_population(pop, "pure_bc_ext")
    assert np.all(arr == 0.0)

    arr = get_cross_section_array_from_population(pop, "total_scat", idx_rh=1, idx_wvl=0)
    assert arr.shape == (2,)
    assert np.all(arr == 2.0)


def test_m_to_nm_converts_scalar_and_array_inputs():
    assert m_to_nm(1e-6) == pytest.approx(1000.0)
    assert np.all(m_to_nm([1e-6, 2e-6]) == np.array([1000.0, 2000.0]))
