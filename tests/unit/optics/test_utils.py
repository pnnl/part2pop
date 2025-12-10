# tests/unit/optics/test_utils.py

import numpy as np
import pytest

from part2pop.optics.base import OpticalPopulation
from part2pop.optics.utils import (
    OPTICS_TYPE_MAP,
    m_to_nm,
    get_cross_section_array_from_population,
)
from part2pop.population.builder import build_population
from part2pop.optics.builder import build_optical_population


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


def _make_optical_population_direct():
    base = _make_monodisperse_population()
    return OpticalPopulation(base, rh_grid=[0.0, 0.5], wvl_grid=[500e-9, 600e-9])


def test_optical_population_coeffs_and_indexing():
    optical_pop = _make_optical_population_direct()
    optical_pop.Cabs.fill(0.5)
    optical_pop.Csca.fill(1.0)
    optical_pop.Cext.fill(1.5)
    optical_pop.g.fill(0.25)

    # scalar lookups
    coeff = optical_pop.get_optical_coeff("b_ext", rh=0.0, wvl=500e-9)
    assert isinstance(coeff, float)
    assert coeff > 0.0

    # range selection uses _safe_index_2d
    # g calculation path
    g_val = optical_pop.get_optical_coeff("g", rh=0.0, wvl=500e-9)
    assert isinstance(g_val, float)
    assert np.isfinite(g_val)

    # compute_effective_kappas populates arrays
    optical_pop.compute_effective_kappas()
    assert optical_pop.tkappas.shape == (len(optical_pop.ids),)
    assert optical_pop.shell_tkappas.shape == (len(optical_pop.ids),)

    # _safe_index_2d handles iterable inputs
    arr = np.arange(4).reshape(2, 2)
    safe = optical_pop._safe_index_2d(arr, [0, 1], [0, 1])
    assert safe.shape == (2, 2)

    # invalid optics type raises
    with pytest.raises(ValueError):
        optical_pop.get_optical_coeff("unknown")

    # invalid grid points raise
    with pytest.raises(ValueError):
        optical_pop.get_optical_coeff("b_abs", rh=0.1, wvl=500e-9)
    with pytest.raises(ValueError):
        optical_pop.get_optical_coeff("b_abs", rh=0.0, wvl=700e-9)
