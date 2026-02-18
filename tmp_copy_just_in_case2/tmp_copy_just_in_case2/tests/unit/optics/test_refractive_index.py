# tests/unit/optics/test_refractive_index.py

import io
import numpy as np
import pytest

from part2pop import AerosolSpecies
from part2pop.optics.refractive_index import (
    _load_water_ri,
    _to_float,
    build_refractive_index,
    RefractiveIndex,
)


def test_build_refractive_index_for_so4():
    spec = AerosolSpecies("SO4")
    wvl_grid = np.array([400e-9, 550e-9, 700e-9])

    spec_out = build_refractive_index(spec, wvl_grid)

    assert hasattr(spec_out, "refractive_index")
    ri = spec_out.refractive_index
    assert isinstance(ri, RefractiveIndex)

    assert ri.wvls.shape == wvl_grid.shape
    assert np.all(np.isfinite(ri.real_ris))
    assert np.all(ri.real_ris > 1.0)
    assert np.all(ri.imag_ris >= 0.0)

    n_550 = ri.real_ri_fun(550e-9)
    k_550 = ri.imag_ri_fun(550e-9)
    assert np.isscalar(n_550)
    assert np.isscalar(k_550)


def test_to_float_handles_unicode_and_scientific():
    assert _to_float("1.2×10−3") == pytest.approx(1.2e-3)
    assert _to_float("1.2x10^-3") == pytest.approx(1.2e-3)
    assert _to_float("10^2") == pytest.approx(1e2)
    with pytest.raises(ValueError):
        _to_float(None)


def test_load_water_ri_parses_csv(monkeypatch):
    data = "\n".join(
        [
            "Wavelength,skip,skip,n,k",
            "400,0,0,1.33,0.01",
            "500,0,0,1.34,0.02",
        ]
    )
    monkeypatch.setattr(
        "part2pop.optics.refractive_index.open_dataset",
        lambda _: io.StringIO(data),
    )
    _load_water_ri.cache_clear()
    f_n, f_k = _load_water_ri()
    val_n = f_n(np.array([450e-9]))
    val_k = f_k(np.array([450e-9]))
    assert np.isfinite(val_n).all()
    assert np.isfinite(val_k).all()


def test_build_refractive_index_applies_modifications():
    spec = AerosolSpecies("BC")
    wvl_grid = np.array([550e-9])
    mod = {"n_550": 2.5, "alpha_n": 0.5}
    spec_out = build_refractive_index(spec, wvl_grid, modifications=mod)
    params = spec_out.refractive_index.RI_params
    assert params["n_550"] == pytest.approx(2.5)
    assert params["alpha_n"] == pytest.approx(0.5)
