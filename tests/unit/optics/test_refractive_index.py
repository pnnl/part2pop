# tests/unit/optics/test_refractive_index.py

import numpy as np

from part2pop import AerosolSpecies
from part2pop.optics.refractive_index import (
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
