import numpy as np
from part2pop.optics.refractive_index import ensure_refractive_index
from part2pop.species.registry import get_species


def test_override_change_rebuilds_ri_and_can_restore_defaults():
    spec = get_species('BC', None)
    ensure_refractive_index(spec, [550e-9], {'n_550': 2.1})
    assert spec.refractive_index.real_ri_fun(550e-9) == 2.1
    ensure_refractive_index(spec, [550e-9], {'n_550': 2.4})
    assert spec.refractive_index.real_ri_fun(550e-9) == 2.4
    ensure_refractive_index(spec, [550e-9], {})
    assert spec.refractive_index.real_ri_fun(550e-9) == 1.82


def test_new_grid_refreshes_sampled_ri_arrays():
    spec = get_species('SO4', None)
    ensure_refractive_index(spec, [550e-9])
    old = spec.refractive_index
    ensure_refractive_index(spec, [500e-9, 600e-9])
    assert spec.refractive_index is not old
    np.testing.assert_array_equal(spec.refractive_index.wvls, [500e-9, 600e-9])
