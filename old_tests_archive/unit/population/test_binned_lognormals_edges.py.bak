import numpy as np
import pytest
from PyParticle.population.builder import build_population


def _cfg(global_edges=False):
    base = {
        "type": "binned_lognormals",
        "N":   [1e9, 5e8],
        "GMD": [100e-9, 200e-9],
        "GSD": [1.6,    1.8],
        "N_bins": [20, 20],
        "aero_spec_names": [["SO4"], ["SO4"]],
        "aero_spec_fracs": [[1.0], [1.0]],
        "D_is_wet": False,
    }
    if global_edges:
        base["D_min"] = 50e-9
        base["D_max"] = 500e-9
    return base


@pytest.mark.parametrize("global_edges", [False, True])
def test_edges_respected(global_edges):
    pop = build_population(_cfg(global_edges))
    # smoke on sizes: non-empty population
    assert len(pop.ids) > 0
    # verify all particles diameters lie within the intended edges
    Ds = np.array([pop.get_particle(pid).get_Ddry() for pid in pop.ids])
    if global_edges:
        assert Ds.min() >= pytest.approx(50e-9, rel=0, abs=1e-20)
        assert Ds.max() <= pytest.approx(500e-9, rel=0, abs=1e-20)
    else:
        # with per-mode edges the min/max should bracket around each mode’s implied range
        assert Ds.min() > 0
        assert Ds.max() < 5e-6  # just a sanity cap
