# tests/unit/optics/factory/test_fractal.py

import numpy as np
import pytest

from part2pop.population.builder import build_population
from part2pop.optics.factory.fractal import FractalParticle, build


pytestmark = pytest.mark.importorskip("pyBCabs")


def _make_bc_rich_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.5, 0.4, 0.1]],
    }
    return build_population(cfg)


@pytest.fixture
def fractal_particle():
    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])
    cfg = {
        "wvl_grid": [400e-9, 600e-9],
        "rh_grid": [0.0, 0.7],
        "single_scatter_albedo": 0.95,
    }
    with pytest.warns(UserWarning, match="fractal particles is not yet implemented"):
        return FractalParticle(base_particle, cfg)


def test_fractal_particle_compute_optics(fractal_particle):
    fp = fractal_particle
    assert fp.Cext.shape == (len(fp.rh_grid), len(fp.wvl_grid))
    assert np.all(fp.Cext >= 0.0)
    assert np.all(np.isfinite(fp.Csca))
    assert fp.single_scatter_albedo == pytest.approx(0.95)
    fp.h2o_vols[:] = 0.0
    fp.shell_dry_vol = 0.0
    assert fp._shell_ri(0, 0) == complex(1.0, 0.0)


def test_fractal_particle_geometric_helpers(fractal_particle):
    fp = fractal_particle
    x_core, x_coated = fp.get_x(Npp=10.0, Vratio=1.2)
    assert isinstance(x_core, float)
    assert isinstance(x_coated, float)
    df_core, df_coated = fp.get_Df(Npp=25.0, Vratio=1.5)
    assert df_core > 0.0
    assert df_coated > 0.0
    sol = fp.meff_solver(0.2)
    assert np.isfinite(sol[0])
    assert np.isfinite(sol[1])
    phase = fp.get_phase_shift(20.0, df_core, fp.wvl_grid[0])
    assert phase >= 0.0


def test_fractal_build_wrapper():
    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])
    cfg = {"wvl_grid": [550e-9], "rh_grid": [0.0]}
    with pytest.warns(UserWarning, match="fractal particles is not yet implemented"):
        fp = build(base_particle, cfg)
    assert isinstance(fp, FractalParticle)
