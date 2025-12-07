# tests/unit/optics/factory/test_fractal.py

import numpy as np
import pytest

from part2pop.population.builder import build_population
from part2pop.optics.factory.fractal import FractalParticle, build


def _make_bc_rich_population():
    cfg = {
        "type": "monodisperse",
        "aero_spec_names": [["BC", "SO4", "H2O"]],
        "N": [1.0e6],
        "D": [0.1e-6],
        "aero_spec_fracs": [[0.5, 0.4, 0.1]],
    }
    return build_population(cfg)


@pytest.mark.importorskip("pyBCabs")
def test_fractal_particle_compute_optics():
    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])

    cfg = {"wvl_grid": [550e-9], "rh_grid": [0.0]}
    fp = FractalParticle(base_particle, cfg)
    fp.compute_optics()

    assert np.isfinite(fp.Cext[0, 0])
    assert fp.Cext[0, 0] >= 0.0


@pytest.mark.importorskip("pyBCabs")
def test_fractal_build_wrapper():
    pop = _make_bc_rich_population()
    base_particle = pop.get_particle(pop.ids[0])
    fp = build(base_particle, {"wvl_grid": [550e-9], "rh_grid": [0.0]})
    assert isinstance(fp, FractalParticle)
