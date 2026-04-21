# tests/unit/optics/test_base.py

import numpy as np
import pytest

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
