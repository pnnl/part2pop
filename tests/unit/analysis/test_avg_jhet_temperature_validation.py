import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.builder import build_variable


@pytest.mark.parametrize("grid,units", [
    ([-300.], "C"), ([-273.15], "C"), ([0.], "K"), ([-1.], "K"),
    ([250., np.inf], "K"), ([np.nan], "C"), ([], "K"),
    ([[250., 260.]], "K"), (250., "K"), (["bad"], "K"),
    (None, "K"), ([250.], "F"),
])
def test_invalid_grid_rejected_before_mutation(grid, units):
    particle = make_particle(1e-7, ["BC", "SO4"], [.5, .5])
    pop = ParticlePopulation(particle.species, np.array([particle.masses]), np.array([1.]), [0])
    before = pop.spec_masses.copy()
    var = build_variable("avg_Jhet", var_cfg={"T_grid": grid, "T_units": units, "RH": .85})
    with pytest.raises(ValueError):
        var.compute(pop)
    np.testing.assert_array_equal(pop.spec_masses, before)


@pytest.mark.parametrize("units,grid,expected", [
    ("C", np.array([-23.15, -13.15]), [250., 260.]),
    ("K", np.array([250., 260.]), [250., 260.]),
])
def test_valid_grid_converts_once_without_changing_caller(monkeypatch, units, grid, expected):
    from part2pop.analysis.population.factory import avg_Jhet
    before = grid.copy()
    class Population:
        species_modifications = {}
        def _equilibrate_h2o(self, rh, temperature):
            assert rh == .85
            assert temperature == pytest.approx(expected[0])
    class Freezing:
        def get_avg_Jhet(self, temperature, config):
            return temperature
    monkeypatch.setattr(avg_Jhet, "build_freezing_population", lambda *args: Freezing())
    var = avg_Jhet.build({"T_grid": grid, "T_units": units, "RH": .85})
    result = var.compute(Population(), as_dict=True)
    np.testing.assert_allclose(result["avg_Jhet"], expected)
    np.testing.assert_allclose(result["T_grid"], expected)
    assert result["T_units"] == "K"
    np.testing.assert_array_equal(grid, before)
