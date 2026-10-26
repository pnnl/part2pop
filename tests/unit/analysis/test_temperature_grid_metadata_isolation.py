import numpy as np
import pytest

from part2pop.analysis.builder import build_variable
from part2pop.analysis.population.factory.T_grid import TemperatureGridVar, build


@pytest.mark.parametrize("factory", [build, lambda cfg: build_variable("T_grid", var_cfg=cfg)])
@pytest.mark.parametrize("first_units,second_units", [("C", "K"), ("K", "C")])
def test_temperature_labels_do_not_depend_on_later_variables(factory, first_units, second_units):
    grids = {"C": [-20., -10.], "K": [253.15, 263.15]}
    first = factory({"T_grid": grids[first_units], "T_units": first_units})
    second = factory({"T_grid": grids[second_units], "T_units": second_units})
    assert first.meta.units == first_units
    assert second.meta.units == second_units
    np.testing.assert_array_equal(first.compute(None), grids[first_units])
    np.testing.assert_array_equal(second.compute(None), grids[second_units])
    assert TemperatureGridVar.meta.units == "K"
