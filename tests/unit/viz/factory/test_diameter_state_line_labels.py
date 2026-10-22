import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.builder import build_variable
from part2pop.analysis.population.factory.diam_grid import DiamGridVar
from part2pop.viz.factory.state_line import StateLinePlotter


@pytest.mark.parametrize("method", ["hist", "kde"])
@pytest.mark.parametrize("wet", [None, False, True])
def test_diameter_plot_labels_the_size_used_by_its_distribution(method, wet):
    particle = make_particle(100e-9, ["SO4", "H2O"], [0.1, 0.9])
    assert particle.get_Dwet() > 2 * particle.get_Ddry()
    pop = ParticlePopulation(
        particle.species, np.array([particle.masses]), np.array([100.0]), [0]
    )
    cfg = {"method": method, "diam_grid": np.geomspace(1e-8, 1e-6, 9)}
    if wet is not None:
        cfg["wetsize"] = wet
    expected_size = "wet" if wet is None or wet else "dry"
    original_axis_label = DiamGridVar.meta.long_label
    plotter = StateLinePlotter({"varname": "dNdlnD", "var_cfg": cfg})
    prepared = plotter.prep(pop)
    expected = build_variable("dNdlnD", var_cfg=cfg).compute(pop, as_dict=True)
    np.testing.assert_array_equal(prepared["x"], expected["D"])
    np.testing.assert_array_equal(prepared["y"], expected["dNdlnD"])
    assert np.count_nonzero(prepared["y"]) > 0
    assert prepared["xlabel"] == f"{expected_size} diameter [m]"
    assert DiamGridVar.meta.long_label == original_axis_label
    fig, ax = plt.subplots()
    try:
        plotter.render(prepared, ax)
        assert ax.get_xlabel() == prepared["xlabel"]
    finally:
        plt.close(fig)
