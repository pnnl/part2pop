import numpy as np
import pytest

from part2pop.analysis.population.factory.dNdlnD import DNdlnDVar, build


def test_build_applies_normalize_and_wetsize_flags():
    var = build({"normalize": True, "wetsize": False})
    assert isinstance(var, DNdlnDVar)
    assert var.meta.units == ""
    assert "Dry" in var.meta.long_label


def test_compute_histogram_uses_provided_edges(simple_population):
    cfg = {
        "method": "hist",
        "diam_grid": np.array([1.0e-6, 2.0e-6]),
        "edges": np.array([0.5e-6, 1.5e-6, 2.5e-6]),
        "wetsize": True,
    }
    var = build(cfg)
    out = var.compute(simple_population, as_dict=True)

    assert np.allclose(out["dNdlnD"], np.array([100.0, 200.0]))
    assert np.allclose(out["edges"], cfg["edges"])


def test_compute_raises_on_unknown_method(simple_population):
    var = build({"method": "not-a-method"})
    with pytest.raises(ValueError):
        var.compute(simple_population)
