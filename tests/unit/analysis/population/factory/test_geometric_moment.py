import numpy as np
import pytest

from part2pop.population import build_population
from part2pop.analysis.population.factory import geometric_moment


def _population():
    return build_population(
        {
            "type": "monodisperse",
            "aero_spec_names": [["SO4"], ["SO4"]],
            "N": [1.0e5, 2.5e5],
            "D": [80e-9, 150e-9],
            "aero_spec_fracs": [[1.0], [1.0]],
        }
    )


def _diameters(pop, wetsize=True):
    getter = "get_Dwet" if wetsize else "get_Ddry"
    return np.asarray([getattr(pop.get_particle(pid), getter)() for pid in pop.ids])


def test_geometric_moment_computes_weighted_diameter_power():
    pop = _population()
    var = geometric_moment.build({"moment_power": 2.0})
    expected = float(np.sum(pop.num_concs * _diameters(pop) ** 2.0))

    assert np.isclose(var.compute(pop), expected)
    out = var.compute(pop, as_dict=True)
    assert np.isclose(out["geometric_moment"], expected)
    assert out["moment_power"] == 2.0


def test_geometric_moment_can_normalize_and_use_alias_power():
    pop = _population()
    var = geometric_moment.build({"k": 1.0, "normalize": True, "wetsize": False})
    expected = float(np.sum(pop.num_concs * _diameters(pop, wetsize=False)) / np.sum(pop.num_concs))

    assert np.isclose(var.compute(pop), expected)
    assert var.meta.long_label == "Mean dry diameter moment"


def test_geometric_moment_rejects_nonfinite_power():
    with pytest.raises(ValueError, match="finite"):
        geometric_moment.build({"moment_power": np.nan})
