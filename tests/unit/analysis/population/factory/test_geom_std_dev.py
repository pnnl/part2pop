import numpy as np

from part2pop.population import build_population
from part2pop.analysis.population.factory import geom_std_dev


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


def _expected(pop, wetsize=True):
    getter = "get_Dwet" if wetsize else "get_Ddry"
    diameters = np.asarray([getattr(pop.get_particle(pid), getter)() for pid in pop.ids])
    log_diameters = np.log(diameters)
    mean = np.sum(pop.num_concs * log_diameters) / np.sum(pop.num_concs)
    var = np.sum(pop.num_concs * (log_diameters - mean) ** 2) / np.sum(pop.num_concs)
    return float(np.exp(np.sqrt(var)))


def test_geom_std_dev_computes_weighted_geometric_standard_deviation():
    pop = _population()
    var = geom_std_dev.build({})

    assert np.isclose(var.compute(pop), _expected(pop))
    assert np.isclose(var.compute(pop, as_dict=True)["geom_std_dev"], _expected(pop))
    assert var.meta.long_label == "Wet geometric standard deviation"


def test_geom_std_dev_can_use_dry_diameter():
    pop = _population()
    var = geom_std_dev.build({"wetsize": False})

    assert np.isclose(var.compute(pop), _expected(pop, wetsize=False))
    assert var.meta.long_label == "Dry geometric standard deviation"
