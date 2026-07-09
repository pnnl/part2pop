import numpy as np

from part2pop.population import build_population
from part2pop.analysis.population.factory import number_concentration


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


def test_number_concentration_returns_total_number():
    pop = _population()
    var = number_concentration.build({})

    assert np.isclose(var.compute(pop), pop.get_Ntot())
    assert var.compute(pop, as_dict=True) == {"number_concentration": pop.get_Ntot()}
