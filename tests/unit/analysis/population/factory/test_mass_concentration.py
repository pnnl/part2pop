import numpy as np

from part2pop.population import build_population
from part2pop.analysis.population.factory import mass_concentration


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


def test_mass_concentration_returns_total_mass():
    pop = _population()
    var = mass_concentration.build({})

    assert np.isclose(var.compute(pop), pop.get_tot_mass())
    assert var.compute(pop, as_dict=True) == {"mass_concentration": pop.get_tot_mass()}


def test_mass_concentration_can_return_dry_mass_and_updates_label():
    pop = _population()
    var = mass_concentration.build({"dry": True})

    assert np.isclose(var.compute(pop), pop.get_tot_dry_mass())
    assert var.meta.long_label == "Dry aerosol mass concentration"
