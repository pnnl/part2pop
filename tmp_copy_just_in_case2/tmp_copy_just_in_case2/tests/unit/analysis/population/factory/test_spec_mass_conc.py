import numpy as np

import part2pop.analysis.population.factory.spec_mass_conc as smc_mod


class _StubPopulation:
    def __init__(self):
        self.species = [type("S", (), {"name": "A"}), type("S", (), {"name": "B"})]
        self.calls = []

    def get_species_mass_conc(self, name):
        self.calls.append(name)
        return {"A": 1.0, "B": 2.0}.get(name, 0.0)


def test_spec_mass_conc_defaults_and_subset():
    pop = _StubPopulation()
    var = smc_mod.build({})

    all_vals = var.compute(pop)
    assert np.allclose(all_vals, [1.0, 2.0])

    subset = var.compute(pop, as_dict=True)
    assert subset["species_names"] == ["A", "B"]

    # explicit species subset
    var_subset = smc_mod.build({"species_names": "A"})
    vals = var_subset.compute(pop)
    assert np.allclose(vals, [1.0])
