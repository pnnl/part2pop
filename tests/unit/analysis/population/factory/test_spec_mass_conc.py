import importlib
import numpy as np

import part2pop.analysis.population.factory.spec_mass_conc as spec_mass_conc_mod


class _Species:
    def __init__(self, name):
        self.name = name


class _PopulationWithMassConc:
    def __init__(self, mass_conc):
        self.species = [_Species(k) for k in mass_conc.keys()]
        self._mass_conc = dict(mass_conc)

    def get_species_mass_conc(self, spec_name):
        return self._mass_conc[spec_name]

def test_import_spec_mass_conc():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.spec_mass_conc")


def test_build_returns_variable_object():
    var = spec_mass_conc_mod.build({})
    assert var is not None
    assert hasattr(var, "compute")


def test_compute_uses_population_species_when_names_omitted():
    pop = _PopulationWithMassConc({"SO4": 1.25, "BC": 2.5})
    var = spec_mass_conc_mod.build({})
    out = var.compute(pop)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
    assert np.allclose(out, [1.25, 2.5])


def test_compute_single_species_string_and_as_dict():
    pop = _PopulationWithMassConc({"BC": 2.5})
    var = spec_mass_conc_mod.build({"species_names": "BC"})

    out = var.compute(pop)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)
    assert np.allclose(out, [2.5])

    out_dict = var.compute(pop, as_dict=True)
    assert set(out_dict) == {"species_names", "mass_conc"}
    assert out_dict["species_names"] == ["BC"]
    assert isinstance(out_dict["mass_conc"], np.ndarray)
    assert np.allclose(out_dict["mass_conc"], out)
