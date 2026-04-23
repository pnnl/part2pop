import numpy as np
import pytest

import part2pop.analysis.population.factory.INSA_distribution as INSA_mod
from part2pop.population import build_population
from part2pop.freezing import build_freezing_population


def test_dnd_hist_returns_density_and_as_dict():
    config = {
        "type": "monodisperse",
        "morphology": "homogeneous",
        "D": [1e-7],
        "N": [1.0],
        "aero_spec_names": [["OC"]],
        "aero_spec_fracs": [[1.0]]
    }
    pop = build_population(config)
    freezing_pop = build_freezing_population(pop, config)
    
    var = INSA_mod.build({"method": "hist", "N_bins": 20})
    dens = var.compute(freezing_pop)
    assert dens.shape[0] == 20
    
    var = INSA_mod.build({"method": "hist", "N_bins": 20, "insa_grid": np.logspace(-10, -6, 10)})
    as_dict = var.compute(pop, as_dict=True)
    assert set(as_dict) == {"INSA", "dNdlnINSA", "edges"}
    
    var = INSA_mod.build({"method": "hist", "N_bins": 20, "edges": np.logspace(-10, -6, 10)})
    as_dict = var.compute(pop, as_dict=True)
    assert set(as_dict) == {"INSA", "dNdlnINSA", "edges"}
    
    var = INSA_mod.build({"method": "kde", "N_bins": 20})
    as_dict = var.compute(pop, as_dict=True)
    assert set(as_dict) == {"INSA", "dNdlnINSA", "edges"}
    
    var = INSA_mod.build({"method": "kde", "N_bins": 20, "normalize": True})
    as_dict = var.compute(pop, as_dict=True)
    assert set(as_dict) == {"INSA", "dNdlnINSA", "edges"}
    
    with pytest.raises(ValueError):
        var = INSA_mod.build({"method": "unknown"})
        var.compute(pop)


'''
def test_dnd_uses_dry_diam_and_errors_on_bad_edges():
    pop = _StubPopulation([1e-7, 2e-7], [1e-8, 2e-8], [1.0, 1.0])
    var = dnd_mod.build({"method": "hist", "wetsize": False, "edges": [1e-9, 1e-8, 1e-7]})
    dens = var.compute(pop)
    assert dens.shape[0] == 2

    var_bad = dnd_mod.build({"method": "hist", "edges": [0.0, 1.0]})
    with pytest.raises(ValueError):
        var_bad.compute(pop)


def test_dnd_kde_bad_method():
    pop = _StubPopulation([1e-7], [1e-7], [1.0])

    # unknown method should raise
    bad = dnd_mod.build({"method": "unknown"})
    with pytest.raises(ValueError):
        bad.compute(pop)

class _ProvidedPopulation(_StubPopulation):
    def __init__(self, dwets, ddrys, num_concs):
        super().__init__(dwets, ddrys, num_concs)
        self.provided = {
            "D": np.array([1e-7, 2e-7]),
            "dNdlnD": np.array([1.0, 2.0]),
            "edges": np.array([5e-8, 1.5e-7, 2.5e-7]),
        }

    def get_provided_dNdlnD(self, cfg):
        return self.provided

def test_dnd_provided_method_respects_population_data():
    pop = _ProvidedPopulation([1e-7], [1e-7], [1.0])
    var = dnd_mod.build({"method": "provided"})
    out = var.compute(pop, as_dict=True)
    assert np.allclose(out["dNdlnD"], pop.provided["dNdlnD"])
    assert np.allclose(var.compute(pop), pop.provided["dNdlnD"])


def test_dnd_interp_conservatively_remaps():
    pop = _ProvidedPopulation([1e-7], [1e-7], [1.0])
    cfg = {
        "method": "interp",
        "edges": [1e-8, 1e-7, 1.1e-6],
    }
    var = dnd_mod.build(cfg)
    dens = var.compute(pop)
    assert dens.shape[0] == len(cfg["edges"]) - 1
'''