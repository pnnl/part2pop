import importlib
import numpy as np

import part2pop.analysis.population.factory.b_abs as b_abs_mod

def test_import_b_abs():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.b_abs")


def test_b_abs_compute_and_as_dict_with_wvls_alias(monkeypatch):
    captured = {}
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])

    class _FakeOpticalPopulation:
        def get_optical_coeff(self, name, rh=None, wvl=None):
            captured["coeff_name"] = name
            captured["rh"] = rh
            captured["wvl"] = wvl
            return expected

    def _fake_build_optical_population(population, ocfg):
        captured["population"] = population
        captured["ocfg"] = ocfg
        return _FakeOpticalPopulation()

    monkeypatch.setattr(b_abs_mod, "build_optical_population", _fake_build_optical_population)

    cfg = {
        "morphology": "core-shell",
        "rh_grid": [0.2, 0.8],
        "wvls": [500e-9, 700e-9],
        "T": 298.15,
        "species_modifications": {"BC": {"density": 1700.0}},
    }
    pop = object()
    var = b_abs_mod.build(cfg)

    out = var.compute(pop)
    assert np.allclose(out, expected)
    assert captured["coeff_name"] == "b_abs"
    assert captured["ocfg"]["type"] == "core_shell"
    assert captured["ocfg"]["rh_grid"] == [0.2, 0.8]
    assert captured["ocfg"]["wvl_grid"] == [500e-9, 700e-9]
    assert captured["ocfg"]["temp"] == 298.15
    assert captured["ocfg"]["species_modifications"] == {"BC": {"density": 1700.0}}

    out_dict = var.compute(pop, as_dict=True)
    assert set(out_dict) == {"rh_grid", "wvl_grid", "b_abs"}
    assert np.allclose(out_dict["rh_grid"], [0.2, 0.8])
    assert np.allclose(out_dict["wvl_grid"], [500e-9, 700e-9])
    assert np.allclose(out_dict["b_abs"], expected)
