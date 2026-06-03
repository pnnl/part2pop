import importlib
import numpy as np

import part2pop.analysis.population.factory.b_scat as b_scat_mod

def test_import_b_scat():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.b_scat")


def test_b_scat_compute_and_as_dict_with_wvls_alias(monkeypatch):
    captured = {}
    expected = np.array([[0.1, 0.2, 0.3]])

    class _FakeOpticalPopulation:
        def get_optical_coeff(self, name, rh=None, wvl=None):
            captured["coeff_name"] = name
            return expected

    def _fake_build_optical_population(population, ocfg):
        captured["ocfg"] = ocfg
        return _FakeOpticalPopulation()

    monkeypatch.setattr(b_scat_mod, "build_optical_population", _fake_build_optical_population)

    cfg = {
        "morphology": "core-shell",
        "rh_grid": [0.1],
        "wvls": [450e-9, 550e-9, 650e-9],
        "T": 300.0,
        "species_modifications": {},
    }
    var = b_scat_mod.build(cfg)
    out = var.compute(object())
    assert np.allclose(out, expected)
    assert captured["coeff_name"] == "b_scat"
    assert captured["ocfg"]["type"] == "core_shell"
    assert captured["ocfg"]["rh_grid"] == [0.1]
    assert captured["ocfg"]["wvl_grid"] == [450e-9, 550e-9, 650e-9]
    assert captured["ocfg"]["temp"] == 300.0
    assert captured["ocfg"]["species_modifications"] == {}

    out_dict = var.compute(object(), as_dict=True)
    assert set(out_dict) == {"rh_grid", "wvl_grid", "b_scat"}
    assert np.allclose(out_dict["rh_grid"], [0.1])
    assert np.allclose(out_dict["wvl_grid"], [450e-9, 550e-9, 650e-9])
    assert np.allclose(out_dict["b_scat"], expected)
