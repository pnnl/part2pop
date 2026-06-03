import importlib
import numpy as np

import part2pop.analysis.population.factory.b_ext as b_ext_mod

def test_import_b_ext():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.population.factory.b_ext")


def test_b_ext_compute_and_as_dict(monkeypatch):
    captured = {}
    expected = np.array([[10.0, 20.0]])

    class _FakeOpticalPopulation:
        def get_optical_coeff(self, name, rh=None, wvl=None):
            captured["coeff_name"] = name
            return expected

    def _fake_build_optical_population(population, ocfg):
        captured["ocfg"] = ocfg
        return _FakeOpticalPopulation()

    monkeypatch.setattr(b_ext_mod, "build_optical_population", _fake_build_optical_population)

    cfg = {
        "morphology": "core-shell",
        "rh_grid": [0.4],
        "wvl_grid": [532e-9, 660e-9],
        "T": 290.0,
        "species_modifications": {"SO4": {"kappa": 0.6}},
    }
    var = b_ext_mod.build(cfg)
    out = var.compute(object())
    assert np.allclose(out, expected)
    assert captured["coeff_name"] == "b_ext"
    assert captured["ocfg"]["type"] == "core_shell"
    assert captured["ocfg"]["rh_grid"] == [0.4]
    assert captured["ocfg"]["wvl_grid"] == [532e-9, 660e-9]
    assert captured["ocfg"]["temp"] == 290.0
    assert captured["ocfg"]["species_modifications"] == {"SO4": {"kappa": 0.6}}

    out_dict = var.compute(object(), as_dict=True)
    assert set(out_dict) == {"rh_grid", "wvl_grid", "b_ext"}
    assert np.allclose(out_dict["rh_grid"], [0.4])
    assert np.allclose(out_dict["wvl_grid"], [532e-9, 660e-9])
    assert np.allclose(out_dict["b_ext"], expected)
