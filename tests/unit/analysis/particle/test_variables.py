import numpy as np
import pytest

import part2pop.analysis.particle.factory.Dwet as Dwet
import part2pop.analysis.particle.factory.kappa as kappa
import part2pop.analysis.particle.factory.P_frz as P_frz


import numpy as np
import pytest

import part2pop.analysis.particle.factory.D_critical as D_critical
import part2pop.analysis.particle.factory.abs_crossect as abs_crossect
import part2pop.analysis.particle.factory.Ddry as Ddry
import part2pop.analysis.particle.factory.Dwet as Dwet
import part2pop.analysis.particle.factory.ext_crossect as ext_crossect
import part2pop.analysis.particle.factory.kappa as kappa
import part2pop.analysis.particle.factory.mass_dry as mass_dry
import part2pop.analysis.particle.factory.mass_tot as mass_tot
import part2pop.analysis.particle.factory.P_frz as P_frz
import part2pop.analysis.particle.factory.scat_crossect as scat_crossect
import part2pop.analysis.particle.factory.s_critical as s_critical
import part2pop.analysis.particle.factory.SSA as ssa_var


class _StubParticle:
    def __init__(self, d=1.0, k=0.3):
        self._d = d
        self._k = k

    def get_Dwet(self):
        return self._d

    def get_Ddry(self):
        return 0.8 * self._d

    def get_tkappa(self):
        return self._k

    def get_mass_dry(self):
        return 1.0e-18 * self._d

    def get_mass_tot(self):
        return 1.2e-18 * self._d

    def get_critical_supersaturation(self, T, return_D_crit=False):
        s_crit = 0.2 * self._d
        d_crit = 1.5 * self._d
        if return_D_crit:
            return s_crit, d_crit
        return s_crit


class _StubPopulation:
    def __init__(self, ids=(1, 2)):
        self.ids = ids
        self._particles = {pid: _StubParticle(d=pid, k=pid * 0.1) for pid in ids}
        self.species_modifications = {}

    def get_particle(self, part_id):
        return self._particles[part_id]


def test_dwet_variable_computes(monkeypatch):
    pop = _StubPopulation(ids=(1, 2))
    var = Dwet.build({})

    assert var.compute_one(pop, 1) == 1
    all_vals = var.compute_all(pop)
    assert np.allclose(all_vals, [1, 2])


def test_kappa_variable_computes(monkeypatch):
    pop = _StubPopulation(ids=(3, 4))
    var = kappa.build({})

    assert np.isclose(var.compute_one(pop, 3), 0.3)
    all_vals = var.compute_all(pop)
    assert np.allclose(all_vals, [0.3, 0.4])


def test_ddry_variable_computes():
    pop = _StubPopulation(ids=(2, 5))
    var = Ddry.build({})
    assert np.isclose(var.compute_one(pop, 2), 1.6)
    assert np.allclose(var.compute_all(pop), [1.6, 4.0])


def test_mass_variables_compute():
    pop = _StubPopulation(ids=(1, 3))
    v_dry = mass_dry.build({})
    v_tot = mass_tot.build({})

    assert np.allclose(v_dry.compute_all(pop), [1.0e-18, 3.0e-18])
    assert np.allclose(v_tot.compute_all(pop), [1.2e-18, 3.6e-18])


def test_critical_variables_compute_with_default_and_custom_T():
    pop = _StubPopulation(ids=(1, 4))

    s_var = s_critical.build({})
    d_var = D_critical.build({"T": 300.0})

    assert np.allclose(s_var.compute_all(pop), [0.2, 0.8])
    assert np.allclose(d_var.compute_all(pop), [1.5, 6.0])


def test_optical_cross_section_variables_compute_and_use_expected_cfg(monkeypatch):
    pop = _StubPopulation(ids=(7, 8))
    called = []

    class _StubOptPop:
        Cabs = np.array([[[1.0]], [[2.0]]])
        Csca = np.array([[[3.0]], [[4.0]]])
        Cext = np.array([[[5.0]], [[8.0]]])

    def _stub_build_optical_population(population, cfg):
        called.append(cfg)
        return _StubOptPop()

    monkeypatch.setattr(abs_crossect, "build_optical_population", _stub_build_optical_population)
    monkeypatch.setattr(scat_crossect, "build_optical_population", _stub_build_optical_population)
    monkeypatch.setattr(ext_crossect, "build_optical_population", _stub_build_optical_population)
    monkeypatch.setattr(ssa_var, "build_optical_population", _stub_build_optical_population)

    cfg = {"morphology": "core-shell", "RH": 0.6, "T": 295.0, "wvl": 532e-9}
    v_abs = abs_crossect.build(cfg)
    v_sca = scat_crossect.build(cfg)
    v_ext = ext_crossect.build(cfg)
    v_ssa = ssa_var.build(cfg)

    assert np.allclose(v_abs.compute_all(pop), [1.0, 2.0])
    assert np.allclose(v_sca.compute_all(pop), [3.0, 4.0])
    assert np.allclose(v_ext.compute_all(pop), [5.0, 8.0])
    assert np.allclose(v_ssa.compute_all(pop), [3.0 / 5.0, 4.0 / 8.0])

    # also exercise compute_one path
    assert np.isclose(v_abs.compute_one(pop, 8), 2.0)
    assert np.isclose(v_ssa.compute_one(pop, 7), 3.0 / 5.0)

    # Validate key configuration mapping into optics builder
    assert called, "Expected optics builder to be called"
    for ocfg in called:
        assert ocfg["type"] == "core_shell"
        assert ocfg["rh_grid"] == [0.6]
        assert ocfg["wvl_grid"] == [532e-9]
        assert np.isclose(ocfg["temp"], 295.0)


def test_ssa_returns_zero_when_cext_is_zero(monkeypatch):
    pop = _StubPopulation(ids=(1, 2))

    class _StubOptPop:
        Csca = np.array([[[2.0]], [[0.0]]])
        Cext = np.array([[[0.0]], [[0.0]]])

    monkeypatch.setattr(ssa_var, "build_optical_population", lambda population, cfg: _StubOptPop())
    var = ssa_var.build({})
    assert np.allclose(var.compute_all(pop), [0.0, 0.0])


def test_p_frz_requires_temperature_and_uses_builder(monkeypatch):
    pop = _StubPopulation(ids=(1,))

    # Missing T should raise
    var = P_frz.build({"T": None})
    with pytest.raises(ValueError):
        var.compute_all(pop)

    class _StubFreezePop:
        def get_freezing_probs(self):
            return np.array([[0.42]])

    monkeypatch.setattr(
        P_frz, "build_freezing_population", lambda population, cfg: _StubFreezePop()
    )

    var_with_T = P_frz.build({"T": 250.0})
    probs = var_with_T.compute_all(pop)
    assert np.allclose(probs, 0.42)
