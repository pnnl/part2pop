"""Optics analyses must inherit a population's refractive-index overrides.

Under this branch's rule, an explicit ``species_modifications={}`` resets the
population's overrides to species defaults. The analysis defaults injected
exactly that ``{}`` into b_ext, b_scat, b_abs and the particle-level optics
variables, so every default optics analysis silently dropped the population's
optical overrides (SO4 n_550 = 1.9 was computed as the packaged 1.55 value).
"""

import numpy as np
import pytest

import part2pop.optics.factory.homogeneous as homogeneous
from part2pop import build_population
from part2pop.analysis.builder import build_variable
from part2pop.optics.builder import build_optical_population

OPTICS_CFG = {"morphology": "homogeneous", "rh_grid": [0.0], "wvl_grid": [550e-9]}


def _population():
    return build_population({
        "type": "monodisperse",
        "aero_spec_names": [["SO4"]],
        "N": [1e9],
        "D": [100e-9],
        "aero_spec_fracs": [[1.0]],
        "species_modifications": {"SO4": {"n_550": 1.9}},
    })


@pytest.fixture
def mie_calls(monkeypatch):
    """Record the refractive index handed to the Mie solver."""
    calls = []

    def fake_mie(m, wavelength_nm, diameter_nm, asDict=True, asCrossSection=False):
        calls.append(complex(m))
        return {"Qext": 1.0, "Qsca": 0.9, "Qabs": 0.1, "g": 0.5}

    monkeypatch.setattr(homogeneous, "MieQ", fake_mie)
    return calls


def _assert_used_real_index(mie_calls, expected):
    assert mie_calls, "the Mie solver was not called"
    for m in mie_calls:
        assert m.real == pytest.approx(expected)


@pytest.mark.parametrize("name", ["b_ext", "b_scat", "b_abs"])
def test_population_optics_variables_inherit_overrides(mie_calls, name):
    build_variable(name, var_cfg=dict(OPTICS_CFG)).compute(_population())
    _assert_used_real_index(mie_calls, 1.9)


@pytest.mark.parametrize("name", ["ext_crossect", "scat_crossect", "abs_crossect", "SSA"])
def test_particle_optics_variables_inherit_overrides(mie_calls, name):
    cfg = {"morphology": "homogeneous", "RH": 0.0, "wvl": 550e-9}
    build_variable(name, scope="particle", var_cfg=cfg).compute_all(_population())
    _assert_used_real_index(mie_calls, 1.9)


def test_explicit_override_in_an_analysis_still_wins(mie_calls):
    cfg = dict(OPTICS_CFG, species_modifications={"SO4": {"n_550": 1.7}})
    build_variable("b_ext", var_cfg=cfg).compute(_population())
    _assert_used_real_index(mie_calls, 1.7)


def test_explicit_none_inherits_like_an_omitted_mapping(mie_calls):
    population = _population()
    cfg = {"type": "homogeneous", "rh_grid": [0.0], "wvl_grid": [550e-9],
           "species_modifications": None}
    build_optical_population(population, cfg)
    _assert_used_real_index(mie_calls, 1.9)


def test_explicit_empty_mapping_still_resets_to_defaults(mie_calls):
    population = _population()
    cfg = {"type": "homogeneous", "rh_grid": [0.0], "wvl_grid": [550e-9],
           "species_modifications": {}}
    build_optical_population(population, cfg)
    so4 = next(s for s in population.species if s.name == "SO4")
    default_n = float(so4.refractive_index.real_ri_fun(550e-9))
    assert default_n != pytest.approx(1.9)
    _assert_used_real_index(mie_calls, default_n)


def test_cached_grid_does_not_alias_the_callers_array(mie_calls):
    population = _population()
    grid = np.array([550e-9])
    build_optical_population(population, {"type": "homogeneous", "rh_grid": [0.0], "wvl_grid": grid})
    so4 = next(s for s in population.species if s.name == "SO4")

    grid[0] = 600e-9  # editing the caller's array must not change the cache key

    assert np.asarray(so4.refractive_index.wvls)[0] == 550e-9
