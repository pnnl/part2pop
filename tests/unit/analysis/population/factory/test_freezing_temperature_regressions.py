from types import SimpleNamespace
import numpy as np
import pytest
from part2pop.analysis.population.factory import frozen_frac, unfrozen_frac


@pytest.mark.parametrize("module", [frozen_frac, unfrozen_frac])
@pytest.mark.parametrize("celsius", [-20., 0.])
def test_celsius_is_converted_once(monkeypatch, module, celsius):
    calls = []
    class Population:
        num_concs = np.array([1.])
        species_modifications = {}
        def _equilibrate_h2o(self, rh, t):
            calls.append((rh, t))
    class FrozenPop:
        num_concs = np.array([1.])
        def get_freezing_probs(self, T, config, dt=1.):
            calls.append(('freezing', T))
            return np.array([0.])
    monkeypatch.setattr(module, 'build_freezing_population', lambda pop, cfg: FrozenPop())
    module.build({'T': celsius, 'T_units':'C', 'RH':0.85, 't_max':1.0, 'dt':1.0}).compute(Population())
    assert calls[0] == pytest.approx((0.85, celsius + 273.15))
    assert calls[1][1] == pytest.approx(celsius + 273.15)


@pytest.mark.parametrize("module", [frozen_frac, unfrozen_frac])
def test_missing_temperature_is_distinguished_from_zero_celsius(module):
    with pytest.raises(ValueError, match="Need to specify T"):
        module.build({'T':None}).compute(None)


@pytest.mark.parametrize("module", [frozen_frac, unfrozen_frac])
@pytest.mark.parametrize("rh", [85.0, 1.01, -0.1, float("nan")])
def test_out_of_range_rh_is_rejected_before_water_mutation(module, rh):
    # RH above 0.99 used to be clipped silently, so RH=85 (percent) ran at 0.99.
    class Population:
        num_concs = np.array([1.])
        species_modifications = {}
        def _equilibrate_h2o(self, rh, t):
            raise AssertionError("water must not be equilibrated for invalid RH")
    with pytest.raises(ValueError, match="RH must be a fraction between 0 and 1"):
        module.build({'T': 250.0, 'RH': rh, 't_max': 1.0, 'dt': 1.0}).compute(Population())


@pytest.mark.parametrize("module", [frozen_frac, unfrozen_frac])
@pytest.mark.parametrize("rh", [0.0, 1.0])
def test_boundary_rh_values_are_accepted(monkeypatch, module, rh):
    calls = []
    class Population:
        num_concs = np.array([1.])
        species_modifications = {}
        def _equilibrate_h2o(self, value, t):
            calls.append(value)
    class FrozenPop:
        num_concs = np.array([1.])
        def get_freezing_probs(self, T, config, dt=1.):
            return np.array([0.])
    monkeypatch.setattr(module, 'build_freezing_population', lambda pop, cfg: FrozenPop())
    module.build({'T': 250.0, 'RH': rh, 't_max': 1.0, 'dt': 1.0}).compute(Population())
    assert calls == [rh]
