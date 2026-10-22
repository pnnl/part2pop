import numpy as np
import part2pop.analysis.population.factory.avg_Jhet as mod


def test_avg_jhet_reports_kelvin_when_input_is_celsius(monkeypatch):
    class Pop:
        species_modifications = {}
        def _equilibrate_h2o(self, rh, temp):
            pass
    class Freeze:
        def get_avg_Jhet(self, T, cfg):
            return T
    monkeypatch.setattr(mod, 'build_freezing_population', lambda pop, cfg: Freeze())
    result = mod.build({'T_grid':[-20., -10.], 'T_units':'C'}).compute(Pop(), as_dict=True)
    assert result['T_units'] == 'K'
    np.testing.assert_allclose(result['T_grid'], [253.15, 263.15])
