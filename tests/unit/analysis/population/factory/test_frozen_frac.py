import numpy as np
import pytest
import part2pop.analysis.population.factory.frozen_frac as ff_mod
from part2pop.population import build_population

def test_frozen_frac_with_h2o():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    particle_pop._equilibrate_h2o(0.85, 298)
    var = ff_mod.build({"T": 253.0, "T_units": "K"})
    arr = var.compute(particle_pop)
    assert arr[1:].all()>0.0

def test_frozen_frac_no_h2o():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    var = ff_mod.build({"T": 253.0, "T_units": "K"})
    arr = var.compute(particle_pop)
    assert arr.all()==0.0

def test_frozen_frac_species_modifications():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    var = ff_mod.build({"T": 253.0, "T_units": "K", "RH": 0.85, "species_modifications": {"OC": {"m_log10Jhet": 0.0, "b_log10Jhet": 1.0}}})
    arr_1 = var.compute(particle_pop)
    var = ff_mod.build({"T": 253.0, "T_units": "K", "RH": 0.85, "species_modifications": {"OC": {"m_log10Jhet": 10.0, "b_log10Jhet": 4.0}}})
    arr_2 = var.compute(particle_pop)
    assert arr_1[-1] != arr_2[-1]

def test_frozen_frac_T_errors():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    with pytest.raises(ValueError):
        var = ff_mod.build({"T": -10.0, "T_units": "K"})
        arr = var.compute(particle_pop)
    with pytest.raises(ValueError):
        var = ff_mod.build({"T": -10.0, "T_units": "X"})
        arr = var.compute(particle_pop)

def test_frozen_frac_T_units():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    var = ff_mod.build({"T": 253.0, "T_units": "K", "RH": 0.85})
    arr_1 = var.compute(particle_pop)
    var = ff_mod.build({"T": 253.0-273.15, "T_units": "C", "RH": 0.85})
    arr_2 = var.compute(particle_pop)
    assert np.isclose(arr_1[-1], arr_2[-1])