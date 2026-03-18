import numpy as np
import pytest
import part2pop.analysis.population.factory.avg_Jhet as jhet_mod
from part2pop.population import build_population

def test_avg_Jhet_with_h2o():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    particle_pop._equilibrate_h2o(0.85, 298)
    var = jhet_mod.build({"T_grid": [250.0], "T_units": "K"})
    arr = var.compute(particle_pop)
    assert arr.all() > 0.0


def test_avg_Jhet_no_h2o():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    var = jhet_mod.build({"T_grid": [250.0], "T_units": "K", "RH": 0.0})
    arr = var.compute(particle_pop)
    assert arr.all() == 0.0

def test_avg_Jhet_T_units():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    particle_pop._equilibrate_h2o(0.85, 298)
    var = jhet_mod.build({"T_grid": [-20+273.15], "T_units": "K"})
    arr_K = var.compute(particle_pop) 
    var = jhet_mod.build({"T_grid": [-20], "T_units": "C"})
    arr_C = var.compute(particle_pop)
    assert arr_C==arr_K

def test_avg_Jhet_T_warnings():
    config = {"type": "monodisperse", "D": [100e-9], "N": [1.0], "aero_spec_names": [["OC"]], "aero_spec_fracs": [[1.0]]}
    particle_pop = build_population(config)
    var = jhet_mod.build({"T_grid": [-10.0], "T_units": "K"})
    with pytest.raises(ValueError):
        arr = var.compute(particle_pop)
    var = jhet_mod.build({"T_grid": [-10.0], "T_units": "X"})
    with pytest.raises(ValueError):
        arr = var.compute(particle_pop)

