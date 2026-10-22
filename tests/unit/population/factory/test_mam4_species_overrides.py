"""MAM4 must retain explicit physical and optical species configuration."""
from copy import deepcopy

import netCDF4
import pytest

from part2pop.population import build_population
from part2pop.optics.builder import build_optical_population


def _configuration(tmp_path, timestep):
    (tmp_path / 'namelist').write_text(
        'numc1 = 1e6,\nmfso41 = 0.5,\nmfpom1 = 0.5,\n', encoding='utf-8')
    with netCDF4.Dataset(tmp_path / 'mam_output.nc', 'w') as data:
        data.createDimension('mode', 1)
        data.createDimension('time', 1)
        for name, value in {'num_aer': 1e6, 'so4_aer': 1e-9, 'soa_aer': 2e-9,
                            'dgn_a': 1e-7, 'dgn_awet': 1.2e-7}.items():
            data.createVariable(name, 'f8', ('mode', 'time'))[:] = value
    return {
        'type': 'mam4', 'mam4_dir': str(tmp_path), 'timestep': timestep,
        'GSD': [1.4], 'GMD_init': [1e-7], 'N_bins': 2,
        'p': 90000.0, 'T': 298.0, 'D_is_wet': False,
        'species_modifications': {'SO4': {'density': 1950.0, 'kappa': 0.25,
                                          'n_550': 1.9}},
    }


@pytest.mark.parametrize('timestep', [1, 2])
def test_mam4_applies_species_properties_without_mutating_config(tmp_path, timestep):
    config = _configuration(tmp_path, timestep)
    original = deepcopy(config)
    population = build_population(config)
    sulfate = next(s for s in population.species if s.name == 'SO4')
    assert sulfate.density == 1950.0
    assert sulfate.kappa == 0.25
    assert config == original


@pytest.mark.parametrize('timestep', [1, 2])
def test_mam4_records_overrides_for_inherited_optics(tmp_path, timestep):
    config = _configuration(tmp_path, timestep)
    population = build_population(config)
    assert population.species_modifications == config['species_modifications']
    build_optical_population(population, {'type': 'homogeneous', 'rh_grid': [0.0],
                                         'wvl_grid': [550e-9]})
    sulfate = next(s for s in population.species if s.name == 'SO4')
    assert sulfate.refractive_index.real_ri_fun(550e-9) == pytest.approx(1.9)
