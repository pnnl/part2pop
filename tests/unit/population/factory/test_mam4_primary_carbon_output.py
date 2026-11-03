"""Characterize the unsupported primary-carbon output of the MAM4 box driver.

Synthetic records mirror its SO4/SOA export convention; no native executable
is run. Source basis and support limits: docs/mam4-input-validation.md.
"""
import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
from part2pop.constants import MOLAR_MASS_DRY_AIR, R
from part2pop.population.factory import mam4
from .test_mam4_active_modes_regression import _namelist


def _primary_carbon_output(tmp_path, mode4_number):
    # POM/BC exist in native MAM4 mode 4, but neither is exported into these
    # two arrays. First and later output records retain zero SO4/SOA there.
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "w") as ds:
        ds.createDimension("mode", 4)
        ds.createDimension("time", 2)
        values = {
            "num_aer": [1e6, 2e6, 3e6, mode4_number],
            "so4_aer": [1e-9, 1e-9, 1e-9, 0.0],
            "soa_aer": [2e-9, 2e-9, 0.0, 0.0],
            "dgn_a": [1e-7, 2e-7, 3e-7, 4e-7],
            "dgn_awet": [1.2e-7, 2.4e-7, 3.6e-7, 4.8e-7],
        }
        for name, values_by_mode in values.items():
            ds.createVariable(name, "f8", ("mode", "time"))[:] = (
                np.asarray(values_by_mode)[:, None]
            )
    return {"mam4_dir": str(tmp_path), "GSD": [1.2, 1.4, 1.6, 1.8],
            "N_bins": [2, 3, 4, 5], "p": 90000.0, "T": 298.0}


@pytest.mark.parametrize("wet", [False, True])
@pytest.mark.parametrize("record", [0, 1])
def test_active_primary_carbon_output_is_rejected(tmp_path, wet, record):
    cfg = _primary_carbon_output(tmp_path, mode4_number=4e6)
    cfg.update(timestep=record + 2, D_is_wet=wet)
    with pytest.raises(ValueError, match="MAM output composition.*positive total"):
        mam4.build(cfg)


@pytest.mark.parametrize("wet", [False, True])
@pytest.mark.parametrize("record", [0, 1])
def test_zero_number_primary_carbon_mode_can_be_omitted(tmp_path, wet, record):
    cfg = _primary_carbon_output(tmp_path, mode4_number=0.0)
    cfg.update(timestep=record + 2, D_is_wet=wet)
    pop = mam4.build(cfg)
    assert len(pop.ids) == 2 + 3 + 4
    density = MOLAR_MASS_DRY_AIR * cfg["p"] / (R * cfg["T"])
    assert pop.get_Ntot() == pytest.approx(6e6 * density)
    assert np.all(np.isfinite(pop.spec_masses))
    assert np.all(pop.spec_masses >= 0.0)


@pytest.mark.parametrize("wet", [False, True])
def test_pom_only_primary_carbon_initialization_can_succeed(tmp_path, wet):
    cfg = _namelist(tmp_path, [0.0, 0.0, 0.0, 4e6], omit_fractions_for=(4,))
    with (tmp_path / "namelist").open("a") as stream:
        # This deliberately does not contain mfso44, just as the driver has
        # no mode-4 sulfate. BC=0 avoids implying complete BC/POM support.
        stream.write("mfpom4 = 1.0,\nmfbc4 = 0.0,\n")
    cfg["D_is_wet"] = wet
    pop = mam4.build(cfg)
    assert len(pop.ids) == 5
    assert pop.get_Ntot() == pytest.approx(4e6)
    assert pop.get_mass_conc("OC") > 0.0
    assert pop.get_mass_conc("SO4") == 0.0
