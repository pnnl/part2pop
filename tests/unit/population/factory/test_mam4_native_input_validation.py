"""The native adapter must reject invalid data before mode filtering."""
import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
from part2pop.population.factory import mam4
from .test_mam4_active_modes_regression import _namelist, _output


@pytest.mark.parametrize("factory", [_namelist, _output])
@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, -np.inf])
def test_invalid_original_numbers_are_not_inactive_modes(tmp_path, factory, bad):
    cfg = factory(tmp_path, [1e6, bad, 3e6, 4e6])
    with pytest.raises(ValueError, match="MAM number concentrations"):
        mam4.build(cfg)


def test_masked_original_number_is_rejected(tmp_path):
    cfg = _output(tmp_path, [1e6] * 4)
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds["num_aer"][1, 0] = np.ma.masked
    with pytest.raises(ValueError, match="missing or masked"):
        mam4.build(cfg)


@pytest.mark.parametrize("field", ["so4_aer", "soa_aer"])
@pytest.mark.parametrize("bad", [-1., np.nan, np.inf])
def test_invalid_active_component_fails_before_normalization(tmp_path, field, bad):
    cfg = _output(tmp_path, [1e6] * 4)
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds[field][1, 0] = bad
    with pytest.raises(ValueError, match="MAM (sulfate|organic) masses"):
        mam4.build(cfg)


@pytest.mark.parametrize("bad", [-.1, np.nan, np.inf])
def test_invalid_initial_fraction_is_rejected(tmp_path, bad):
    cfg = _namelist(tmp_path, [1e6] * 4)
    p = tmp_path / "namelist"
    p.write_text(p.read_text().replace("mfso42 = 0.25,", f"mfso42 = {bad},"))
    with pytest.raises(ValueError, match="MAM initial composition"):
        mam4.build(cfg)


@pytest.mark.parametrize("field", ["p", "T"])
@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, [298.], "invalid",
    np.complex128(298 + 1j), np.complex64(298 + 1j), complex(298, 1)])
def test_air_density_inputs_must_be_positive_finite_scalars(tmp_path, field, bad):
    cfg = _output(tmp_path, [1e6] * 4)
    cfg[field] = bad
    with pytest.raises(ValueError, match="MAM (pressure|temperature)"):
        mam4.build(cfg)


@pytest.mark.parametrize("factor", [.8, 0., np.nan, np.inf])
def test_invalid_wet_diameter_cannot_create_negative_water(tmp_path, factor):
    cfg = _output(tmp_path, [1e6] * 4)
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds["dgn_awet"][:] = factor * ds["dgn_a"][:]
    with pytest.raises(ValueError, match="MAM wet diameters"):
        mam4.build(cfg)


def test_dry_only_construction_does_not_use_wet_diameters(tmp_path):
    cfg = _output(tmp_path, [1e6] * 4)
    cfg["D_is_wet"] = False
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds["dgn_awet"][:] = np.nan
    pop = mam4.build(cfg)
    assert len(pop.ids) == 14
    assert np.all(np.isfinite(pop.spec_masses))
    assert np.all(pop.spec_masses >= 0)


def test_zero_active_dry_composition_is_rejected_before_normalization(tmp_path):
    cfg = _output(tmp_path, [1e6] * 4)
    cfg["D_is_wet"] = False
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds["so4_aer"][:] = 0.
        ds["soa_aer"][:] = 0.
    with pytest.raises(ValueError, match="MAM output composition.*positive total"):
        mam4.build(cfg)


def test_masked_active_mass_is_rejected(tmp_path):
    cfg = _output(tmp_path, [1e6] * 4)
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds["so4_aer"][1, 0] = np.ma.masked
    with pytest.raises(ValueError, match="missing or masked"):
        mam4.build(cfg)


def test_overflowed_converted_numbers_are_rejected(tmp_path):
    cfg = _output(tmp_path, [1e308] * 4)
    cfg["p"] = 1e7
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="Converted MAM number concentrations"):
            mam4.build(cfg)


def test_invalid_composition_of_zero_number_mode_is_not_used(tmp_path):
    cfg = _output(tmp_path, [1e6, 0., 3e6, 4e6])
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "r+") as ds:
        ds["so4_aer"][1, 0] = np.nan
        ds["dgn_awet"][1, 0] = np.nan
    pop = mam4.build(cfg)
    assert len(pop.ids) == 11
    assert np.all(np.isfinite(pop.spec_masses))
    assert np.all(pop.spec_masses >= 0)
