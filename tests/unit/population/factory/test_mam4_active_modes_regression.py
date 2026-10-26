import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
from part2pop.population.factory import mam4
from part2pop.utilities import power_moments_from_lognormal
from part2pop.constants import DENSITY_LIQUID_WATER, MOLAR_MASS_DRY_AIR, R


def _output(tmp_path, numbers):
    with netCDF4.Dataset(tmp_path / "mam_output.nc", "w") as ds:
        ds.createDimension("mode", 4)
        ds.createDimension("time", 1)
        values = {
            "num_aer": numbers,
            "so4_aer": [1e-9] * 4,
            "soa_aer": [2e-9] * 4,
            "dgn_a": [1e-7, 2e-7, 3e-7, 4e-7],
            "dgn_awet": [1.2e-7, 2.4e-7, 3.6e-7, 4.8e-7],
        }
        for name, data in values.items():
            ds.createVariable(name, "f8", ("mode", "time"))[:, 0] = data
    return {"mam4_dir": str(tmp_path), "timestep": 2,
            "GSD": [1.2, 1.4, 1.6, 1.8], "N_bins": [2, 3, 4, 5],
            "p": 90000.0, "T": 298.0}


@pytest.mark.parametrize("wet", [True, False])
@pytest.mark.parametrize("bins", [[2, 3, 4, 5], 3])
def test_middle_inactive_mode_preserves_mode_indices(tmp_path, monkeypatch, wet, bins):
    cfg = _output(tmp_path, [1e6, 0.0, 3e6, 4e6])
    cfg.update(D_is_wet=wet, N_bins=bins)
    captured = {}

    def capture(config):
        captured.update(config)
        return "built"

    monkeypatch.setattr(mam4, "build_binned_lognormals", capture)
    assert mam4.build(cfg) == "built"
    assert captured["GSD"] == [1.2, 1.6, 1.8]
    assert captured["N_bins"] == ([2, 4, 5] if isinstance(bins, list) else 3)
    expected_names = ["SO4", "OC", "H2O"] if wet else ["SO4", "OC"]
    assert captured["aero_spec_names"] == [expected_names] * 3
    np.testing.assert_allclose(captured["GMD"],
                               np.array([1e-7, 3e-7, 4e-7]) * (1.2 if wet else 1.0),
                               rtol=1e-12, atol=0.0)
    if wet:
        density = MOLAR_MASS_DRY_AIR * cfg["p"] / (R * cfg["T"])
        for row, n, dry_d, gsd in zip(captured["aero_spec_fracs"],
                                    np.array([1e6, 3e6, 4e6]) * density,
                                    [1e-7, 3e-7, 4e-7], [1.2, 1.6, 1.8]):
            water = DENSITY_LIQUID_WATER * np.pi / 6.0 * (
                power_moments_from_lognormal(3, n, dry_d * 1.2, gsd)
                - power_moments_from_lognormal(3, n, dry_d, gsd))
            expected = np.array([1e-9 * density, 2e-9 * density, water])
            np.testing.assert_allclose(row, expected / expected.sum(), rtol=1e-12, atol=0.0)


def test_real_builder_accepts_middle_inactive_mode(tmp_path):
    cfg = _output(tmp_path, [1e6, 0.0, 3e6, 4e6])
    pop = mam4.build(cfg)
    assert len(pop.ids) == 2 + 4 + 5
    expected = 8e6 * MOLAR_MASS_DRY_AIR * cfg["p"] / (R * cfg["T"])
    assert pop.get_Ntot() == pytest.approx(expected)


def test_all_inactive_modes_return_empty_population(tmp_path):
    cfg = _output(tmp_path, [0.0] * 4)
    pop = mam4.build(cfg)
    assert len(pop.ids) == 0
    assert pop.get_Ntot() == 0.0
    assert pop.spec_masses.shape == (0, len(pop.species))
    assert pop.num_concs.shape == (0,)
    assert pop.get_tot_mass() == 0.0
    assert pop.get_tot_dry_mass() == 0.0


@pytest.mark.parametrize("field,value", [("GSD", [1.2, 1.4, 1.6]),
                                         ("N_bins", [2, 3, 4])])
def test_mode_parameters_must_describe_original_modes(tmp_path, field, value):
    cfg = _output(tmp_path, [1e6, 0.0, 3e6, 4e6])
    cfg[field] = value
    with pytest.raises(ValueError, match=field + " must have one entry per original MAM mode"):
        mam4.build(cfg)


# --- Initial conditions (timestep=1, read from the namelist) ---------------

def _namelist(tmp_path, numbers, omit_fractions_for=()):
    lines = []
    for mode, number in enumerate(numbers, start=1):
        lines.append(f"numc{mode} = {number},")
        if mode not in omit_fractions_for:
            lines += [f"mfso4{mode} = 0.25,", f"mfpom{mode} = 0.75,"]
    (tmp_path / "namelist").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"mam4_dir": str(tmp_path), "timestep": 1, "D_is_wet": False,
            "GSD": [1.2, 1.4, 1.6, 1.8], "N_bins": [2, 3, 4, 5],
            "GMD_init": [1e-7, 2e-7, 3e-7, 4e-7], "p": 90000.0, "T": 298.0}


@pytest.mark.parametrize("bins", [[2, 3, 4, 5], 3])
def test_initial_inactive_mode_is_dropped_with_aligned_indices(tmp_path, monkeypatch, bins):
    cfg = _namelist(tmp_path, [1e6, 0.0, 3e6, 4e6])
    cfg["N_bins"] = bins
    captured = {}

    def capture(config):
        captured.update(config)
        return "built"

    monkeypatch.setattr(mam4, "build_binned_lognormals", capture)
    assert mam4.build(cfg) == "built"
    np.testing.assert_array_equal(captured["N"], [1e6, 3e6, 4e6])
    assert captured["GSD"] == [1.2, 1.6, 1.8]
    assert captured["GMD"] == [1e-7, 3e-7, 4e-7]
    assert captured["N_bins"] == ([2, 4, 5] if isinstance(bins, list) else 3)
    assert captured["aero_spec_names"] == [["SO4", "OC"]] * 3
    for row in captured["aero_spec_fracs"]:
        np.testing.assert_allclose(row, [0.25, 0.75], rtol=1e-12, atol=0.0)


def test_initial_real_builder_skips_inactive_mode(tmp_path):
    pop = mam4.build(_namelist(tmp_path, [1e6, 0.0, 3e6, 4e6]))
    assert len(pop.ids) == 2 + 4 + 5
    assert pop.get_Ntot() == pytest.approx(8e6)
    assert np.all(pop.num_concs > 0)


def test_initial_inactive_mode_without_fractions_builds(tmp_path):
    # An unused mode often has no mass fractions in the namelist at all; it
    # used to produce a 0/0 composition before reaching the lognormal builder.
    cfg = _namelist(tmp_path, [1e6, 0.0, 3e6, 4e6], omit_fractions_for=(2,))
    with np.errstate(invalid="raise", divide="raise"):
        pop = mam4.build(cfg)
    assert pop.get_Ntot() == pytest.approx(8e6)


def test_initial_all_inactive_modes_return_empty_population(tmp_path):
    pop = mam4.build(_namelist(tmp_path, [0.0] * 4))
    assert len(pop.ids) == 0
    assert pop.get_Ntot() == 0.0
    assert pop.spec_masses.shape == (0, len(pop.species))
    assert pop.num_concs.shape == (0,)
    assert pop.get_tot_mass() == 0.0
    assert pop.get_tot_dry_mass() == 0.0
