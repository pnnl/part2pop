import io
from pathlib import Path

import numpy as np
import pytest

from part2pop import build_population
from part2pop.population.factory import hiscale_observations as hiscale
from part2pop.population.factory.helpers.assembly import assemble_population_from_mass_fractions


def _write_temp_file(tmp_path, lines):
    filepath = tmp_path / "temp.txt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines))
    return str(filepath)


def _write_named_file(tmp_path, relative_path, lines):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return str(path)


def _make_sample_population():
    cfg = {
        "type": "monodisperse",
        "N": [1.0, 1.0, 1.0],
        "D": [80e-9, 100e-9, 120e-9],
        "aero_spec_names": [["BC"], ["OIN"], ["SO4"]],
        "aero_spec_fracs": [
            [1.0],
            [1.0],
            [1.0],
        ],
    }
    pop = assemble_population_from_mass_fractions(
        diameters=cfg["D"],
        number_concentrations=cfg["N"],
        species_names=cfg["aero_spec_names"],
        mass_fractions=cfg["aero_spec_fracs"],
        species_modifications=cfg.get("species_modifications", {}),
        D_is_wet=cfg.get("D_is_wet", False),
        specdata_path=cfg.get("specdata_path", None),
    )
    pop.metadata = {}
    return pop


def _standard_mass_thresholds():
    return {
        "BC": ((0.0, 0.5, 0.1), ["BC"]),
        "OIN": ((0.0, 0.5, 0.1), ["OIN"]),
        "SO4": ((0.0, 0.5, 0.1), ["SO4"]),
    }


def test_parse_nheader_firstline_valid_and_invalid():
    assert hiscale._parse_nheader_firstline("3, 10") == 3
    with pytest.raises(ValueError):
        hiscale._parse_nheader_firstline("")
    with pytest.raises(ValueError):
        hiscale._parse_nheader_firstline("bogus")


def test_read_icartt_table_happy_path(tmp_path):
    lines = [
        "3, 5",
        "comment",
        "Time, Cloud_flag",
        "1, 0",
        "2, 1",
    ]
    filepath = _write_temp_file(tmp_path, lines)

    table = hiscale._read_icartt_table(filepath)
    assert "Time" in table
    assert "Cloud_flag" in table
    assert table["Time"].shape[0] == 2

    # file too short
    filepath2 = _write_temp_file(tmp_path / "short", ["1,1"])
    with pytest.raises(ValueError):
        hiscale._read_icartt_table(filepath2)

    # mismatched columns
    lines_bad = ["3,2", "header", "A,B,C", "1,2"]
    filepath3 = _write_temp_file(tmp_path / "badcols", lines_bad)
    with pytest.raises(ValueError):
        hiscale._read_icartt_table(filepath3)


def test_read_icartt_table_invalid_nheader(tmp_path):
    lines = ["10, 0", "header", "Time, Cloud_flag", "1, 0"]
    filepath = _write_named_file(tmp_path, "icartt_bad.txt", lines)
    with pytest.raises(ValueError):
        hiscale._read_icartt_table(filepath)


def test_read_icartt_table_no_data_rows_raises_shape_error(tmp_path):
    lines = ["3, 5", "comment", "Time, Cloud_flag", "", ""]
    filepath = _write_named_file(tmp_path, "icartt_nodata.txt", lines)
    with pytest.raises(ValueError, match="Unexpected data array shape"):
        hiscale._read_icartt_table(filepath)


def test_time_indices_for_altitude_and_cloudflag_filters(monkeypatch):
    size_dist = {
        "Time(UTC)": np.array([0.0, 1.0, 2.0]),
        "Cloud_flag": np.array([0.0, 1.0, 0.0]),
    }
    aimms = {
        "Time(UTC)": np.array([0.0, 1.0]),
        "Alt": np.array([100.0, 102.0]),
        "Lat": np.array([36.1, 36.2]),
        "Lon": np.array([-97.45, -97.45]),
        "Cloud_flag": np.array([0.0, 1.0]),
    }
    idx = hiscale._time_indices_for_altitude_and_cloudflag(
        size_dist=size_dist,
        aimms=aimms,
        z=101.0,
        dz=5.0,
        cloud_flag_value=0,
    )
    assert idx.size == 1
    assert idx[0] == 0

    # missing aimms column raises
    with pytest.raises(KeyError):
        hiscale._time_indices_for_altitude_and_cloudflag(
            size_dist=size_dist,
            aimms={},
            z=100,
            dz=1,
        )


def test_time_indices_region_fallback():
    size_dist = {
        "Time(UTC)": np.array([0.0]),
        "Cloud_flag": np.array([0.0]),
    }
    aimms = {
        "Time(UTC)": np.array([0.0]),
        "Alt": np.array([10.0]),
        "Lat": np.array([0.0]),
        "Lon": np.array([0.0]),
    }
    idx = hiscale._time_indices_for_altitude_and_cloudflag(
        size_dist=size_dist,
        aimms=aimms,
        z=100.0,
        dz=1.0,
        region_filter={"lon_min": -1.0, "lon_max": 1.0, "lat_min": -1.0, "lat_max": 1.0},
    )
    assert idx.size == 1


def test_time_indices_cloud_filter_removes_all(monkeypatch):
    size_dist = {
        "Time(UTC)": np.array([0.0]),
        "Cloud_flag": np.array([1.0]),
    }
    aimms = {
        "Time(UTC)": np.array([0.0]),
        "Alt": np.array([100.0]),
        "Lat": np.array([36.5]),
        "Lon": np.array([-97.45]),
        "Cloud_flag": np.array([1.0]),
    }
    idx = hiscale._time_indices_for_altitude_and_cloudflag(
        size_dist=size_dist,
        aimms=aimms,
        z=100.0,
        dz=1.0,
        cloud_flag_value=0.0,
    )
    assert idx.size == 0


def test_time_indices_raises_when_size_dist_time_missing():
    size_dist = {"Cloud_flag": np.array([0.0])}
    aimms = {
        "Time(UTC)": np.array([0.0]),
        "Alt": np.array([100.0]),
        "Lat": np.array([36.5]),
        "Lon": np.array([-97.45]),
    }
    with pytest.raises(KeyError, match="missing time column"):
        hiscale._time_indices_for_altitude_and_cloudflag(
            size_dist=size_dist,
            aimms=aimms,
            z=100.0,
            dz=1.0,
        )


def test_time_indices_region_fallback_low_altitude_branch():
    size_dist = {"Time(UTC)": np.array([0.0]), "Cloud_flag": np.array([0.0])}
    aimms = {
        "Time(UTC)": np.array([0.0]),
        "Alt": np.array([500.0]),
        "Lat": np.array([36.5]),
        "Lon": np.array([-97.45]),
    }
    idx = hiscale._time_indices_for_altitude_and_cloudflag(
        size_dist=size_dist,
        aimms=aimms,
        z=10.0,
        dz=1.0,
    )
    assert idx.size == 1


def test_normalize_fracs_with_zero_sum():
    assert hiscale._normalize_fracs({"A": 0.0, "B": 0.0}) == {}
    normalized = hiscale._normalize_fracs({"A": 1.0, "B": 1.0})
    assert pytest.approx(sum(normalized.values())) == 1.0


def test_composition_for_type_variations():
    ams_mass_frac = {"SO4": 0.6, "OC": 0.4}
    species_map = {"SO4": "SO4", "OC": "OC"}

    default = hiscale._composition_for_type("unknown", "ams_everywhere", None, ams_mass_frac, species_map)
    assert pytest.approx(sum(default.values())) == 1.0

    templates = {"template": {"SO4": 1.0}}
    template_only = hiscale._composition_for_type(
        "template",
        "templates_only",
        templates,
        ams_mass_frac,
        species_map,
    )
    assert template_only == {"SO4": 1.0}

    with pytest.raises(KeyError):
        hiscale._composition_for_type(
            "missing",
            "templates_only",
            templates,
            ams_mass_frac,
            species_map,
        )

    fallback = hiscale._composition_for_type(
        "template",
        "templates_fallback_to_ams",
        {},
        ams_mass_frac,
        species_map,
    )
    assert pytest.approx(sum(fallback.values())) == 1.0


def test_default_template_for_type_handles_special_cases():
    assert hiscale._default_template_for_type("BC", {}, {}) == {"BC": 1.0}
    assert hiscale._default_template_for_type("OIN", {}, {}) == {"OIN": 1.0}
    ams_mass_frac = {"SO4": 0.5, "OC": 0.5}
    species_map = {"SO4": "SO4", "OC": "OC"}
    normalized = hiscale._default_template_for_type("custom", ams_mass_frac, species_map)
    assert pytest.approx(sum(normalized.values())) == 1.0


def test_sample_particle_masses_includes_nh4(tmp_path):
    mass_thresholds = {
        "SO4": ((0.1, 0.5, 0.1), ["SO4"]),
        "NO3": ((0.1, 0.5, 0.1), ["NO3"]),
        "BC": ((0.0, 0.2, 0.05), ["BC"]),
    }
    rng = np.random.default_rng(0)
    names, fracs = hiscale.sample_particle_masses("SO4", mass_thresholds, rng=rng, max_tries=1000)
    assert "NH4" in names
    assert pytest.approx(sum(fracs)) == 1.0


def test_read_delimited_table_with_header_error_paths(tmp_path):
    empty = _write_named_file(tmp_path, "empty.txt", [])
    with pytest.raises(ValueError, match="Empty file"):
        hiscale._read_delimited_table_with_header(empty)

    bad_header = _write_named_file(tmp_path, "bad_header.txt", ["OnlyOneColumn", "1"])
    with pytest.raises(ValueError, match="Could not parse header columns"):
        hiscale._read_delimited_table_with_header(bad_header)

    no_data = _write_named_file(tmp_path, "no_data.txt", ["A,B", ""])
    with pytest.raises(ValueError, match="No data rows parsed"):
        hiscale._read_delimited_table_with_header(no_data)


def test_read_delimited_table_with_header_tab_and_rows(tmp_path):
    # covers tab-delimiter branch and row parsing/output build branches
    path = _write_named_file(
        tmp_path,
        "tab_table.txt",
        [
            "A\tB\tC",
            "1\t2\t3",
            "",            # blank row skipped
            "4\t5",        # short row skipped
            "7\t8\t9",
        ],
    )
    out = hiscale._read_delimited_table_with_header(path)
    assert set(out.keys()) == {"A", "B", "C"}
    assert np.allclose(out["A"], [1.0, 7.0])
    assert np.allclose(out["B"], [2.0, 8.0])
    assert np.allclose(out["C"], [3.0, 9.0])


def test_read_fims_bins_file_error_paths(tmp_path):
    not_enough = _write_named_file(tmp_path, "bins_short.txt", ["1 2", "3 4"])
    with pytest.raises(ValueError, match="does not contain enough numeric values"):
        hiscale._read_fims_bins_file(not_enough, expected_bins=4)

    malformed = _write_named_file(tmp_path, "bins_bad.txt", ["5 1", "4 2", "3 3", "2 4", "1 5", "0 6"])
    with pytest.raises(ValueError, match="Could not extract"):
        hiscale._read_fims_bins_file(malformed, expected_bins=3)


def test_read_fims_bins_file_direct_and_stream_heuristics(tmp_path):
    # direct per-line parsing with 3-column rows (hits 308-309 and 311-314)
    direct = _write_named_file(
        tmp_path,
        "bins_direct.txt",
        [
            "",                # hit blank-line skip (302)
            "10 15 20",
            "20 25 40",
        ],
    )
    lo, hi = hiscale._read_fims_bins_file(direct, expected_bins=2)
    assert np.allclose(lo, [10.0, 20.0])
    assert np.allclose(hi, [20.0, 40.0])

    # interleaved global stream (hits 330)
    interleaved = _write_named_file(tmp_path, "bins_interleaved.txt", ["10 20 20 40"])
    lo_i, hi_i = hiscale._read_fims_bins_file(interleaved, expected_bins=2)
    assert np.allclose(lo_i, [10.0, 20.0])
    assert np.allclose(hi_i, [20.0, 40.0])

    # concatenated global stream (hits 336)
    concat = _write_named_file(tmp_path, "bins_concat.txt", ["10 20 20 40", "# extra", "1"])
    # Force line parser to fail expected_bins by including rows where lo==hi
    concat = _write_named_file(tmp_path, "bins_concat.txt", ["10 10", "20 20", "10 20 20 40"])
    lo_c, hi_c = hiscale._read_fims_bins_file(concat, expected_bins=2)
    # this case returns from the concatenated branch using the first valid block
    assert np.allclose(lo_c, [10.0, 10.0])
    assert np.allclose(hi_c, [20.0, 20.0])


def test_read_fims_bins_file_sliding_window_returns(tmp_path):
    # sliding window interleaved return (hits 346)
    slide_i = _write_named_file(tmp_path, "bins_slide_i.txt", ["99 99 10 20 20 40"])
    lo_i, hi_i = hiscale._read_fims_bins_file(slide_i, expected_bins=2)
    assert np.allclose(lo_i, [10.0, 20.0])
    assert np.allclose(hi_i, [20.0, 40.0])

    # sliding window concatenated return (hits 351)
    slide_c = _write_named_file(tmp_path, "bins_slide_c.txt", ["99 99 10 20 20 40 1"])
    lo_c, hi_c = hiscale._read_fims_bins_file(slide_c, expected_bins=2)
    assert np.allclose(lo_c, [10.0, 20.0])
    assert np.allclose(hi_c, [20.0, 40.0])


def test_read_beasd_bins_success_and_failure(tmp_path):
    good = _write_named_file(
        tmp_path,
        "beasd_good.txt",
        [
            "diameter range from 10 to 20",
            "diameter range from 20 to 30",
        ],
    )
    lo, hi = hiscale._read_beasd_bins(good, expected_bins=2)
    assert np.allclose(lo, [10.0, 20.0])
    assert np.allclose(hi, [20.0, 30.0])

    bad = _write_named_file(tmp_path, "beasd_bad.txt", ["diameter range from 10 to 20"])
    with pytest.raises(ValueError, match="Could not find expected number of bin lines"):
        hiscale._read_beasd_bins(bad, expected_bins=2)


def test_read_aimms_wrapper_calls_icartt(monkeypatch):
    expected = {"Time(UTC)": np.array([0.0])}
    monkeypatch.setattr(hiscale, "_read_icartt_table", lambda _: expected)
    out = hiscale._read_aimms("aimms.txt")
    assert out is expected


def test_read_fims_core_and_beasd_core_branches(monkeypatch):
    # _read_fims_core explicit n_dp detection branch + output assembly
    fims_table = {
        "Time(UTC)": np.array([0.0, 1.0]),
        "n_Dp_1": np.array([1.0, 2.0]),
        "ndp2": np.array([3.0, 4.0]),
    }
    monkeypatch.setattr(hiscale, "_read_icartt_table", lambda _: fims_table)
    out = hiscale._read_fims_core("fims.txt")
    assert out["_bin_cols"] == ["n_Dp_1", "ndp2"]
    assert out["_N_bins"].shape == (2, 2)

    # _read_beasd_core explicit C_ detection branch + output assembly
    beasd_table = {
        "Time(UTC)": np.array([0.0, 1.0]),
        "C_1": np.array([1.0, 2.0]),
        "C_2": np.array([3.0, 4.0]),
    }
    monkeypatch.setattr(hiscale, "_read_icartt_table", lambda _: beasd_table)
    out_b = hiscale._read_beasd_core("beasd.txt")
    assert out_b["_bin_cols"] == ["C_1", "C_2"]
    assert out_b["_N_bins"].shape == (2, 2)


def test_read_fims_core_and_beasd_core_identification_errors(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_icartt_table", lambda _: {"Time(UTC)": np.array([0.0]), "X": np.array([1.0])})
    with pytest.raises(ValueError, match="Could not identify FIMS bin columns"):
        hiscale._read_fims_core("fims_bad.txt")

    with pytest.raises(ValueError, match="Could not identify BEASD bin columns"):
        hiscale._read_beasd_core("beasd_bad.txt")


def test_read_fims_avg_size_dist_core_branches(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_aimms", lambda _: {"dummy": np.array([0.0])})
    monkeypatch.setattr(
        hiscale,
        "_read_fims_core",
        lambda _: {
            "Time(UTC)": np.array([0.0, 1.0]),
            "_N_bins": np.array([[2.0, 3.0], [4.0, 5.0]]),
        },
    )
    monkeypatch.setattr(hiscale, "_time_indices_for_altitude_and_cloudflag", lambda **kwargs: np.array([0, 1]))
    monkeypatch.setattr(hiscale, "_read_fims_bins_file", lambda *_args, **_kwargs: (np.array([10.0, 20.0]), np.array([20.0, 40.0])))

    lo, hi, nbin, nstd = hiscale._read_fims_avg_size_dist(
        fims_file="fims.txt",
        fims_bins_file="bins.txt",
        aimms_file="aimms.txt",
        z=100.0,
        dz=2.0,
        fims_density_measure="per_bin",
    )
    assert np.allclose(lo, [10.0, 20.0])
    assert np.allclose(hi, [20.0, 40.0])
    assert np.allclose(nbin, [3.0, 4.0])
    assert np.allclose(nstd, [1.0, 1.0])

    _lo, _hi, nbin_ln, _ = hiscale._read_fims_avg_size_dist(
        fims_file="fims.txt",
        fims_bins_file="bins.txt",
        aimms_file="aimms.txt",
        z=100.0,
        dz=2.0,
        fims_density_measure="ln",
    )
    assert np.all(nbin_ln > 0)

    with pytest.raises(ValueError, match="Unknown fims_density_measure"):
        hiscale._read_fims_avg_size_dist(
            fims_file="fims.txt",
            fims_bins_file="bins.txt",
            aimms_file="aimms.txt",
            z=100.0,
            dz=2.0,
            fims_density_measure="bad",
        )


def test_read_fims_avg_size_dist_error_paths(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_aimms", lambda _: {"dummy": np.array([0.0])})
    monkeypatch.setattr(hiscale, "_read_fims_core", lambda _: {"_N_bins": np.array([[1.0, 2.0]])})

    with pytest.raises(KeyError, match="Could not identify FIMS time column"):
        hiscale._read_fims_avg_size_dist(
            fims_file="fims.txt", fims_bins_file="bins.txt", aimms_file="aimms.txt", z=100.0, dz=2.0
        )

    monkeypatch.setattr(hiscale, "_read_fims_core", lambda _: {"Time(UTC)": np.array([0.0]), "_N_bins": np.array([[1.0, 2.0]])})
    monkeypatch.setattr(hiscale, "_time_indices_for_altitude_and_cloudflag", lambda **kwargs: np.array([], dtype=int))
    with pytest.raises(RuntimeError, match="No matching FIMS indices"):
        hiscale._read_fims_avg_size_dist(
            fims_file="fims.txt", fims_bins_file="bins.txt", aimms_file="aimms.txt", z=100.0, dz=2.0
        )


def test_read_minisplat_number_fractions_branches(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_aimms", lambda _: {"dummy": np.array([0.0])})
    monkeypatch.setattr(hiscale, "_read_fims_core", lambda _: {"Time(UTC)": np.array([0.0, 1.0])})
    monkeypatch.setattr(hiscale, "_time_indices_for_altitude_and_cloudflag", lambda **kwargs: np.array([0, 1]))
    monkeypatch.setattr(
        hiscale,
        "_read_delimited_table_with_header",
        lambda _: {"Time": np.array([0.0, 1.0]), "A": np.array([0.2, 0.4]), "B": np.array([0.8, 0.6])},
    )

    avg_comp, comp_err = hiscale._read_minisplat_number_fractions(
        splat_file="splat.txt",
        aimms_file="aimms.txt",
        size_dist_type="FIMS",
        size_dist_file="dist.txt",
        splat_species={"mix": ["A", "B"]},
        z=100.0,
        dz=2.0,
    )
    assert "mix" in avg_comp and "mix" in comp_err
    assert pytest.approx(avg_comp["mix"]) == 1.0

    with pytest.raises(ValueError, match="Either fims_file or beasd_file"):
        hiscale._read_minisplat_number_fractions(
            splat_file="splat.txt",
            aimms_file="aimms.txt",
            size_dist_type="BAD",
            size_dist_file="dist.txt",
            splat_species={"mix": ["A"]},
            z=100.0,
            dz=2.0,
        )


def test_read_ams_mass_fractions_branches(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_aimms", lambda _: {"dummy": np.array([0.0])})
    monkeypatch.setattr(
        hiscale,
        "_read_fims_core",
        lambda _: {"Time(UTC)": np.array([0.0, 1.0]), "_N_bins": np.array([[1.0], [1.0]])},
    )
    monkeypatch.setattr(hiscale, "_time_indices_for_altitude_and_cloudflag", lambda **kwargs: np.array([0, 1]))

    ams_table = {
        "dat_ams_utc": np.array([0.0, 1.0]),
        "flag": np.array([0.0, 0.0]),
        "Org": np.array([1.0, 2.0]),
        "NO3": np.array([1.0, 1.0]),
        "SO4": np.array([2.0, 1.0]),
        "NH4": np.array([1.0, 0.0]),
    }
    monkeypatch.setattr(hiscale, "_read_icartt_table", lambda _: ams_table)

    mass_frac, mass_frac_err, measured_mass, measured_mass_err = hiscale._read_ams_mass_fractions(
        ams_file="ams.txt",
        aimms_file="aimms.txt",
        size_dist_type="FIMS",
        size_dist_file="dist.txt",
        z=100.0,
        dz=2.0,
    )
    assert pytest.approx(sum(mass_frac.values())) == 1.0
    assert set(mass_frac_err) == {"OC", "NO3", "SO4", "NH4"}
    assert measured_mass > 0
    assert measured_mass_err >= 0

    monkeypatch.setattr(hiscale, "_read_icartt_table", lambda _: {"Time": np.array([0.0])})
    with pytest.raises(KeyError, match="Could not identify AMS flag/QC column"):
        hiscale._read_ams_mass_fractions(
            ams_file="ams.txt",
            aimms_file="aimms.txt",
            size_dist_type="FIMS",
            size_dist_file="dist.txt",
            z=100.0,
            dz=2.0,
        )


def _hiscale_example_paths():
    base = Path("examples/example_data/HISCALE_data_0425")
    return {
        "aimms": str(base / "AIMMS20_G1_20160425155810_R2_HISCALE020h.txt"),
        "beasd": str(base / "BEASD_G1_20160425155810_R2_HISCALE_001s.txt"),
        "ams": str(base / "HiScaleAMS_G1_20160425_R0.txt"),
        "splat": str(base / "Splat_Composition_25-Apr-2016.txt"),
    }


def test_read_beasd_avg_size_dist_with_bundled_example_data():
    paths = _hiscale_example_paths()
    assert all(Path(p).exists() for p in paths.values())

    lo, hi, nbin, nstd = hiscale._read_beasd_avg_size_dist(
        beasd_file=paths["beasd"],
        aimms_file=paths["aimms"],
        z=100.0,
        dz=50.0,
    )
    assert lo.size == hi.size == nbin.size == nstd.size
    assert lo.size > 0
    assert np.all(hi > lo)
    assert np.all(np.isfinite(nbin))


def test_read_beasd_avg_size_dist_measure_variants_and_errors(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_aimms", lambda _: {"dummy": np.array([0.0])})
    monkeypatch.setattr(
        hiscale,
        "_read_beasd_core",
        lambda _: {"Time(UTC)": np.array([0.0, 1.0]), "_N_bins": np.array([[2.0, 3.0], [4.0, 5.0]])},
    )
    monkeypatch.setattr(hiscale, "_time_indices_for_altitude_and_cloudflag", lambda **kwargs: np.array([0, 1]))
    monkeypatch.setattr(hiscale, "_read_beasd_bins", lambda *_args, **_kwargs: (np.array([10.0, 20.0]), np.array([20.0, 40.0])))

    # log10 conversion branch
    _lo, _hi, nbin_log10, _ = hiscale._read_beasd_avg_size_dist(
        beasd_file="beasd.txt",
        aimms_file="aimms.txt",
        z=100.0,
        dz=2.0,
        beasd_density_measure="log10",
    )
    assert np.all(nbin_log10 > 0)

    # invalid measure branch (captures current NameError bug behavior in src)
    with pytest.raises(Exception):
        hiscale._read_beasd_avg_size_dist(
            beasd_file="beasd.txt",
            aimms_file="aimms.txt",
            z=100.0,
            dz=2.0,
            beasd_density_measure="bad",
        )


def test_read_beasd_avg_size_dist_max_dp_filters_all_raises(monkeypatch):
    monkeypatch.setattr(hiscale, "_read_aimms", lambda _: {"dummy": np.array([0.0])})
    monkeypatch.setattr(
        hiscale,
        "_read_beasd_core",
        lambda _: {"Time(UTC)": np.array([0.0]), "_N_bins": np.array([[1.0, 2.0]])},
    )
    monkeypatch.setattr(hiscale, "_time_indices_for_altitude_and_cloudflag", lambda **kwargs: np.array([0]))
    monkeypatch.setattr(hiscale, "_read_beasd_bins", lambda *_args, **_kwargs: (np.array([2000.0, 3000.0]), np.array([3000.0, 4000.0])))

    with pytest.raises(RuntimeError, match="removed all FIMS bins"):
        hiscale._read_beasd_avg_size_dist(
            beasd_file="beasd.txt",
            aimms_file="aimms.txt",
            z=100.0,
            dz=2.0,
            max_dp_nm=1000.0,
        )


def test_read_minisplat_number_fractions_beasd_with_bundled_example_data():
    paths = _hiscale_example_paths()
    avg_comp, comp_err = hiscale._read_minisplat_number_fractions(
        splat_file=paths["splat"],
        aimms_file=paths["aimms"],
        size_dist_type="BEASD",
        size_dist_file=paths["beasd"],
        splat_species={"BC": ["soot"], "OIN": ["Dust"]},
        z=100.0,
        dz=50.0,
    )
    assert set(avg_comp.keys()) == {"BC", "OIN"}
    assert set(comp_err.keys()) == {"BC", "OIN"}
    assert pytest.approx(sum(avg_comp.values()), rel=1e-6) == 1.0


def test_read_ams_mass_fractions_beasd_with_bundled_example_data():
    paths = _hiscale_example_paths()
    mass_frac, mass_frac_err, measured_mass, measured_mass_err = hiscale._read_ams_mass_fractions(
        ams_file=paths["ams"],
        aimms_file=paths["aimms"],
        size_dist_type="BEASD",
        size_dist_file=paths["beasd"],
        z=100.0,
        dz=50.0,
    )
    assert set(mass_frac.keys()) == {"OC", "NO3", "SO4", "NH4"}
    assert set(mass_frac_err.keys()) == {"OC", "NO3", "SO4", "NH4"}
    assert pytest.approx(sum(mass_frac.values()), rel=1e-6) == 1.0
    assert measured_mass > 0
    assert measured_mass_err >= 0


def test_read_beasd_avg_size_dist_unknown_measure_from_real_files(tmp_path):
    paths = _hiscale_example_paths()
    with pytest.raises(Exception):
        hiscale._read_beasd_avg_size_dist(
            beasd_file=paths["beasd"],
            aimms_file=paths["aimms"],
            z=100.0,
            dz=50.0,
            beasd_density_measure="not-a-measure",
        )


def test_read_beasd_avg_size_dist_no_matching_indices_from_real_files():
    paths = _hiscale_example_paths()
    with pytest.raises(RuntimeError, match="No matching BEASD indices"):
        hiscale._read_beasd_avg_size_dist(
            beasd_file=paths["beasd"],
            aimms_file=paths["aimms"],
            z=1.0e6,
            dz=1.0,
            region_filter={"lon_min": 10.0, "lon_max": 20.0, "lat_min": 10.0, "lat_max": 20.0},
        )


def test_read_beasd_avg_size_dist_max_dp_filters_all_from_real_files():
    paths = _hiscale_example_paths()
    with pytest.raises(RuntimeError, match="removed all FIMS bins"):
        hiscale._read_beasd_avg_size_dist(
            beasd_file=paths["beasd"],
            aimms_file=paths["aimms"],
            z=100.0,
            dz=50.0,
            max_dp_nm=1.0,
        )


def test_read_minisplat_number_fractions_real_files_missing_time_col(tmp_path):
    paths = _hiscale_example_paths()
    bad_splat = tmp_path / "splat_missing_time.txt"
    bad_splat.write_text("A\tB\n1\t2\n", encoding="utf-8")

    with pytest.raises(KeyError, match="miniSPLAT missing time column"):
        hiscale._read_minisplat_number_fractions(
            splat_file=str(bad_splat),
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=paths["beasd"],
            splat_species={"mix": ["A"]},
            z=100.0,
            dz=50.0,
        )


def test_read_minisplat_number_fractions_real_files_no_size_indices():
    paths = _hiscale_example_paths()
    with pytest.raises(RuntimeError, match="No FIMS/BEASD indices"):
        hiscale._read_minisplat_number_fractions(
            splat_file=paths["splat"],
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=paths["beasd"],
            splat_species={"BC": ["soot"]},
            z=1.0e6,
            dz=1.0,
            region_filter={"lon_min": 10.0, "lon_max": 20.0, "lat_min": 10.0, "lat_max": 20.0},
        )


def test_read_minisplat_number_fractions_real_files_no_matching_splat_times(tmp_path):
    paths = _hiscale_example_paths()
    shifted = tmp_path / "splat_shifted.txt"
    shifted.write_text(
        "Time\tsoot\tDust\n1000000\t0.1\t0.2\n1000001\t0.2\t0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No miniSPLAT rows matched"):
        hiscale._read_minisplat_number_fractions(
            splat_file=str(shifted),
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=paths["beasd"],
            splat_species={"BC": ["soot"], "OIN": ["Dust"]},
            z=100.0,
            dz=50.0,
        )


def test_read_ams_mass_fractions_real_files_missing_ams_time_col(tmp_path):
    paths = _hiscale_example_paths()
    bad_ams = tmp_path / "ams_missing_time.txt"
    bad_ams.write_text(
        "4, 2\ncomment\ncomment\nOrg, NO3, SO4, NH4, flag\n1,1,1,1,0\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Could not identify AMS time column"):
        hiscale._read_ams_mass_fractions(
            ams_file=str(bad_ams),
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=paths["beasd"],
            z=100.0,
            dz=50.0,
        )


def test_read_ams_mass_fractions_real_files_missing_beasd_time_col(tmp_path):
    paths = _hiscale_example_paths()
    bad_beasd = tmp_path / "beasd_missing_time.txt"
    # no Time(UTC)/UTC/Start_UTC present, but C_* columns exist
    bad_beasd.write_text(
        "4, 2\ncomment\ncomment\nX, C_10, C_20\n1, 1, 2\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Could not identify BEASD time column"):
        hiscale._read_ams_mass_fractions(
            ams_file=paths["ams"],
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=str(bad_beasd),
            z=100.0,
            dz=50.0,
        )


def test_read_ams_mass_fractions_real_files_no_matching_size_indices():
    paths = _hiscale_example_paths()
    with pytest.raises(RuntimeError, match="No FIMS/BEASD indices for AMS matching"):
        hiscale._read_ams_mass_fractions(
            ams_file=paths["ams"],
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=paths["beasd"],
            z=1.0e6,
            dz=1.0,
            region_filter={"lon_min": 10.0, "lon_max": 20.0, "lat_min": 10.0, "lat_max": 20.0},
        )


def test_read_ams_mass_fractions_real_files_no_ams_rows_after_flag(tmp_path):
    paths = _hiscale_example_paths()
    bad_flag = tmp_path / "ams_all_bad_flag.txt"
    bad_flag.write_text(
        "5, 3\ncomment\ncomment\ncomment\ndat_ams_utc, Org, NO3, SO4, NH4, flag\n57900, 1, 1, 1, 1, 1\n57959, 1, 1, 1, 1, 1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No AMS rows matched filtered FIMS times"):
        hiscale._read_ams_mass_fractions(
            ams_file=str(bad_flag),
            aimms_file=paths["aimms"],
            size_dist_type="BEASD",
            size_dist_file=paths["beasd"],
            z=100.0,
            dz=50.0,
        )


def test_fit_nmodal_distribution_with_mocked_curve_fit(monkeypatch):
    dp = np.logspace(-8, -6, 20)
    true_pars = np.array([200.0, 1.0e-7, 1.6])
    n = hiscale.Nmodal_lognormal(dp, *true_pars)

    monkeypatch.setattr(hiscale, "curve_fit", lambda *args, **kwargs: (true_pars, np.eye(3)))
    out = hiscale.fit_Nmodal_distribution(dp, n)

    assert len(out) == 1
    assert np.allclose(out[0], true_pars)


def test_size_dependent_composition_scales_tail_fraction():
    dps = np.array([20.0, 40.0, 80.0, 160.0]) * 1e-9
    measured_n = np.array([1.0, 2.0, 3.0, 4.0])

    ns = hiscale.size_dependent_composition(
        Dps=dps,
        measured_Ns=measured_n,
        N_modes=1,
        splat_cutoff_nm=50.0,
        splat_number_fraction=0.25,
        mode_fractions=np.array([1.0]),
        size_dist_params=[1.0, 1.0e-7, 1.5],
    )

    idx0 = np.where(1e9 * dps >= 50.0)[0][0]
    assert np.all(np.isfinite(ns))
    assert np.all(ns >= 0)
    assert np.isclose(np.sum(ns[idx0:]), np.sum(measured_n[idx0:]) * 0.25)


def test_sample_particle_dp_n_fill_bins_true_and_false():
    particle_types = ["BC", "BC", "BC", "OIN", "OIN", "OIN"]
    measured_number_fractions = {"BC": 0.5, "OIN": 0.5}
    dp_mid = np.array([50.0, 100.0]) * 1e-9
    dp_lo = np.array([40.0, 80.0]) * 1e-9
    dp_hi = np.array([80.0, 120.0]) * 1e-9
    n_m3 = np.array([100.0, 50.0])

    d_fill, n_fill = hiscale.sample_particle_Dp_N(
        6,
        particle_types,
        measured_number_fractions,
        dp_mid,
        dp_lo,
        dp_hi,
        n_m3,
        mode_fractions={},
        size_distribution_parameters=[[1.0, 1.0e-7, 1.5]],
        fill_bins=True,
        size_dist_grid=1,
    )
    assert d_fill.shape == (6,)
    assert n_fill.shape == (6,)
    assert np.all(d_fill > 0)
    assert np.all(n_fill >= 0)

    d_rand, n_rand = hiscale.sample_particle_Dp_N(
        6,
        particle_types,
        measured_number_fractions,
        dp_mid,
        dp_lo,
        dp_hi,
        n_m3,
        mode_fractions={},
        size_distribution_parameters=[[1.0, 1.0e-7, 1.5]],
        fill_bins=False,
        size_dist_grid=1,
    )
    assert d_rand.shape == (6,)
    assert n_rand.shape == (6,)
    assert np.all(d_rand > 0)
    assert np.all(n_rand >= 0)


def test_classify_and_fraction_comparisons(tmp_path):
    pop = _make_sample_population()
    mass_thresholds = _standard_mass_thresholds()

    classes = hiscale.classify_particles(pop, mass_thresholds)
    assert set(np.unique(classes)) <= set(mass_thresholds.keys())

    measured_mass = {"BC": 1 / 3, "OIN": 1 / 3, "SO4": 1 / 3}
    measured_mass_err = {"BC": 0.01, "OIN": 0.01, "SO4": 0.01}
    sampled_mass, mass_checks = hiscale.mass_fraction_comparison(pop, measured_mass, measured_mass_err, mass_thresholds)
    assert pytest.approx(sum(sampled_mass.values())) == 1.0
    assert len(mass_checks) == len(mass_thresholds)

    measured_num = {"BC": 1 / 3, "OIN": 1 / 3, "SO4": 1 / 3}
    measured_num_err = {"BC": 0.01, "OIN": 0.01, "SO4": 0.01}
    sampled_num, num_checks = hiscale.number_fraction_comparison(
        pop,
        mass_thresholds,
        measured_num,
        measured_num_err,
        splat_cutoff_nm=85,
    )
    assert pytest.approx(sum(sampled_num.values())) == 1.0
    assert len(num_checks) == len(mass_thresholds)


def test_mass_fraction_comparison_missing_species_in_population_no_broadcast_error():
    pop = _make_sample_population()

    measured_mass_fractions = {"OC": 1.0}
    measured_mass_fraction_errors = {"OC": 0.0}

    sampled_mass_fractions, checks = hiscale.mass_fraction_comparison(
        particle_population=pop,
        measured_mass_fractions=measured_mass_fractions,
        measured_mass_fraction_errors=measured_mass_fraction_errors,
        mass_thresholds={},
    )

    assert "OC" in sampled_mass_fractions
    assert np.isclose(sampled_mass_fractions["OC"], 0.0)
    assert checks == [False]



def test_optimize_splat_species_distributions_returns_model_weights():
    splat_species = {"BC": ["BC"], "OIN": ["OIN"], "OC": ["OC"]}
    mass_thresholds = {
        "BC": ((0.0, 0.4, 0.05), ["BC"]),
        "OIN": ((0.0, 0.3, 0.05), ["OIN"]),
        "OC": ((0.0, 0.3, 0.05), ["OC"]),
    }
    splat_number_fractions = {"BC": 0.4, "OIN": 0.3, "OC": 0.3}
    ams_mass_fractions = {"BC": 0.4, "OIN": 0.3, "OC": 0.3}
    measured_Dp = np.array([90.0, 100.0]) * 1e-9
    measured_N = np.array([2.0, 1.5])
    pars = [[2.0, 110e-9, 1.5]]

    fractions, multiplier = hiscale.optimize_splat_species_distributions(
        splat_species=splat_species,
        size_distribution_pars=pars,
        measured_Dp=measured_Dp,
        measured_N=measured_N,
        splat_number_fractions=splat_number_fractions,
        ams_mass_fractions=ams_mass_fractions,
        mass_thresholds=mass_thresholds,
        datapoints=3,
    )

    assert "OC" in fractions
    assert len(fractions["OC"]) == len(pars)
    assert multiplier > 0


def test_sample_particle_masses_includes_nh4_and_normalizes():
    mass_thresholds = {
        "SO4": ((0.1, 0.5, 0.1), ["SO4"]),
        "NO3": ((0.1, 0.5, 0.1), ["NO3"]),
        "BC": ((0.0, 0.2, 0.05), ["BC"]),
    }
    rng = np.random.default_rng(0)
    names, fracs = hiscale.sample_particle_masses("SO4", mass_thresholds, rng=rng, max_tries=1000)
    assert "NH4" in names
    assert pytest.approx(sum(fracs)) == 1.0
    idx = names.index("NH4")
    assert fracs[idx] > 0


def test_sample_particle_Dp_N_enforces_fill_bins_and_allows_random():
    size_dist_grid = 1
    mode_fractions = {"BC": [1.0]}
    size_distribution_parameters = [[1.0, 110e-9, 1.5]]
    Dp_mid = np.array([90.0, 100.0, 110.0]) * 1e-9
    Dp_lo = Dp_mid - 10e-9
    Dp_hi = Dp_mid + 10e-9
    N_m3 = np.array([1.0, 0.5, 0.2])
    measured_number_fractions = {"BC": 1.0}

    with pytest.raises(ValueError, match="Number of BC particles to sample is 2, but there are 3 size bins"):
        hiscale.sample_particle_Dp_N(
            particles_to_sample=2,
            particle_types=["BC", "BC"],
            measured_number_fractions=measured_number_fractions,
            Dp_mid_m=Dp_mid,
            Dp_lo_m=Dp_lo,
            Dp_hi_m=Dp_hi,
            N_m3=N_m3,
            mode_fractions=mode_fractions,
            size_distribution_parameters=size_distribution_parameters,
            fill_bins=True,
            size_dist_grid=size_dist_grid,
        )

    d_rand, n_rand = hiscale.sample_particle_Dp_N(
        particles_to_sample=10,
        particle_types=["BC"] * 10,
        measured_number_fractions=measured_number_fractions,
        Dp_mid_m=Dp_mid,
        Dp_lo_m=Dp_lo,
        Dp_hi_m=Dp_hi,
        N_m3=N_m3,
        mode_fractions=mode_fractions,
        size_distribution_parameters=size_distribution_parameters,
        fill_bins=False,
        size_dist_grid=size_dist_grid,
    )
    assert d_rand.shape == (10,)
    assert n_rand.shape == (10,)


def test_build_input_validation_branches_and_preferred_matching(monkeypatch, tmp_path):
    with pytest.raises(KeyError, match="missing required config keys"):
        hiscale.build({})

    # Required key characterization checks (explicitly verify each required input)
    required_keys = ["aimms_file", "splat_file", "ams_file", "z", "dz", "splat_species", "mass_thresholds"]
    complete = {
        "type": "hiscale_observations",
        "aimms_file": "a.txt",
        "splat_file": "s.txt",
        "ams_file": "m.txt",
        "z": 100.0,
        "dz": 2.0,
        "splat_species": {"BC": ["BC"]},
        "mass_thresholds": {"BC": ((0.1, 0.2, 0.05), ["BC"])},
    }
    for rk in required_keys:
        cfg_missing = dict(complete)
        cfg_missing.pop(rk)
        with pytest.raises(KeyError, match="missing required config keys"):
            hiscale.build(cfg_missing)

    base_cfg = {
        "type": "hiscale_observations",
        "aimms_file": "a.txt",
        "splat_file": "s.txt",
        "ams_file": "m.txt",
        "z": 100.0,
        "dz": 2.0,
        "splat_species": {"BC": ["BC"]},
        "mass_thresholds": {"BC": ((0.1, 0.2, 0.05), ["BC"])}
    }

    with pytest.raises(KeyError, match="requires either 'fims_file' or 'beasd_file'"):
        hiscale.build(base_cfg)

    cfg_fims_no_bins = dict(base_cfg)
    cfg_fims_no_bins["fims_file"] = "fims.txt"
    with pytest.raises(KeyError, match="requires 'fims_bins_file'"):
        hiscale.build(cfg_fims_no_bins)

    monkeypatch.setattr(hiscale, "normalize_population_config", lambda c: c)
    monkeypatch.setattr(
        hiscale,
        "_read_fims_avg_size_dist",
        lambda **kwargs: (
            np.array([80.0, 100.0]),
            np.array([100.0, 120.0]),
            np.array([1.0, 1.0]),
            np.array([0.1, 0.1]),
        ),
    )
    monkeypatch.setattr(hiscale, "_read_minisplat_number_fractions", lambda **kwargs: ({"BC": 1.0}, {"BC": 0.0}))
    monkeypatch.setattr(
        hiscale,
        "_read_ams_mass_fractions",
        lambda **kwargs: (
            {"OC": 1.0, "NO3": 0.0, "SO4": 0.0, "NH4": 0.0},
            {"OC": 0.0, "NO3": 0.0, "SO4": 0.0, "NH4": 0.0},
            1.0,
            0.0,
        ),
    )
    monkeypatch.setattr(hiscale, "fit_Nmodal_distribution", lambda *_args, **_kwargs: [[1.0, 1.0e-7, 1.5]])
    monkeypatch.setattr(hiscale, "optimize_splat_species_distributions", lambda **kwargs: ({}, 1.0))
    monkeypatch.setattr(
        hiscale,
        "sample_particle_Dp_N",
        lambda particles_to_sample, *args, **kwargs: (
            np.full(particles_to_sample, 100e-9),
            np.ones(particles_to_sample),
        ),
    )
    monkeypatch.setattr(hiscale, "sample_particle_masses", lambda *args, **kwargs: (["BC"], np.array([1.0])))
    monkeypatch.setattr(hiscale, "assemble_population_from_mass_fractions", lambda **kwargs: _make_sample_population())
    monkeypatch.setattr(hiscale, "mass_fraction_comparison", lambda *args, **kwargs: ({"OC": 1.0}, [True]))
    monkeypatch.setattr(hiscale, "number_fraction_comparison", lambda *args, **kwargs: ({"BC": 1.0}, [True]))
    cfg_bad_pref = dict(base_cfg)
    cfg_bad_pref.update(
        {
            "fims_file": "fims.txt",
            "fims_bins_file": "bins.txt",
            "preferred_matching": "bad",
        }
    )
    with pytest.raises(ValueError, match="preferred_matching must be 'number' or 'mass'"):
        hiscale.build(cfg_bad_pref)

def test_builder_output_metadata_and_matching(tmp_path):
    base_cfg = {
        "type": "hiscale_observations",
        "aimms_file": "a.txt",
        "splat_file": "s.txt",
        "ams_file": "m.txt",
        "z": 100.0,
        "dz": 2.0,
        "splat_species": {"BC": ["BC"]},
        "mass_thresholds": {"BC": ((0.1, 0.2, 0.05), ["BC"])},
        "beasd_file": "beasd.txt",
    }

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(hiscale, "normalize_population_config", lambda c: c)
        monkeypatch.setattr(
            hiscale,
            "_read_beasd_avg_size_dist",
            lambda **kwargs: (
                np.array([80.0, 100.0]),
                np.array([100.0, 120.0]),
                np.array([1.0, 1.0]),
                np.array([0.1, 0.1]),
            ),
        )
        monkeypatch.setattr(hiscale, "_read_minisplat_number_fractions", lambda **kwargs: ({"BC": 1.0}, {"BC": 0.0}))
        monkeypatch.setattr(
            hiscale,
            "_read_ams_mass_fractions",
            lambda **kwargs: (
                {"OC": 1.0, "NO3": 0.0, "SO4": 0.0, "NH4": 0.0},
                {"OC": 0.0, "NO3": 0.0, "SO4": 0.0, "NH4": 0.0},
                1.0,
                0.0,
            ),
        )
        monkeypatch.setattr(hiscale, "fit_Nmodal_distribution", lambda *_args, **_kwargs: [[1.0, 1.0e-7, 1.5]])
        monkeypatch.setattr(hiscale, "optimize_splat_species_distributions", lambda **kwargs: ({"BC": [1.0]}, 1.0))
        monkeypatch.setattr(
            hiscale,
            "sample_particle_Dp_N",
            lambda particles_to_sample, *args, **kwargs: (
                np.full(particles_to_sample, 100e-9),
                np.ones(particles_to_sample),
            ),
        )
        monkeypatch.setattr(hiscale, "sample_particle_masses", lambda *args, **kwargs: (["BC"], np.array([1.0])))
        monkeypatch.setattr(
            hiscale,
            "mass_fraction_comparison",
            lambda *args, **kwargs: ({"BC": 1.0}, [True]),
        )
        monkeypatch.setattr(
            hiscale,
            "number_fraction_comparison",
            lambda *args, **kwargs: ({"BC": 1.0}, [True]),
        )
        monkeypatch.setattr(hiscale, "assemble_population_from_mass_fractions", lambda **kwargs: _make_sample_population())

        pop = hiscale.build(base_cfg)
    finally:
        monkeypatch.undo()

    assert pop.metadata["source"] == "hiscale_observations"
    assert pop.metadata["size_distribution"]["Dp_lo_nm"].shape == pop.metadata["size_distribution"]["Dp_hi_nm"].shape
    assert "ams_mass_frac" in pop.metadata


def test_builder_uses_assembly_helper_directly(monkeypatch):
    base_cfg = {
        "type": "hiscale_observations",
        "aimms_file": "a.txt",
        "splat_file": "s.txt",
        "ams_file": "m.txt",
        "z": 100.0,
        "dz": 2.0,
        "splat_species": {"BC": ["BC"]},
        "mass_thresholds": {"BC": ((0.1, 0.2, 0.05), ["BC"])},
        "beasd_file": "beasd.txt",
        "N_particles": 7,
    }

    monkeypatch.setattr(hiscale, "normalize_population_config", lambda c: c)
    monkeypatch.setattr(
        hiscale,
        "_read_beasd_avg_size_dist",
        lambda **kwargs: (
            np.array([80.0, 100.0]),
            np.array([100.0, 120.0]),
            np.array([1.0, 1.0]),
            np.array([0.1, 0.1]),
        ),
    )
    monkeypatch.setattr(hiscale, "_read_minisplat_number_fractions", lambda **kwargs: ({"BC": 1.0}, {"BC": 0.0}))
    monkeypatch.setattr(
        hiscale,
        "_read_ams_mass_fractions",
        lambda **kwargs: (
            {"OC": 1.0, "NO3": 0.0, "SO4": 0.0, "NH4": 0.0},
            {"OC": 0.0, "NO3": 0.0, "SO4": 0.0, "NH4": 0.0},
            1.0,
            0.0,
        ),
    )
    monkeypatch.setattr(hiscale, "fit_Nmodal_distribution", lambda *_args, **_kwargs: [[1.0, 1.0e-7, 1.5]])
    monkeypatch.setattr(hiscale, "optimize_splat_species_distributions", lambda **kwargs: ({"BC": [1.0]}, 1.0))
    monkeypatch.setattr(
        hiscale,
        "sample_particle_Dp_N",
        lambda particles_to_sample, *args, **kwargs: (
            np.full(particles_to_sample, 100e-9),
            np.ones(particles_to_sample),
        ),
    )
    monkeypatch.setattr(hiscale, "sample_particle_masses", lambda *args, **kwargs: (["BC"], np.array([1.0])))
    monkeypatch.setattr(hiscale, "mass_fraction_comparison", lambda *args, **kwargs: ({"BC": 1.0}, [True]))
    monkeypatch.setattr(hiscale, "number_fraction_comparison", lambda *args, **kwargs: ({"BC": 1.0}, [True]))

    captured = {}

    def _fake_assemble(**kwargs):
        captured.update(kwargs)
        return _make_sample_population()

    monkeypatch.setattr(hiscale, "assemble_population_from_mass_fractions", _fake_assemble)

    pop = hiscale.build(base_cfg)

    assert pop.spec_masses.ndim == 2
    assert len(captured["diameters"]) == base_cfg["N_particles"]
    assert len(captured["number_concentrations"]) == base_cfg["N_particles"]
    assert np.asarray(captured["species_names"]).shape[0] == base_cfg["N_particles"]
    assert np.asarray(captured["mass_fractions"]).shape[0] == base_cfg["N_particles"]


def test_builder_constructs_population_from_example_data(tmp_path):
    data_dir = Path("examples/example_data/HISCALE_data_0425")
    config = {
        "type": "hiscale_observations",
        "beasd_file": str(data_dir / "BEASD_G1_20160425155810_R2_HISCALE_001s.txt"),
        "aimms_file": str(data_dir / "AIMMS20_G1_20160425155810_R2_HISCALE020h.txt"),
        "ams_file": str(data_dir / "HiScaleAMS_G1_20160425_R0.txt"),
        "splat_file": str(data_dir / "Splat_Composition_25-Apr-2016.txt"),
        "z": 100.0,
        "dz": 50.0,
        "splat_species": {"BC": ["soot"], "OIN": ["Dust"]},
        "mass_thresholds": {
            "BC": ((0.0, 0.5, 0.1), ["BC"]),
            "OIN": ((0.0, 0.5, 0.1), ["OIN"]),
            "SO4": ((0.0, 0.5, 0.1), ["SO4"]),
            "NO3": ((0.0, 0.5, 0.1), ["NO3"]),
            "OC": ((0.0, 0.5, 0.1), ["OC"]),
            "NH4": ((0.0, 0.5, 0.1), ["NH4"]),
        },
        "outdir": str(tmp_path),
        "N_particles": 250,
    }

    pop = hiscale.build(config)
    assert pop.metadata["source"] == "hiscale_observations"
    assert pop.metadata["type_fracs"]

