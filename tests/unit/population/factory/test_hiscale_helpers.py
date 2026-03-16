import io
from pathlib import Path

import numpy as np
import pytest

from part2pop.population.factory import hiscale_observations as hiscale


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