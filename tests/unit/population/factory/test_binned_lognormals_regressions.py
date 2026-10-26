import pytest

from part2pop import build_population


def _config():
    return {
        "type": "binned_lognormals",
        "GMD": [0.07e-6],
        "GSD": [1.6],
        "N": [1e8],
        "N_bins": [20],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
    }


@pytest.mark.parametrize("bad_n_bins", [0, 1, 1.5, 2.5, "bogus"])
def test_binned_lognormals_rejects_invalid_bin_count(bad_n_bins):
    config = _config()
    config["N_bins"] = [bad_n_bins]

    with pytest.raises(ValueError, match="integer >= 2"):
        build_population(config)


def test_binned_lognormals_rejects_duplicate_species_after_alias_resolution():
    config = _config()
    # "soot" resolves to BC, so this would otherwise become BC twice and
    # dict(zip(...)) would silently discard one fraction.
    config["aero_spec_names"] = [["soot", "BC"]]
    config["aero_spec_fracs"] = [[0.5, 0.5]]

    with pytest.raises(ValueError, match="duplicate canonical species"):
        build_population(config)


@pytest.mark.parametrize("bad_n_bins", ["bogus", None, float("inf"), float("nan")])
def test_binned_lognormals_reports_invalid_scalar_bin_count(bad_n_bins):
    config = _config()
    config["N_bins"] = bad_n_bins

    with pytest.raises(ValueError, match="integer >= 2"):
        build_population(config)


def test_binned_lognormals_accepts_numeric_string_without_mutating_config():
    config = _config()
    config["N_bins"] = "3"

    population = build_population(config)

    assert len(population.num_concs) == 3
    assert config["N_bins"] == "3"
