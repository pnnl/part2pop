"""Integration coverage driven by examples/04_hiscale_observations.ipynb."""

import numpy as np

from pathlib import Path

from part2pop.population.factory import hiscale_observations as hiscale
from part2pop.population.factory.helpers import hiscale as hiscale_helpers


def _example_paths():
    base = Path("examples/example_data/HISCALE_data_0425")
    return {
        "aimms": base / "AIMMS20_G1_20160425155810_R2_HISCALE020h.txt",
        "beasd": base / "BEASD_G1_20160425155810_R2_HISCALE_001s.txt",
        "ams": base / "HiScaleAMS_G1_20160425_R0.txt",
        "splat": base / "Splat_Composition_25-Apr-2016.txt",
    }


def test_hiscale_example_matches_observations(tmp_path):
    paths = _example_paths()
    assert all(p.exists() for p in paths.values())

    config = {
        "type": "hiscale_observations",
        "beasd_file": str(paths["beasd"]),
        "aimms_file": str(paths["aimms"]),
        "ams_file": str(paths["ams"]),
        "splat_file": str(paths["splat"]),
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
        "fill_bins": False,
    }

    population = hiscale.build(config)
    metadata = population.metadata
    assert metadata["source"] == "hiscale_observations"

    beasd_lo, beasd_hi, beasd_n_cm3, _ = hiscale_helpers._read_beasd_avg_size_dist(
        beasd_file=str(paths["beasd"]),
        aimms_file=str(paths["aimms"]),
        z=100.0,
        dz=50.0,
    )

    expected_n_m3 = beasd_n_cm3 * 1e6
    actual_n = metadata["size_distribution"]["N_bin_m3"]
    np.testing.assert_allclose(actual_n, expected_n_m3, rtol=0.05, atol=1e-3)

    expected_type_fracs, _ = hiscale_helpers._read_minisplat_number_fractions(
        splat_file=str(paths["splat"]),
        aimms_file=str(paths["aimms"]),
        size_dist_type="BEASD",
        size_dist_file=str(paths["beasd"]),
        splat_species={"BC": ["soot"], "OIN": ["Dust"]},
        z=100.0,
        dz=50.0,
    )
    actual_type_fracs = metadata["type_fracs"]
    for key, expect in expected_type_fracs.items():
        assert key in actual_type_fracs
        assert np.isclose(actual_type_fracs[key], expect, rtol=0.01, atol=1e-3)

    expected_mass_frac, _, _, _ = hiscale_helpers._read_ams_mass_fractions(
        ams_file=str(paths["ams"]),
        aimms_file=str(paths["aimms"]),
        size_dist_type="BEASD",
        size_dist_file=str(paths["beasd"]),
        z=100.0,
        dz=50.0,
    )
    actual_ams = metadata["ams_mass_frac"]
    for key, expect in expected_mass_frac.items():
        assert key in actual_ams
        assert np.isclose(actual_ams[key], expect, rtol=0.05, atol=1e-3)