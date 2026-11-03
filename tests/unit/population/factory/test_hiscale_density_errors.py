import numpy as np
import pytest

from part2pop.population.factory.helpers import hiscale as hiscale_helpers


def test_invalid_beasd_density_measure_raises_value_error_with_requested_value(
    monkeypatch,
):
    monkeypatch.setattr(hiscale_helpers, "_read_aimms", lambda _: {})
    monkeypatch.setattr(
        hiscale_helpers,
        "_read_beasd_core",
        lambda _: {
            "Time(UTC)": np.array([0.0]),
            "_N_bins": np.array([[1.0]]),
        },
    )
    monkeypatch.setattr(
        hiscale_helpers,
        "_time_indices_for_altitude_and_cloudflag",
        lambda **kwargs: np.array([0], dtype=int),
    )
    monkeypatch.setattr(
        hiscale_helpers,
        "_read_beasd_bins",
        lambda *args, **kwargs: (
            np.array([10.0]),
            np.array([20.0]),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"Unknown beasd_density_measure 'bogus'",
    ):
        hiscale_helpers._read_beasd_avg_size_dist(
            beasd_file="unused-beasd.dat",
            aimms_file="unused-aimms.dat",
            z=100.0,
            dz=1.0,
            beasd_density_measure="bogus",
        )
