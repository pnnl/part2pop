"""
Tests for part2pop.utilities
"""
import numpy as np
import pytest

from part2pop.utilities import (
    get_number,
    power_moments_from_lognormal,
    power_moments_from_lognormals,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("1.0", 1.0),
        ("3.5e2", 350.0),
        ("2×10^3", 2000.0),
        ("2x10^3", 2000.0),
        ("  4.2 × 10^1  ", 42.0),
    ],
)
def test_get_number_parses_various_notations(s, expected):
    val = get_number(s)
    assert np.isclose(val, expected)


def test_get_number_raises_on_invalid_input():
    with pytest.raises(ValueError):
        get_number("not-a-number")


def test_power_moments_from_lognormal_matches_manual_formulation():
    # For k=0 the expression should reduce to N
    N = 1000.0
    gmd = 0.1e-6
    gsd = 1.5
    k = 0.0
    m0 = power_moments_from_lognormal(k, N, gmd, gsd)
    assert np.isclose(m0, N)

    # For k=3, compare against a known closed form
    k = 3.0
    m3 = power_moments_from_lognormal(k, N, gmd, gsd)
    expected = N * np.exp(k * np.log(gmd) + k ** 2 * np.log(gsd) / 2.0)
    assert np.isclose(m3, expected)


def test_power_moments_from_lognormals_adds_contributions():
    Ns = np.array([100.0, 200.0])
    GMDs = np.array([0.05e-6, 0.1e-6])
    GSDs = np.array([1.4, 1.6])
    k = 3.0

    separate = [
        power_moments_from_lognormal(k, N, gmd, gsd)
        for (N, gmd, gsd) in zip(Ns, GMDs, GSDs)
    ]
    combined = power_moments_from_lognormals(k, Ns, GMDs, GSDs)

    assert np.isclose(combined, sum(separate))
