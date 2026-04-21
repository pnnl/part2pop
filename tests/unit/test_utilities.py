import pytest

import part2pop.utilities as util


def test_get_number_parses_variants_and_errors():
    assert util.get_number("1.2") == 1.2
    assert util.get_number("1.2x10^3") == 1200.0
    assert util.get_number("1.2×10-3") == 0.0012
    assert util.get_number(5) == 5.0

    with pytest.raises(ValueError):
        util.get_number("not-a-number")


def test_power_moments_from_lognormals_additive():
    single = util.power_moments_from_lognormal(2, N=1.0, gmd=2.0, gsd=1.5)
    summed = util.power_moments_from_lognormals(2, [1.0, 2.0], [2.0, 3.0], [1.5, 1.1])
    assert summed > single
    # Should be linear in N for fixed gmd/gsd
    double = util.power_moments_from_lognormal(2, N=2.0, gmd=2.0, gsd=1.5)
    assert pytest.approx(double) == 2 * single
