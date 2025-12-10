from part2pop.freezing.factory.utils import calculate_Psat


def test_calculate_psat_returns_positive_values():
    psat_wv, psat_ice = calculate_Psat(270.0)
    assert psat_wv > 0.0
    assert psat_ice > 0.0


def test_calculate_psat_increases_with_temperature():
    low_wv, low_ice = calculate_Psat(260.0)
    high_wv, high_ice = calculate_Psat(280.0)
    assert high_wv > low_wv
    assert high_ice > low_ice
