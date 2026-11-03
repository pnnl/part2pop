import numpy as np
import pytest
from part2pop.analysis.distributions import kde1d_in_measure

TRAP = getattr(np, "trapezoid", None) or np.trapz


@pytest.mark.parametrize("measure", ["linear", "ln"])
@pytest.mark.parametrize("single_value", [False, True])
@pytest.mark.parametrize("scale", [1e-320, 1.0, 1e300])
def test_normalized_kde_is_finite_and_independent_of_total_weight(
    measure, single_value, scale
):
    query = np.linspace(1 - 1e-12, 1 + 3e-12, 1001)
    samples = (
        np.array([1 + 1e-12]) if single_value else np.array([1.0, 1 + 1e-12, 1 + 2e-12])
    )
    weights = np.ones(samples.size)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        density = kde1d_in_measure(
            samples, weights * scale, query, measure=measure, normalize=True
        )
    control = kde1d_in_measure(samples, weights, query, measure=measure, normalize=True)
    coordinate = np.log(query) if measure == "ln" else query
    assert np.isfinite(density).all()
    assert TRAP(density, coordinate) == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(density, control, rtol=1e-12, atol=0)


@pytest.mark.parametrize("measure", ["linear", "ln"])
def test_normalized_broad_kde_avoids_subnormal_concentration_roundoff(measure):
    samples = np.array([1.0, 2.0, 4.0])
    weights = np.array([1.0, 2.0, 3.0])
    query = np.geomspace(0.01, 100.0, 1000)
    density = kde1d_in_measure(
        samples, weights * 1e-320, query, measure=measure, normalize=True
    )
    control = kde1d_in_measure(samples, weights, query, measure=measure, normalize=True)
    coordinate = np.log(query) if measure == "ln" else query
    assert TRAP(density, coordinate) == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(density, control, rtol=1e-12, atol=0)
