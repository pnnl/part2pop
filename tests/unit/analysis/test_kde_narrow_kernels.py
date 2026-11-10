"""KDE number density must be conserved when the kernel is narrower than the grid.

Only exactly single-valued input used to take the conservative point-mass
path. Two samples 0.1% apart on a 21-node log grid gave a Gaussian kernel far
narrower than the node spacing: sampled at the nodes, 1e9 m^-3 integrated to
9.5e10 when the samples sat on a node and to about 0 between nodes.
"""

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from part2pop.analysis.distributions import kde1d_in_measure

GRID = np.logspace(-8, -6, 21)          # log spacing 0.230 in ln(D)
U = np.log(GRID)
TOTAL = 1e9


def _pair(center, relative_spread):
    return np.array([center, center * (1 + relative_spread)])


@pytest.mark.parametrize("relative_spread", [1e-12, 1e-6, 1e-3, 1e-2, 5e-2, 0.2])
@pytest.mark.parametrize("position", ["on_node", "between_nodes"])
def test_narrow_pair_conserves_number(relative_spread, position):
    center = GRID[9] if position == "on_node" else np.sqrt(GRID[9] * GRID[10])
    dens = kde1d_in_measure(_pair(center, relative_spread), np.array([TOTAL / 2] * 2), GRID)
    assert np.trapezoid(dens, U) == pytest.approx(TOTAL, rel=1e-12)


def test_narrow_pair_conserves_interior_first_moment():
    samples = _pair(np.sqrt(GRID[9] * GRID[10]), 1e-3)
    weights = np.array([0.3, 0.7]) * TOTAL
    dens = kde1d_in_measure(samples, weights, GRID)
    expected = np.sum(weights * np.log(samples))
    assert np.trapezoid(dens * U, U) == pytest.approx(expected, rel=1e-12)


def test_samples_one_ulp_apart_are_conserved_without_linalg_error():
    samples = np.array([1e-7, np.nextafter(1e-7, 1.0)])
    dens = kde1d_in_measure(samples, np.array([TOTAL / 2] * 2), GRID)
    assert np.trapezoid(dens, U) == pytest.approx(TOTAL, rel=1e-12)


def test_normalized_narrow_pair_integrates_to_one():
    dens = kde1d_in_measure(_pair(GRID[5], 1e-3), np.array([1e-320, 1e-320]), GRID, normalize=True)
    assert np.trapezoid(dens, U) == pytest.approx(1.0, rel=1e-12)


def test_resolved_kernels_are_still_sampled_directly():
    # 200 lognormal samples: bandwidth about 0.75 of the grid spacing.
    rng = np.random.default_rng(1)
    samples = np.exp(rng.normal(np.log(1e-7), 0.5, 200))
    weights = rng.uniform(1.0, 2.0, 200)
    dens = kde1d_in_measure(samples, weights, GRID)
    expected = gaussian_kde(np.log(samples), weights=weights)(U) * weights.sum()
    np.testing.assert_allclose(dens, expected, rtol=1e-12, atol=0.0)


def test_narrow_pair_is_independent_of_query_order():
    samples = _pair(np.sqrt(GRID[9] * GRID[10]), 1e-3)
    ordered = kde1d_in_measure(samples, np.array([1.0, 2.0]), GRID)
    order = np.random.default_rng(2).permutation(GRID.size)
    shuffled = kde1d_in_measure(samples, np.array([1.0, 2.0]), GRID[order])
    np.testing.assert_allclose(shuffled, ordered[order], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("samples", [np.array([1e-5]), np.array([1e-5, 1.001e-5])])
def test_point_masses_outside_the_support_warn_instead_of_vanishing(samples):
    # Used to return all zeros silently: the population simply disappeared.
    with pytest.warns(RuntimeWarning, match="outside the distribution support"):
        dens = kde1d_in_measure(samples, np.ones(samples.size), GRID)
    assert not np.any(dens)
