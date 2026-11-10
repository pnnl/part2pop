"""Numerically singular weighted covariance must retain represented counts."""
import warnings

import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.builder import build_variable
from part2pop.analysis.distributions import kde1d_in_measure


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("total", [1e-300, 1.0, 1e300])
@pytest.mark.parametrize("measure", ["linear", "ln"])
def test_dominant_weight_conserves_integral_and_first_moment(normalize, total, measure):
    samples = np.array([1.5, 2.5])
    weights = np.array([total, total * 1e-20])
    grid = np.linspace(1., 3., 21)
    u = np.log(grid) if measure == "ln" else grid
    locations = np.log(samples) if measure == "ln" else samples
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        density = kde1d_in_measure(samples, weights, grid, measure, normalize)
    expected_weights = weights / weights.sum() if normalize else weights
    assert np.all(np.isfinite(density))
    assert np.trapezoid(density, u) == pytest.approx(expected_weights.sum(), rel=1e-12, abs=0)
    assert np.trapezoid(density * u, u) == pytest.approx(
        np.sum(expected_weights * locations), rel=1e-12, abs=0)


@pytest.mark.parametrize("name", ["dNdlnD", "INSA_distribution"])
def test_public_distributions_accept_dominant_concentration(name):
    particles = [make_particle(d, ["BC"], [1.]) for d in (1e-7, 2e-7)]
    pop = ParticlePopulation(particles[0].species, np.array([p.masses for p in particles]),
                             np.array([1e9, 1e-9]), [0, 1])
    cfg = {"method": "kde", "normalize": False}
    cfg.update({"diam_grid": np.geomspace(1e-8, 1e-6, 31)} if name == "dNdlnD"
               else {"insa_grid": np.geomspace(1e-16, 1e-11, 31)})
    out = build_variable(name, var_cfg=cfg).compute(pop, as_dict=True)
    x, y = ("D", "dNdlnD") if name == "dNdlnD" else ("INSA", "dNdlnINSA")
    assert np.all(np.isfinite(out[y]))
    assert np.trapezoid(out[y], np.log(out[x])) == pytest.approx(1e9, rel=1e-12)


def test_unrelated_kde_value_errors_still_propagate(monkeypatch):
    import scipy.stats
    def broken(*args, **kwargs):
        raise ValueError("unrelated fitting failure")
    monkeypatch.setattr(scipy.stats, "gaussian_kde", broken)
    with pytest.raises(ValueError, match="unrelated fitting failure"):
        kde1d_in_measure(np.array([1., 2.]), np.array([1., 1.]), np.linspace(.5, 3., 20))
