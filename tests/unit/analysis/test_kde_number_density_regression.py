import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.distributions import kde1d_in_measure
from part2pop.analysis.population.factory.INSA_distribution import build

TRAP = getattr(np, "trapezoid", None) or np.trapz


@pytest.mark.parametrize(
    "measure,grid",
    [
        ("ln", np.geomspace(0.001, 10000.0, 10000)),
        ("linear", np.linspace(-30.0, 40.0, 10000)),
    ],
)
def test_regular_kde_preserves_number_and_scales_with_weights(measure, grid):
    x = np.array([1.0, 2.0, 4.0])
    w = np.array([100.0, 200.0, 300.0])
    a = kde1d_in_measure(x, w, grid, measure=measure)
    b = kde1d_in_measure(x, 10 * w, grid, measure=measure)
    np.testing.assert_allclose(b, 10 * a, rtol=1e-12, atol=0.0)
    u = np.log(grid) if measure == "ln" else grid
    assert TRAP(a, u) == pytest.approx(600.0, rel=1e-9)
    norm = kde1d_in_measure(x, w, grid, measure=measure, normalize=True)
    assert TRAP(norm, u) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("measure", ["ln", "linear"])
@pytest.mark.parametrize("points", [101, 201])
@pytest.mark.parametrize(
    "x,w", [([2.0], [600.0]), ([2.0, 2.0], [100.0, 500.0]), ([2.0, 3.0], [600.0, 0.0])]
)
def test_degenerate_kde_conserves_number(measure, points, x, w):
    grid = np.geomspace(1.0, 4.0, points)
    out = kde1d_in_measure(np.array(x), np.array(w), grid, measure=measure)
    u = np.log(grid) if measure == "ln" else grid
    assert np.isfinite(out).all()
    assert TRAP(out, u) == pytest.approx(600.0, rel=1e-12)


@pytest.mark.parametrize("measure", ["ln", "linear"])
@pytest.mark.parametrize("value", [1.0, 1.05, 3.3, 4.0, 0.5, 5.0])
def test_point_mass_endpoints_nonuniform_and_outside_domain(measure, value):
    grid = np.array([1.0, 1.1, 2.0, 4.0])
    u = np.log(grid) if measure == "ln" else grid
    out = kde1d_in_measure(np.array([value]), np.array([600.0]), grid, measure=measure)
    expected = 600.0 if 1.0 <= value <= 4.0 else 0.0
    assert TRAP(out, u) == pytest.approx(expected, rel=1e-12, abs=0.0)
    if expected:
        location = np.log(value) if measure == "ln" else value
        # Count conservation alone would also accept depositing at the wrong node.
        assert TRAP(out * u, u) / expected == pytest.approx(
            location, rel=1e-12, abs=1e-15
        )
    norm = kde1d_in_measure(
        np.array([value]), np.array([600.0]), grid, measure=measure, normalize=True
    )
    assert TRAP(norm, u) == pytest.approx(1.0 if expected else 0.0, rel=1e-12, abs=0.0)


def test_regular_kde_keeps_finite_domain_tail_loss():
    grid = np.linspace(1.9, 2.1, 100)
    out = kde1d_in_measure(
        np.array([1.0, 2.0, 4.0]),
        np.array([100.0, 200.0, 300.0]),
        grid,
        measure="linear",
    )
    assert 0.0 < TRAP(out, grid) < 600.0


def test_zero_weights_and_invalid_degenerate_query():
    grid = np.array([1.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        kde1d_in_measure(np.array([2.0, 3.0]), np.zeros(2), grid), np.zeros(3)
    )
    with pytest.raises(ValueError):
        kde1d_in_measure(np.array([2.0]), np.array([1.0]), np.array([2.0]))
    # Ordinary KDE still permits a single evaluation point.
    assert kde1d_in_measure(
        np.array([1.0, 2.0, 4.0]), np.ones(3), np.array([2.0])
    ).shape == (1,)


def test_empty_supported_insa_stays_finite_when_peak_normalized():
    p = make_particle(100e-9, ["OC"], [1.0])
    pop = ParticlePopulation(p.species, np.array([p.masses]), np.array([600.0]), [0])
    out = build(
        {"method": "kde", "normalize": True, "insa_grid": np.geomspace(1e-10, 1e-8, 20)}
    ).compute(pop)
    np.testing.assert_array_equal(out, np.zeros(20))


@pytest.mark.parametrize(
    "query", [[2.0], [], [4.0, 2.0, 1.0], [1.0, 2.0, 2.0], [1.0, 4.0, 2.0]]
)
def test_normalization_requires_an_increasing_integration_grid(query):
    with pytest.raises(ValueError, match="increasing"):
        kde1d_in_measure(
            np.array([1.0, 2.0, 4.0]), np.ones(3), np.array(query), normalize=True
        )


def test_raw_kde_preserves_arbitrary_query_order():
    grid = np.array([1.0, 2.0, 4.0])
    ordered = kde1d_in_measure(grid, np.array([10.0, 20.0, 30.0]), grid)
    order = [2, 0, 1]
    out = kde1d_in_measure(grid, np.array([10.0, 20.0, 30.0]), grid[order])
    np.testing.assert_allclose(out, ordered[order], rtol=1e-12)


def test_point_mass_requires_representable_integration_widths():
    with (
        np.errstate(over="raise", invalid="raise"),
        pytest.raises(ValueError, match="finite"),
    ):
        kde1d_in_measure(
            np.array([0.0]),
            np.array([1.0]),
            np.array([-1e308, 1e308]),
            measure="linear",
        )


@pytest.mark.parametrize("weights", [[1.0, -1.0], [1.0, np.nan], [1.0, np.inf]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        kde1d_in_measure(
            np.array([1.0, 2.0]), np.array(weights), np.array([1.0, 2.0, 4.0])
        )


def test_filtering_zero_weights_matches_removing_samples_and_does_not_mutate_inputs():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([100.0, 200.0, 0.0, 300.0])
    grid = np.geomspace(0.1, 40.0, 30)
    before = [v.copy() for v in (x, weights, grid)]
    out = kde1d_in_measure(x, weights, grid)
    control = kde1d_in_measure(x[[0, 1, 3]], weights[[0, 1, 3]], grid)
    np.testing.assert_allclose(out, control, rtol=1e-12)
    for value, original in zip((x, weights, grid), before):
        np.testing.assert_array_equal(value, original)


def test_public_diameter_kde_retains_absolute_number_concentration():
    from part2pop.analysis.builder import build_variable

    parts = [make_particle(d, ["OC"], [1.0]) for d in [50e-9, 100e-9, 200e-9]]
    pop = ParticlePopulation(
        parts[0].species,
        np.array([p.masses for p in parts]),
        np.array([100.0, 200.0, 300.0]),
        [0, 1, 2],
    )
    cfg = {"method": "kde", "diam_grid": np.geomspace(1e-10, 1e-4, 1000)}
    variable = build_variable("dNdlnD", var_cfg=cfg)
    out = variable.compute(pop, as_dict=True)
    assert TRAP(out["dNdlnD"], np.log(out["D"])) == pytest.approx(600.0, rel=1e-9)
    pop.num_concs *= 10
    scaled = variable.compute(pop, as_dict=True)
    np.testing.assert_allclose(
        scaled["dNdlnD"], 10 * out["dNdlnD"], rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_public_monodisperse_insa_kde_keeps_count_or_existing_peak_convention(
    normalize,
):
    p = make_particle(100e-9, ["OC"], [1.0])
    pop = ParticlePopulation(p.species, np.array([p.masses]), np.array([600.0]), [0])
    grid = np.geomspace(1e-16, 1e-11, 101)
    out = build({"method": "kde", "normalize": normalize, "insa_grid": grid}).compute(
        pop
    )
    assert np.isfinite(out).all()
    if normalize:
        assert np.max(out) == pytest.approx(1.0)
    else:
        assert TRAP(out, np.log(grid)) == pytest.approx(600.0, rel=1e-12)


@pytest.mark.parametrize(
    "x,weights,query",
    [
        ([[1.0, 2.0]], [1.0, 1.0], [1.0, 2.0, 4.0]),
        ([1.0, 2.0], [[1.0, 1.0]], [1.0, 2.0, 4.0]),
        ([1.0, 2.0], [1.0], [1.0, 2.0, 4.0]),
        ([1.0, 2.0], [1.0, 1.0], [[1.0, 2.0, 4.0]]),
    ],
)
def test_kde_rejects_mismatched_or_multidimensional_inputs(x, weights, query):
    with pytest.raises(ValueError, match="1D"):
        kde1d_in_measure(x, weights, query)


@pytest.mark.parametrize(
    "target,value",
    [("sample", np.nan), ("sample", np.inf), ("query", np.nan), ("query", np.inf)],
)
def test_kde_rejects_nonfinite_coordinates(target, value):
    x = [1.0, 2.0]
    query = [1.0, 2.0, 4.0]
    (x if target == "sample" else query)[0] = value
    with pytest.raises(ValueError, match="coordinates must be finite"):
        kde1d_in_measure(x, [1.0, 1.0], query)


def test_finite_individual_weights_cannot_overflow_total_weight():
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="Total KDE weight must be finite"),
    ):
        kde1d_in_measure([1.0, 2.0], [1e308, 1e308], [1.0, 2.0, 4.0])
