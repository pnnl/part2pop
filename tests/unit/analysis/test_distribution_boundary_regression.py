import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.distributions import kde1d_in_measure, make_edges
from part2pop.analysis.population.factory.dNdlnD import build as build_diameter
from part2pop.analysis.population.factory.INSA_distribution import build as build_insa
from part2pop.freezing.builder import build_freezing_population


@pytest.mark.parametrize("variable", ["diameter", "insa"])
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("grid_source", ["bins", "edges"])
@pytest.mark.parametrize("normalize", [False, True])
def test_point_mass_inside_outer_half_bin_keeps_its_count(
    variable, side, grid_source, normalize
):
    particle = make_particle(100e-9, ["OC"], [1.0])
    pop = ParticlePopulation(
        particle.species, np.array([particle.masses]), np.array([100.0]), [0]
    )
    if variable == "diameter":
        location = particle.get_Dwet()
        factory, axis, key = build_diameter, "D", "dNdlnD"
        lo_key, hi_key = "D_min", "D_max"
    else:
        location = build_freezing_population(pop, {"morphology": "homogeneous"}).INSA[0]
        factory, axis, key = build_insa, "INSA", "dNdlnINSA"
        lo_key, hi_key = "INSA_min", "INSA_max"
    lo, hi = (
        (location / 1.1, location * 10)
        if side == "lower"
        else (location / 10, location * 1.1)
    )
    cfg = {"method": "kde", "normalize": normalize}
    if grid_source == "bins":
        cfg.update({lo_key: lo, hi_key: hi, "N_bins": 4})
    else:
        cfg["edges"] = make_edges(lo, hi, 4)[0]
    out = factory(cfg).compute(pop, as_dict=True)
    assert out["edges"][0] < location < out["edges"][-1]
    assert location < out[axis][0] if side == "lower" else location > out[axis][-1]
    endpoint = 0 if side == "lower" else -1
    assert np.count_nonzero(out[key]) == 1
    assert out[key][endpoint] > 0
    if normalize and variable == "insa":
        assert out[key].max() == pytest.approx(1.0)
    else:
        count = np.trapezoid(out[key], np.log(out[axis]))
        assert count == pytest.approx(1.0 if normalize else 100.0)


@pytest.mark.parametrize("location", [0.9, 1.0, 1.1, 7.0, 8.0, 9.0])
def test_point_mass_bin_support_includes_edges_and_excludes_outside(location):
    grid = np.array([2.0, 4.0, 6.0])
    out = kde1d_in_measure(
        [location], [100.0], grid, measure="linear", point_mass_bounds=(1.0, 8.0)
    )
    assert np.trapezoid(out, grid) == pytest.approx(
        100.0 if 1.0 <= location <= 8.0 else 0.0
    )


@pytest.mark.parametrize(
    "bounds",
    [(1.0,), (1.0, 8.0, 9.0), (3.0, 8.0), (1.0, 5.0), (np.nan, 8.0), (1.0, np.inf)],
)
def test_point_mass_bounds_must_enclose_query_grid(bounds):
    with pytest.raises(ValueError, match="bounds"):
        kde1d_in_measure(
            [4.0], [100.0], [2.0, 4.0, 6.0], measure="linear", point_mass_bounds=bounds
        )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_insa_kde_omits_zero_area_without_losing_active_count(normalize, mixed):
    particle = make_particle(100e-9, ["SO4", "OC"], [1.0, 0.0])
    rows = [particle.masses]
    weights = [700.0]
    if mixed:
        rows.append(make_particle(100e-9, ["SO4", "OC"], [0.0, 1.0]).masses)
        weights.append(300.0)
    pop = ParticlePopulation(
        particle.species, np.array(rows), np.array(weights), list(range(len(rows)))
    )
    grid = np.geomspace(1e-16, 1e-11, 101)
    before = pop.spec_masses.copy()
    out = build_insa(
        {"method": "kde", "normalize": normalize, "insa_grid": grid}
    ).compute(pop)
    assert np.isfinite(out).all()
    if not mixed:
        np.testing.assert_array_equal(out, np.zeros_like(grid))
    elif normalize:
        assert out.max() == pytest.approx(1.0)
    else:
        assert np.trapezoid(out, np.log(grid)) == pytest.approx(300.0)
    np.testing.assert_array_equal(pop.spec_masses, before)
