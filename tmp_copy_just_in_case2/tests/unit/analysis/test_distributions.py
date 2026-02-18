import sys

import numpy as np
import pytest

from part2pop.analysis import distributions as dist


def _get_trapezoid():
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = getattr(np, "trapz", None)
    if trap is None:
        raise AttributeError("numpy is missing both trapezoid and trapz")
    return trap


TRAPEZOID = _get_trapezoid()


def _density1d_inputs():
    x = np.array([0.5, 1.5, 2.5])
    weights = np.ones_like(x)
    edges = np.array([0.1, 1.0, 2.0, 3.0])
    return x, weights, edges


def test_density1d_from_samples_normalize_without_trapezoid(monkeypatch):
    trapz = getattr(np, "trapz", None)
    if trapz is None:
        pytest.skip("numpy trapz unavailable in this environment")
    if hasattr(np, "trapezoid"):
        monkeypatch.delattr(np, "trapezoid")
    x, weights, edges = _density1d_inputs()
    centers, dens, _ = dist.density1d_from_samples(
        x, weights, edges, measure="ln", normalize=True
    )
    total = trapz(dens, np.log(centers))
    assert np.isclose(total, 1.0)


def test_density1d_from_samples_normalize_without_trapz(monkeypatch):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        pytest.skip("numpy trapezoid unavailable in this environment")
    if hasattr(np, "trapz"):
        monkeypatch.delattr(np, "trapz")
    x, weights, edges = _density1d_inputs()
    centers, dens, _ = dist.density1d_from_samples(
        x, weights, edges, measure="ln", normalize=True
    )
    total = trapezoid(dens, np.log(centers))
    assert np.isclose(total, 1.0)


def test_make_edges_computes_log_and_linear_scales():
    log_edges, log_centers = dist.make_edges(1.0, 10.0, 3, scale="log")
    assert log_edges.shape == (4,)
    assert log_centers.shape == (3,)
    assert log_centers[0] > 0

    lin_edges, lin_centers = dist.make_edges(0.0, 4.0, 2, scale="linear")
    assert np.allclose(lin_centers, [1.0, 3.0])
    with pytest.raises(ValueError):
        dist.make_edges(-1.0, 1.0, 2, scale="log")
    with pytest.raises(ValueError):
        dist.make_edges(1.0, 2.0, 2, scale="invalid")


def test_bin_widths_and_u_transform_errors():
    edges = np.array([1.0, 2.0, 3.0])
    assert np.allclose(dist.bin_widths(edges, measure="linear"), np.diff(edges))
    with pytest.raises(ValueError):
        dist.bin_widths(np.array([0.0, 1.0]), measure="ln")
    with pytest.raises(ValueError):
        dist.bin_widths(edges, measure="bad")

    with pytest.raises(ValueError):
        dist._u_from_x(np.array([0.0, 1.0]), measure="ln")
    with pytest.raises(ValueError):
        dist._u_from_x(np.array([1.0, 2.0]), measure="bad")


def test_density1d_from_samples_normalize():
    x = np.array([0.5, 1.5, 2.5])
    weights = np.ones_like(x)
    edges = np.array([0.1, 1.0, 2.0, 3.0])
    centers, dens, passed_edges = dist.density1d_from_samples(
        x, weights, edges, measure="ln", normalize=True
    )
    assert np.allclose(passed_edges, edges)
    assert centers.shape == dens.shape
    u_centers = np.log(centers)
    total = TRAPEZOID(dens, u_centers)
    assert np.isclose(total, 1.0)


def test_density1d_cdf_map_errors_and_success():
    x_src = np.array([1.0, 3.0])
    dens_src = np.array([0.5, 1.5])
    edges_tgt = np.array([0.5, 2.0, 4.0])

    # mismatched shapes
    with pytest.raises(ValueError):
        dist.density1d_cdf_map(x_src_centers=np.array([[1.0]]), dens_src=dens_src, edges_tgt=edges_tgt)
    with pytest.raises(ValueError):
        dist.density1d_cdf_map(x_src, np.array([1.0]), edges_tgt)
    with pytest.raises(ValueError):
        dist.density1d_cdf_map(x_src, dens_src, np.array([[1.0], [2.0]]))

    centers, dens_mapped, out_edges = dist.density1d_cdf_map(x_src, dens_src, edges_tgt)
    assert centers.shape == (2,)
    assert np.allclose(out_edges, edges_tgt)
    assert np.all(dens_mapped >= 0.0)


def test_kde1d_requires_scipy(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "scipy.stats", types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="scipy is required"):
        dist.kde1d_in_measure(
            x=np.array([1.0]), weights=np.array([1.0]), xq=np.array([1.0])
        )


def test_density2d_from_samples_normalizes_two_dimensional():
    x = np.array([1.0, 2.1, 3.2, 1.5])
    y = np.array([0.5, 1.5, 2.5, 3.5])
    edges_x = np.linspace(1.0, 3.0, 3)
    edges_y = np.linspace(0.0, 4.0, 3)
    weights = np.ones_like(x)

    cx, cy, dens, out_x, out_y = dist.density2d_from_samples(
        x, y, weights, edges_x, edges_y, measure_x="linear", measure_y="linear", normalize=True
    )

    assert np.allclose(out_x, edges_x)
    assert np.allclose(out_y, edges_y)
    assert cx.shape == (edges_x.size - 1,)
    assert cy.shape == (edges_y.size - 1,)
    assert np.all(dens >= 0.0)
    total = TRAPEZOID(TRAPEZOID(dens, cy, axis=1), cx)
    assert np.isclose(total, 1.0)


def test_density2d_from_samples_handles_log_scales():
    x = np.array([1.2, 1.8, 2.5])
    y = np.array([0.2, 0.5, 0.9])
    edges_x = np.array([1.0, 2.0, 3.0])
    edges_y = np.array([0.1, 1.0])
    weights = np.array([1.0, 2.0, 1.0])

    _, _, dens, _, _ = dist.density2d_from_samples(x, y, weights, edges_x, edges_y)
    assert dens.shape == (edges_x.size - 1, edges_y.size - 1)


def test_density2d_cdf_map_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        dist.density2d_cdf_map(
            x_src_centers=np.array([1.0]),
            y_src_centers=np.array([1.0, 2.0]),
            dens_src=np.ones((2, 2)),
            edges_x_tgt=np.array([1.0, 2.0]),
            edges_y_tgt=np.array([1.0, 2.0]),
        )


def test_density2d_cdf_map_projects_onto_new_edges():
    x_src_centers = np.array([1.0, 2.0])
    y_src_centers = np.array([10.0, 20.0])
    dens_src = np.array([[1.0, 2.0], [3.0, 4.0]])
    edges_x_tgt = np.array([1.0, 1.5, 2.5])
    edges_y_tgt = np.array([5.0, 15.0, 25.0])

    cx, cy, dens_tgt, out_x, out_y = dist.density2d_cdf_map(
        x_src_centers,
        y_src_centers,
        dens_src,
        edges_x_tgt,
        edges_y_tgt,
        measure_x="linear",
        measure_y="linear",
    )

    assert np.allclose(out_x, edges_x_tgt)
    assert np.allclose(out_y, edges_y_tgt)
    assert dens_tgt.shape == (edges_x_tgt.size - 1, edges_y_tgt.size - 1)
    assert np.all(dens_tgt >= 0.0)


def test_kde1d_in_measure_normalizes_with_scipy():
    x = np.array([1.0, 2.0, 3.0])
    weights = np.ones_like(x)
    xq = np.linspace(1.0, 3.0, 10)

    dens = dist.kde1d_in_measure(x, weights, xq, normalize=True)
    assert dens.shape == xq.shape
    assert np.isfinite(dens).all()
    total = TRAPEZOID(dens, np.log(xq))
    assert np.isclose(total, 1.0, atol=1e-2)


def test_u_from_x_linear_passes_through():
    arr = np.array([1.2, 2.3, 4.5])
    result = dist._u_from_x(arr, measure="linear")
    assert np.allclose(arr, result)


def test_density1d_cdf_map_handles_single_bin():
    centers = np.array([2.0])
    dens = np.array([1.0])
    edges = np.array([1.0, 3.0])
    cx, dens_out, out_edges = dist.density1d_cdf_map(centers, dens, edges, measure="linear")
    assert cx.size == edges.size - 1
    assert np.allclose(out_edges, edges)
    assert np.all(dens_out >= 0.0)


def test_density1d_from_samples_linear_measure_no_normalize():
    x = np.array([0.5, 1.0, 1.5])
    weights = np.ones_like(x)
    edges = np.array([0.1, 1.0, 2.0])
    centers, dens, _ = dist.density1d_from_samples(x, weights, edges, measure="linear", normalize=False)
    expected_centers = 0.5 * (edges[:-1] + edges[1:])
    assert np.allclose(centers, expected_centers)
    assert np.all(dens >= 0.0)
