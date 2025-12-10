import sys

import numpy as np
import pytest

from part2pop.analysis import distributions as dist


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
    total = np.trapz(dens, u_centers)
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
