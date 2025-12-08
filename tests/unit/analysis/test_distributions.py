import numpy as np
import pytest

import part2pop.analysis.distributions as dist


def test_make_edges_and_bin_widths_linear_and_log():
    edges_lin, centers_lin = dist.make_edges(0.0, 10.0, 5, scale="linear")
    assert len(edges_lin) == 6
    assert np.allclose(centers_lin, np.linspace(1, 9, 5))

    edges_log, centers_log = dist.make_edges(1.0, 100.0, 2, scale="log")
    assert np.allclose(edges_log, [1.0, 10.0, 100.0])
    assert np.allclose(centers_log, [np.sqrt(10), np.sqrt(1000)])

    with pytest.raises(ValueError):
        dist.make_edges(-1.0, 1.0, 3, scale="log")
    with pytest.raises(ValueError):
        dist.bin_widths([0.0, 1.0], measure="ln")
    with pytest.raises(ValueError):
        dist.bin_widths([0, 1], measure="bogus")


def test_density1d_from_samples_conserves_total_and_normalizes():
    edges = np.array([1.0, 2.0, 3.0])
    x = np.array([1.2, 1.8, 2.2, 2.8])
    weights = np.ones_like(x)

    centers, dens, _ = dist.density1d_from_samples(x, weights, edges, measure="linear")
    assert np.allclose(centers, [1.5, 2.5])
    assert np.allclose(dens, [2.0, 2.0])

    centers_n, dens_n, _ = dist.density1d_from_samples(
        x, weights, edges, measure="linear", normalize=True
    )
    total = np.trapz(dens_n, centers_n)
    assert np.isclose(total, 1.0)


def test_density1d_cdf_map_conserves_total_and_validates_inputs():
    src_centers = np.array([1.0, 2.0])
    dens_src = np.array([1.0, 1.0])
    edges_tgt = np.array([1.0, 1.5, 3.0])

    tgt_centers, dens_tgt, _ = dist.density1d_cdf_map(
        x_src_centers=src_centers,
        dens_src=dens_src,
        edges_tgt=edges_tgt,
        measure="linear",
    )
    assert tgt_centers.shape == (2,)
    assert np.all(dens_tgt >= 0)
    total_src = np.sum(dens_src * np.diff([0.5, 1.5, 2.5]))  # approximate total in src
    total_tgt = np.sum(dens_tgt * np.diff(edges_tgt))
    assert total_tgt <= total_src + 1e-6

    with pytest.raises(ValueError):
        dist.density1d_cdf_map(np.array([[1.0]]), dens_src, edges_tgt)
    with pytest.raises(ValueError):
        dist.density1d_cdf_map(src_centers, np.array([[1.0]]), edges_tgt)
    with pytest.raises(ValueError):
        dist.density1d_cdf_map(src_centers, dens_src, np.array([1.0]))


def test_density2d_from_samples_normalizes():
    x = np.array([1.0, 2.0, 1.5, 2.5])
    y = np.array([10.0, 20.0, 15.0, 25.0])
    w = np.ones_like(x)
    edges_x = np.array([1.0, 2.0, 3.0])
    edges_y = np.array([10.0, 20.0, 30.0])

    cx, cy, dens, _, _ = dist.density2d_from_samples(
        x, y, w, edges_x, edges_y, measure_x="linear", measure_y="linear", normalize=True
    )
    assert dens.shape == (2, 2)
    total = np.trapz(np.trapz(dens, cy, axis=1), cx)
    assert np.isclose(total, 1.0)


def test_density2d_cdf_map_conserves_total_and_validates_inputs():
    x_centers = np.array([1.0, 2.0])
    y_centers = np.array([10.0, 20.0])
    dens_src = np.full((2, 2), 1.0)
    edges_x_tgt = np.array([1.0, 2.0, 3.0])
    edges_y_tgt = np.array([10.0, 20.0, 30.0])

    cx_tgt, cy_tgt, dens_tgt, _, _ = dist.density2d_cdf_map(
        x_centers, y_centers, dens_src, edges_x_tgt, edges_y_tgt, measure_x="linear", measure_y="linear"
    )
    assert dens_tgt.shape == (2, 2)
    assert np.all(dens_tgt >= 0)
    total_src = np.sum(dens_src * np.diff([0.5, 1.5, 2.5])[:, None] * np.diff([5.0, 15.0, 25.0]))
    total_tgt = np.sum(dens_tgt * np.diff(edges_x_tgt)[:, None] * np.diff(edges_y_tgt))
    assert total_tgt <= total_src + 1e-6

    with pytest.raises(ValueError):
        dist.density2d_cdf_map(x_centers, y_centers, np.ones((3, 3)), edges_x_tgt, edges_y_tgt)
