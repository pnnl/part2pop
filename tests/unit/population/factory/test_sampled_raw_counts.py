"""Sample counts must not be truncated by generic population normalization."""
from copy import deepcopy
import numpy as np
import pytest

from part2pop.population.factory import sampled_lognormals


def cfg(counts):
    return {"N": [1., 2.], "GMD": [1e-7, 2e-7], "GSD": [1.2, 1.3],
            "aero_spec_names": [["SO4"], ["SO4"]],
            "aero_spec_fracs": [[1.], [1.]], "N_parts": counts, "seed": 7}


@pytest.mark.parametrize("counts", [2.9, "2.9", [2.9, 3.9], (2, 3.5),
    np.array([2., 3.9]), [[2], [3]], 0, -1, np.nan, np.inf, None,
    [2, 0], [2, -1], [2, np.nan], [2, np.inf], 2 ** 100])
def test_noninteger_or_nonpositive_counts_fail_before_sampling(counts):
    with pytest.raises(ValueError, match="N_parts"):
        sampled_lognormals.build(cfg(counts))


@pytest.mark.parametrize("counts,expected", [(2, 2), (3., 3), ("4.0", 4),
    ([2., 3.], 5), (("2", "3.0"), 5), (np.array([2, 3]), 5),
    (np.array(4.), 4)])
def test_integral_counts_preserve_exact_sample_total_and_input(counts, expected):
    config = cfg(counts)
    before = deepcopy(config)
    pop = sampled_lognormals.build(config)
    assert len(pop.ids) == expected
    assert pop.get_Ntot() == pytest.approx(3.)
    np.testing.assert_array_equal(config["N_parts"], before["N_parts"])
    assert set(config) == set(before)


def test_default_count_and_generator_counts():
    config = cfg(1)
    config.pop("N_parts")
    assert len(sampled_lognormals.build(config).ids) == 100
    assert len(sampled_lognormals.build(cfg(iter([2., 3.]))).ids) == 5


@pytest.mark.parametrize("counts", [np.complex128(3 + 1j), np.complex64(3 + 1j),
    np.complex128(3 + 0j), [2, np.complex128(3 + 1j)], np.array(3 + 1j)])
def test_complex_counts_cannot_discard_imaginary_components(counts):
    with pytest.raises(ValueError, match="N_parts"):
        sampled_lognormals.build(cfg(counts))
