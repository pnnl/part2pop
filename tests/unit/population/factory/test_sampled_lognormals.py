import numpy as np
import pytest

from part2pop.population.factory import sampled_lognormals


def _base_config():
    return {
        "N": [1.0, 1.0],
        "GMD": [1e-6, 2e-6],
        "GSD": [1.2, 1.3],
        "aero_spec_names": [["SO4"], ["SO4"]],
        "aero_spec_fracs": [[1.0], [1.0]],
    }


def test_scalar_n_parts_distributed_evenly(monkeypatch):
    config = _base_config()
    config["N_parts"] = 10

    rng = np.random.default_rng(0)
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, size: np.full(size, loc))
    mock_make_particle_calls = []

    def fake_make_particle(D, species, fracs, **kwargs):
        mock_make_particle_calls.append((D, tuple(fracs)))
        class Dummy:
            def __init__(self):
                self.species = species
                self.masses = np.ones(len(fracs))

            def get_Dwet(self):
                return D

        return Dummy()

    monkeypatch.setattr(sampled_lognormals, "make_particle", fake_make_particle)

    pop = sampled_lognormals.build(config)
    assert pop.num_concs.size == 10
    assert sum(pop.num_concs) == pytest.approx(2.0)
    assert len(mock_make_particle_calls) == 10


def test_n_parts_scalar_nonpositive_raises():
    config = _base_config()
    config["N_parts"] = 0
    with pytest.raises(ValueError):
        sampled_lognormals.build(config)


def test_inconsistent_length_raises():
    config = _base_config()
    config["aero_spec_names"] = [["SO4"], ["SO4"], ["SO4"]]
    with pytest.raises(ValueError):
        sampled_lognormals.build(config)