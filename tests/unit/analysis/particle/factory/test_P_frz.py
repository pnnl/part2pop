import pytest

from part2pop.analysis.particle.factory import P_frz
from part2pop.freezing import build_freezing_population
from part2pop.population import build_population


def test_p_frz_requires_temperature_and_computes_probabilities():
    pop_config = {
        "type": "monodisperse",
        "morphology": "homogeneous",
        "N": [1e9],
        "D": [1e-6],
        "aero_spec_names": [["OC"]],
        "aero_spec_fracs": [[1.0]],
    }
    particle_pop = build_population(pop_config)
    frz_pop = build_freezing_population(particle_pop, pop_config)

    with pytest.raises(ValueError):
        P_frz.build({"T": None}).compute_all(frz_pop)

    probs = P_frz.build({"T": 250.0}).compute_all(frz_pop)
    assert (probs > 0).all()
