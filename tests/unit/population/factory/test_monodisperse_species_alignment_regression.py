import numpy as np
import pytest
from part2pop.population.builder import build_population


def test_permuted_species_order_is_aligned_before_particle_construction():
    cfg = {
        "type": "monodisperse", "N": [1., 1.], "D": [1e-7, 1e-7],
        "aero_spec_names": [["SO4", "OC"], ["OC", "SO4"]],
        "aero_spec_fracs": [[.9, .1], [.8, .2]],
    }
    population = build_population(cfg)
    names = [s.name for s in population.species]
    masses = population.spec_masses
    for i, expected in enumerate((.9, .2)):
        dry = masses[i, names.index("SO4")] + masses[i, names.index("OC")]
        assert masses[i, names.index("SO4")] / dry == pytest.approx(expected)


def test_shared_species_row_remains_compatible():
    pop = build_population({"type":"monodisperse", "N":[1., 2.],
                            "D":[1e-7, 2e-7], "aero_spec_names":[["SO4"]],
                            "aero_spec_fracs":[[1.], [1.]]})
    assert len(pop.ids) == 2


def test_species_row_length_mismatch_fails_early():
    with pytest.raises(ValueError, match="mismatched"):
        build_population({"type":"monodisperse", "N":[1.], "D":[1e-7],
                          "aero_spec_names":[["SO4", "OC"]],
                          "aero_spec_fracs":[[1.]]})


def test_dry_diameter_and_water_fractions_with_water_first_and_missing_water_row():
    cfg = {"type": "monodisperse", "N": [1.0, 1.0], "D": [100e-9, 100e-9],
           "aero_spec_names": [["H2O", "SO4"], ["SO4"]],
           "aero_spec_fracs": [[0.5, 0.5], [1.0]], "D_is_wet": False}
    pop = build_population(cfg)
    for ident, water_fraction in zip(pop.ids, [0.5, 0.0]):
        particle = pop.get_particle(ident)
        assert np.all(np.isfinite(particle.masses))
        assert particle.get_Ddry() == pytest.approx(100e-9, rel=1e-12, abs=0.0)
        water_idx = next(i for i, spec in enumerate(particle.species) if spec.name == "H2O")
        assert particle.masses[water_idx] / particle.masses.sum() == pytest.approx(water_fraction)
