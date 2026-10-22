from types import SimpleNamespace

import numpy as np
import pytest

from part2pop import make_particle
from part2pop.population.base import ParticlePopulation
from part2pop.optics.base import OpticalParticle
from part2pop.optics import builder


class RecordingParticle(OpticalParticle):
    def __init__(self, particle, config):
        super().__init__(particle, config)
        self.Cext[:] = self.species[0].refractive_index.real_ri_fun(self.wvl_grid)

    def compute_optics(self):
        pass


@pytest.fixture
def optics(monkeypatch):
    monkeypatch.setattr(builder, "discover_morphology_types", lambda: {"recording": RecordingParticle})
    return {"type": "recording", "wvl_grid": [550e-9], "rh_grid": [0.0]}


def test_direct_particle_omission_reuses_but_empty_mapping_resets(optics):
    particle = make_particle(100e-9, ["BC"], [1.0])
    override = {"BC": {"n_550": 2.4}}
    builder.build_optical_particle(particle, dict(optics, species_modifications=override))
    inherited = builder.build_optical_particle(particle, optics)
    assert inherited.Cext.item() == pytest.approx(2.4)
    reset = builder.build_optical_particle(particle, dict(optics, species_modifications={}))
    assert reset.Cext.item() == pytest.approx(1.82)
    changed = builder.build_optical_particle(
        particle, dict(optics, species_modifications={"BC": {"n_550": 2.1}}))
    assert changed.Cext.item() == pytest.approx(2.1)


def test_explicit_empty_mapping_resets_untracked_custom_ri(optics):
    particle = make_particle(100e-9, ["BC"], [1.0])
    particle.species[0].refractive_index = SimpleNamespace(
        wvls=[550e-9], real_ri_fun=lambda lam: np.full_like(lam, 2.4),
        imag_ri_fun=lambda lam: np.zeros_like(lam), RI_params=None)
    reset = builder.build_optical_particle(particle, dict(optics, species_modifications={}))
    assert reset.Cext.item() == pytest.approx(1.82)


def test_population_omission_empty_and_override_match_direct_rules(optics):
    particle = make_particle(100e-9, ["BC"], [1.0])
    mods = {"BC": {"n_550": 2.4}}
    pop = ParticlePopulation(tuple(particle.species), np.array([particle.masses]),
                             np.array([1.0]), [1], species_modifications=mods)
    for cfg, expected in [(optics, 2.4), (dict(optics, species_modifications={}), 1.82),
                          (dict(optics, species_modifications={"BC": {"n_550": 2.1}}), 2.1),
                          (optics, 2.4)]:
        result = builder.build_optical_population(pop, cfg)
        assert result.Cext.item() == pytest.approx(expected)
    assert pop.species_modifications == mods
    assert "species_modifications" not in optics
