# tests/unit/freezing/factory/test_homogeneous.py

import numpy as np

from pyparticle import AerosolSpecies
from pyparticle.aerosol_particle import Particle
from pyparticle.freezing.factory.homogeneous import HomogeneousParticle, build


def _make_droplet_with_insolute():
    spec_insol = AerosolSpecies("SO4")
    spec_h2o = AerosolSpecies("H2O")
    species = (spec_insol, spec_h2o)
    vols = np.array([0.1, 0.9])
    p = Particle(
        diameter=0.5e-6,
        species=species,
        spec_vol_fracs=vols,
        D_is_wet=True,
    )
    return p


def test_homogeneous_particle_Jhet_positive():
    p = _make_droplet_with_insolute()
    hp = HomogeneousParticle(p, cfg={})
    J = hp.get_Jhet(T=235.0)
    assert J >= 0.0


def test_homogeneous_build_wrapper():
    p = _make_droplet_with_insolute()
    hp = build(p, cfg={})
    assert isinstance(hp, HomogeneousParticle)
