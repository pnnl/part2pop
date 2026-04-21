# tests/unit/freezing/factory/test_homogeneous.py

import numpy as np
import pytest
from part2pop.aerosol_particle import make_particle
from part2pop.freezing.factory.homogeneous import HomogeneousParticle, build


def _make_droplet_with_inp():
    """
    Construct a physically reasonable wet particle using the public
    `make_particle` helper, rather than manually instantiating Particle.

    Returns
    -------
    base_particle : part2pop.aerosol_particle.Particle
    cfg : dict
        Configuration dict passed to HomogeneousParticle. Currently only
        `species_modifications` is used by the model.
    """
    # 0.5 µm wet diameter, 10% insoluble OC, 90% water by mass.
    D_wet = 0.5e-6
    aero_spec_names = ["BC", "H2O"]
    aero_spec_frac = [0.1, 0.9]

    base_particle = make_particle(
        D=D_wet,
        aero_spec_names=aero_spec_names,
        aero_spec_frac=aero_spec_frac,
        D_is_wet=True,
    )

    # Minimal config that is consistent with the implementation:
    cfg = {
        "species_modifications": {},
    }
    return base_particle, cfg

def _make_droplet_without_inp():
    """
    Construct a physically reasonable wet particle using the public
    `make_particle` helper, rather than manually instantiating Particle.

    Returns
    -------
    base_particle : part2pop.aerosol_particle.Particle
    cfg : dict
        Configuration dict passed to HomogeneousParticle. Currently only
        `species_modifications` is used by the model.
    """
    # 0.5 µm wet diameter, 10% soluble SO4, 90% water by mass.
    D_wet = 0.5e-6
    aero_spec_names = ["SO4", "H2O"]
    aero_spec_frac = [0.05, 0.95]

    base_particle = make_particle(
        D=D_wet,
        aero_spec_names=aero_spec_names,
        aero_spec_frac=aero_spec_frac,
        D_is_wet=True,
    )

    # Minimal config that is consistent with the implementation:
    cfg = {
        "species_modifications": {},
    }
    return base_particle, cfg


def test_homogeneous_particle_Jhet_positive():
    """
    J_het for a physically reasonable supercooled droplet should be > 0.
    """
    base_particle, cfg = _make_droplet_with_inp()
    fpart = HomogeneousParticle(base_particle, cfg)

    T = np.array([233.15])  # K, clearly below freezing
    Jhet = fpart.get_Jhet(T)

    # Finite and strictly positive
    assert np.isfinite(Jhet)
    assert Jhet > 0.0

def test_homogeneous_particle_Jhom_positive():
    """
    J_hom for a physically reasonable supercooled droplet should be > 0.
    """
    base_particle, cfg = _make_droplet_without_inp()
    fpart = HomogeneousParticle(base_particle, cfg)

    T = np.array([233.15])  # K, clearly below freezing
    Jhom = fpart.get_Jhom(T)

    # Finite and strictly positive
    assert np.isfinite(Jhom)
    assert Jhom > 0.0


def test_homogeneous_Jhet_temperature_sensitivity():
    """
    At colder temperatures, the homogeneous freezing rate J_het should
    increase compared to warmer (but still subfreezing) temperatures.
    """
    base_particle, cfg = _make_droplet_with_inp()
    fpart = HomogeneousParticle(base_particle, cfg)
    T = np.array([228.15, 238.15])  # K
    Jhet = fpart.get_Jhet(T)
    J_cold = Jhet[0]
    J_warm = Jhet[1]

    # Both positive, and colder temperature gives larger rate.
    assert J_cold > 0.0
    assert J_warm > 0.0
    assert J_cold > J_warm

def test_homogeneous_Jhom_temperature_sensitivity():
    """
    At colder temperatures, the homogeneous freezing rate J_het should
    increase compared to warmer (but still subfreezing) temperatures.
    """
    base_particle, cfg = _make_droplet_without_inp()
    fpart = HomogeneousParticle(base_particle, cfg)
    T = np.array([228.15, 238.15])  # K
    Jhom = fpart.get_Jhom(T)
    J_cold = Jhom[0]
    J_warm = Jhom[1]

    # Both positive, and colder temperature gives larger rate.
    assert J_cold > 0.0
    assert J_warm > 0.0
    assert J_cold > J_warm


def test_homogeneous_build_wrapper_returns_equivalent_object():
    """
    The factory-level `build` wrapper should construct a HomogeneousParticle
    equivalent to calling the constructor directly.
    """
    base_particle, cfg = _make_droplet_with_inp()

    direct = HomogeneousParticle(base_particle, cfg)

    # `build` expects a model_config with `model_name` and `model_kwargs`.
    model_config = {
        "model_name": "homogeneous",
        "model_kwargs": cfg,
    }
    built = build(base_particle, model_config)

    assert isinstance(built, HomogeneousParticle)

    # Check that the key internal fields are numerically identical.
    np.testing.assert_allclose(built.INSA, direct.INSA)
    np.testing.assert_allclose(built.m_log10_Jhet, direct.m_log10_Jhet)
    np.testing.assert_allclose(built.b_log10_Jhet, direct.b_log10_Jhet)

def test_particle_aws():
    base_particle, cfg = _make_droplet_without_inp()
    fpart = HomogeneousParticle(base_particle, cfg)
    T = np.array([228])  # K
    Jhom = fpart.get_Jhom(T)
    assert Jhom[0] > 0

    base_particle.masses[base_particle.idx_h2o()]=0.0
    with pytest.raises(ValueError):
        Jhom = fpart.get_Jhom(T)
