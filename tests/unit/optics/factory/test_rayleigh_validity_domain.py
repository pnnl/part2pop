"""Enforce the proposed fallback domain and compare its boundaries with Mie."""
import math
import numpy as np
import pytest

import part2pop.optics.factory.homogeneous as homogeneous
from part2pop.population.builder import build_population


@pytest.mark.parametrize("m", [.99 + 0j, 2.01 + 0j, 1.5 - .001j,
    1.5 + 1.01j, 10 + .1j, 30 + .1j, complex(np.nan, 0), complex(1.5, np.inf)])
def test_fallback_constructor_rejects_unvalidated_indices(monkeypatch, m):
    pop = build_population({"type": "monodisperse", "N": [1.], "D": [1e-8],
        "aero_spec_names": [["BC"]], "aero_spec_fracs": [[1.]]})
    monkeypatch.setattr(homogeneous, "MieQ", None)
    with pytest.warns(RuntimeWarning, match="Rayleigh-sphere approximation"):
        with pytest.raises(ValueError, match="Rayleigh fallback refractive index"):
            homogeneous.HomogeneousParticle(pop.get_particle(0), {
                "wvl_grid": [550e-9], "rh_grid": [0.], "allow_rayleigh_fallback": True,
                "species_modifications": {"BC": {"n_550": m.real, "k_550": m.imag,
                                                  "alpha_n": 0., "alpha_k": 0.}}})


@pytest.mark.parametrize("radius,wavelength", [(-1., 550e-9), (np.inf, 550e-9),
    (1e-9, 0.), (1e-9, -1.), (1e-9, np.nan)])
def test_invalid_geometry_is_rejected(radius, wavelength):
    with pytest.raises(ValueError, match="Rayleigh fallback geometry"):
        homogeneous.HomogeneousParticle._validate_rayleigh_domain(1.5 + .02j, wavelength, radius)


@pytest.mark.parametrize("m", [1+0j, 1+1j, 2+0j, 2+1j])
def test_inclusive_index_boundaries_and_zero_radius_are_valid(m):
    homogeneous.HomogeneousParticle._validate_rayleigh_domain(m, 550e-9, 0.)
    homogeneous.HomogeneousParticle._validate_rayleigh_domain(
        m, 550e-9, .099999 * 550e-9 / (2 * math.pi))


@pytest.mark.skipif(homogeneous.MieQ is None, reason="Mie reference solver unavailable")
@pytest.mark.parametrize("x", [.03, .099999])
@pytest.mark.parametrize("n", [1., 1.2, 1.5, 2.])
@pytest.mark.parametrize("k", [0., 1e-6, .02, .3, 1.])
def test_rayleigh_domain_boundaries_and_interior_match_mie(x, n, k):
    m = complex(n, k)
    wavelength = 550e-9
    radius = x * wavelength / (2 * math.pi)
    homogeneous.HomogeneousParticle._validate_rayleigh_domain(m, wavelength, radius)
    ext, scat, absorption, _ = homogeneous.HomogeneousParticle._rayleigh_cross_sections(
        m, wavelength, radius)
    mie = homogeneous.MieQ(m, 550., 2 * radius * 1e9, asDict=True)
    area = math.pi * radius ** 2
    noise = 128 * np.finfo(float).eps * area
    if m == 1 + 0j:
        assert ext == scat == absorption == 0.
        assert abs(mie["Qext"] * area) <= noise
        assert abs(mie["Qsca"] * area) <= noise
    else:
        assert scat == pytest.approx(mie["Qsca"] * area, rel=.02, abs=0)
        assert ext == pytest.approx(mie["Qext"] * area, rel=.02, abs=0)
    if k == 0:
        assert absorption == 0.
        assert abs(mie["Qabs"] * area) <= noise
    else:
        assert absorption == pytest.approx(mie["Qabs"] * area, rel=.02, abs=0)


def test_mie_path_does_not_apply_fallback_index_limits(monkeypatch):
    pop = build_population({"type": "monodisperse", "N": [1.], "D": [1e-8],
        "aero_spec_names": [["BC"]], "aero_spec_fracs": [[1.]]})
    called = []
    def mie(m, *args, **kwargs):
        called.append(m)
        return {"Qext": 3., "Qsca": 1., "Qabs": 2., "g": .1}
    monkeypatch.setattr(homogeneous, "MieQ", mie)
    homogeneous.HomogeneousParticle(pop.get_particle(0), {
        "wvl_grid": [550e-9], "rh_grid": [0.],
        "species_modifications": {"BC": {"n_550": 30., "k_550": .1,
                                          "alpha_n": 0., "alpha_k": 0.}}})
    assert called == [30 + .1j]


@pytest.mark.parametrize("n,k", [(1., 0.), (1., 1.), (2., 0.), (2., 1.)])
@pytest.mark.parametrize("fraction", [.1, .9762437057077041])
def test_boundary_mixtures_survive_roundoff_in_effective_index(monkeypatch, n, k, fraction):
    # Both components are exactly on the supported boundary. Their volume
    # averaging can round the effective index a few ULPs outside that boundary.
    pop = build_population({"type": "monodisperse", "N": [1.], "D": [1e-8],
        "aero_spec_names": [["BC", "SO4"]],
        "aero_spec_fracs": [[fraction, 1. - fraction]]})
    monkeypatch.setattr(homogeneous, "MieQ", None)
    mods = {name: {"n_550": n, "k_550": k, "alpha_n": 0., "alpha_k": 0.}
            for name in ["BC", "SO4"]}
    with pytest.warns(RuntimeWarning, match="Rayleigh-sphere approximation"):
        result = homogeneous.HomogeneousParticle(pop.get_particle(0), {
            "wvl_grid": [550e-9], "rh_grid": [0.],
            "allow_rayleigh_fallback": True, "species_modifications": mods})
    assert np.all(np.isfinite(result.Cext))
    assert np.all(result.Cext >= 0)


@pytest.mark.parametrize("m", [complex(1. - 1e-10, 0.), complex(2. + 1e-10, 0.),
    complex(1.5, 1. + 1e-10), complex(1.5, -1e-20)])
def test_roundoff_allowance_does_not_expand_physical_input_domain(m):
    with pytest.raises(ValueError, match="Rayleigh fallback refractive index"):
        homogeneous.HomogeneousParticle._validate_rayleigh_domain(m, 550e-9, 1e-9)
