import math

import pytest

import part2pop.optics.factory.homogeneous as homogeneous
from part2pop.optics.factory.homogeneous import HomogeneousParticle
from part2pop.population.builder import build_population


def _base_particle(diameter_m=0.20e-6):
    population = build_population(
        {
            "type": "monodisperse",
            "aero_spec_names": [["BC", "SO4", "H2O"]],
            "N": [1.0e6],
            "D": [diameter_m],
            "aero_spec_fracs": [[0.2, 0.7, 0.1]],
        }
    )
    return population.get_particle(population.ids[0])


@pytest.mark.skipif(
    homogeneous.MieQ is None,
    reason=f"PyMieScatt not available: {homogeneous._PMS_ERR}",
)
def test_rayleigh_matches_mie_in_small_particle_limit():
    refractive_index = complex(1.5, 0.0)
    wavelength_m = 550e-9
    radius_m = 5e-9

    cext, csca, cabs, g = HomogeneousParticle._rayleigh_cross_sections(
        m=refractive_index,
        wavelength_m=wavelength_m,
        radius_m=radius_m,
    )

    diameter_nm = 2.0 * radius_m * 1e9
    wavelength_nm = wavelength_m * 1e9
    mie = homogeneous.MieQ(
        refractive_index,
        wavelength_nm,
        diameter_nm,
        asDict=True,
        asCrossSection=False,
    )
    geometric_area = math.pi * radius_m**2

    assert csca == pytest.approx(mie["Qsca"] * geometric_area, rel=0.02, abs=0.0)
    assert cext == pytest.approx(mie["Qext"] * geometric_area, rel=0.02, abs=0.0)
    assert cabs == pytest.approx(0.0, abs=1e-30)
    assert g == pytest.approx(0.0)


@pytest.mark.skipif(
    homogeneous.MieQ is None,
    reason=f"PyMieScatt not available: {homogeneous._PMS_ERR}",
)
def test_rayleigh_matches_mie_for_absorbing_particle_in_small_particle_limit():
    refractive_index = complex(1.5, 0.02)
    wavelength_m = 550e-9
    radius_m = 5e-9

    cext, csca, cabs, g = HomogeneousParticle._rayleigh_cross_sections(
        m=refractive_index,
        wavelength_m=wavelength_m,
        radius_m=radius_m,
    )

    diameter_nm = 2.0 * radius_m * 1e9
    wavelength_nm = wavelength_m * 1e9
    mie = homogeneous.MieQ(
        refractive_index,
        wavelength_nm,
        diameter_nm,
        asDict=True,
        asCrossSection=False,
    )
    geometric_area = math.pi * radius_m**2

    assert csca == pytest.approx(mie["Qsca"] * geometric_area, rel=0.02, abs=0.0)
    assert cabs == pytest.approx(mie["Qabs"] * geometric_area, rel=0.02, abs=0.0)
    assert cext == pytest.approx(mie["Qext"] * geometric_area, rel=0.02, abs=0.0)
    assert g == pytest.approx(0.0)


def test_rayleigh_fallback_rejects_large_size_parameter(monkeypatch):
    monkeypatch.setattr(homogeneous, "MieQ", None)

    with pytest.warns(RuntimeWarning, match="Rayleigh-sphere approximation"):
        with pytest.raises(
            ValueError,
            match=r"outside its supported small-particle regime: x=.*must be < 0\.1",
        ):
            HomogeneousParticle(
                _base_particle(),
                {
                    "wvl_grid": [550e-9],
                    "rh_grid": [0.0],
                    "allow_rayleigh_fallback": True,
                },
            )
