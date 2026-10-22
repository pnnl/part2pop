import numpy as np
import pytest
from part2pop.freezing.builder import build_freezing_particle, build_freezing_population

from part2pop import AerosolSpecies, Particle, ParticlePopulation, get_species
from part2pop.freezing.factory.utils import calculate_Psat


@pytest.mark.parametrize("tabulated", [False, True])
@pytest.mark.parametrize("disabled", ["m_log10_Jhet", "b_log10_Jhet", "both"])
def test_disabled_material_does_not_enlarge_remaining_active_area(tabulated, disabled):
    material = (
        get_species("OC", None)
        if tabulated
        else AerosolSpecies("custom_disabled", density=2000, kappa=0.1, molar_mass=0.1)
    )
    specs = (material, get_species("BC", None), get_species("H2O", None))
    # Both NaN OC coefficients also classify OC as a solute under the
    # retained get_aw rule. Keep this geometry control dilute enough that it
    # remains ice supersaturated, including when PR #123's cutoff is present.
    particle = Particle(
        specs, np.array([1e-21, 1e-21, 1e-18]) * [s.density for s in specs]
    )
    coefficients = {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}
    for name in coefficients:
        if disabled in (name, "both"):
            coefficients[name] = np.nan
    cfg = {
        "morphology": "homogeneous",
        "species_modifications": {
            material.name: coefficients,
            "BC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 12.0},
        },
    }
    frozen = build_freezing_particle(particle, cfg)
    expected_area = 4 * np.pi * (3e-21 / (4 * np.pi)) ** (2 / 3)
    assert frozen.INSA == pytest.approx(expected_area, rel=1e-12, abs=0)
    vapor_pressure, ice_pressure = calculate_Psat(260.0)
    assert frozen.get_aw() > ice_pressure / vapor_pressure
    assert frozen.compute_Jhet(260.0) == pytest.approx(1e12)
    assert frozen.compute_Jhom(260.0) == 0.0
    population = ParticlePopulation(
        specs, np.array([particle.masses]), np.array([1.0]), [0]
    )
    freezing_population = build_freezing_population(population, cfg)
    assert freezing_population.get_freezing_probs(260.0, cfg)[0] == pytest.approx(
        1 - np.exp(-1e12 * expected_area), rel=1e-12, abs=0
    )


def test_all_tabulated_material_disabled_has_no_area():
    specs = (get_species("OC", None), get_species("H2O", None))
    particle = Particle(specs, [1e-18, 1e-17])
    frozen = build_freezing_particle(
        particle,
        {
            "morphology": "homogeneous",
            "species_modifications": {
                "OC": {"m_log10_Jhet": np.nan, "b_log10_Jhet": np.nan}
            },
        },
    )
    assert frozen.INSA == 0.0


@pytest.mark.parametrize("zero_pair", [False, True])
def test_finite_tabulated_coefficients_keep_active_volume(zero_pair):
    specs = (get_species("OC", None), get_species("H2O", None))
    particle = Particle(specs, [1e-18, 1e-17])
    cfg = (
        {"species_modifications": {"OC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 0.0}}}
        if zero_pair
        else {}
    )
    cfg["morphology"] = "homogeneous"
    frozen = build_freezing_particle(particle, cfg)
    area = (
        4 * np.pi * (3 * particle.masses[0] / specs[0].density / (4 * np.pi)) ** (2 / 3)
    )
    assert frozen.INSA == pytest.approx(area, rel=1e-12, abs=0)
