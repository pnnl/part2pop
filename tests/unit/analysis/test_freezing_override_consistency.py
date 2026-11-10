import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle

MODULES = {
    "P_frz": "part2pop.analysis.particle.factory.P_frz",
    **{
        name: f"part2pop.analysis.population.factory.{name}"
        for name in ("avg_Jhet", "frozen_frac", "unfrozen_frac", "nucleating_sites")
    },
}


def configuration():
    return {
        "T": 250.0,
        "T_grid": [260.0, 250.0],
        "T_units": "K",
        "RH": 0.85,
        "cooling_rate": 1.0,
        "t_max": 1.0,
        "dt": 0.5,
    }


def evaluate(name, population, cfg):
    variable = importlib.import_module(MODULES[name]).build(cfg)
    if name == "P_frz":
        return variable.compute_all(population)
    return variable.compute(population)


@pytest.fixture
def recorded_configs(monkeypatch):
    recorded = []

    class FreezingResult:
        num_concs = np.array([100.0])

        def get_freezing_probs(self, temperature, cfg, dt=1.0):
            recorded.append(copy.deepcopy(cfg["species_modifications"]))
            return np.array([0.25])

        def get_avg_Jhet(self, temperature, cfg):
            recorded.append(copy.deepcopy(cfg["species_modifications"]))
            return 0.25

        def get_nucleating_sites(self, *args, **kwargs):
            # The numerical method/signature is repaired separately by PR #117.
            return np.array([0.0, 0.25])

    def builder(population, cfg):
        recorded.append(copy.deepcopy(cfg.get("species_modifications")))
        return FreezingResult()

    for path in MODULES.values():
        monkeypatch.setattr(
            importlib.import_module(path), "build_freezing_population", builder
        )
    return recorded


def lightweight_population():
    return SimpleNamespace(
        species_modifications={
            "OC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 10.0},
            "BC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 11.0},
        },
        num_concs=np.array([100.0]),
        _equilibrate_h2o=lambda rh, temperature: None,
    )


@pytest.mark.parametrize("name", MODULES)
@pytest.mark.parametrize("mode", ["omitted", "none", "empty", "explicit"])
def test_freezing_analyses_resolve_authoritative_mapping_without_mutation(
    name, mode, recorded_configs
):
    pop = lightweight_population()
    original_mapping = pop.species_modifications
    before_population = copy.deepcopy(original_mapping)
    cfg = configuration()
    if mode == "none":
        cfg["species_modifications"] = None
    elif mode == "empty":
        cfg["species_modifications"] = {}
    elif mode == "explicit":
        # Replaces the mapping: OC must not leak in from population defaults.
        cfg["species_modifications"] = {
            "BC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 12.0}
        }
    before_cfg = copy.deepcopy(cfg)
    expected = (
        before_population
        if mode in ("omitted", "none")
        else cfg["species_modifications"]
    )
    evaluate(name, pop, cfg)
    assert recorded_configs
    assert all(value == expected for value in recorded_configs)
    assert pop.species_modifications is original_mapping
    assert pop.species_modifications == before_population
    assert cfg == before_cfg


@pytest.mark.parametrize(
    "first,second",
    list(zip(MODULES, [*list(MODULES)[1:], next(iter(MODULES))])),
)
def test_calculation_specific_overrides_do_not_leak_between_analyses(
    first, second, recorded_configs
):
    pop = lightweight_population()
    expected = copy.deepcopy(pop.species_modifications)
    cfg = configuration()
    cfg["species_modifications"] = {"OC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 14.0}}
    evaluate(first, pop, cfg)
    recorded_configs.clear()
    evaluate(second, pop, configuration())
    assert recorded_configs
    assert all(value == expected for value in recorded_configs)
    assert pop.species_modifications == expected


@pytest.mark.parametrize("name", ["P_frz", "avg_Jhet", "frozen_frac", "unfrozen_frac"])
def test_real_freezing_results_recover_inherited_settings_after_an_override(name):
    particle = make_particle(100e-9, ["OC"], [1.0])
    pop = ParticlePopulation(
        particle.species,
        np.array([particle.masses]),
        np.array([100.0]),
        [0],
        species_modifications={"OC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 11.0}},
    )
    original = pop.species_modifications
    inherited = copy.deepcopy(original)
    cfg = configuration()
    reference = evaluate(name, copy.deepcopy(pop), copy.deepcopy(cfg))
    overridden_cfg = dict(
        cfg, species_modifications={"OC": {"m_log10_Jhet": 0.0, "b_log10_Jhet": 14.0}}
    )
    overridden = evaluate(name, pop, overridden_cfg)
    restored = evaluate(name, pop, cfg)
    assert not np.array_equal(overridden, reference)
    np.testing.assert_allclose(restored, reference, rtol=1e-12, atol=0.0)
    assert pop.species_modifications is original
    assert pop.species_modifications == inherited
