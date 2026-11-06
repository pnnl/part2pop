from dataclasses import replace

import numpy as np
import pytest

from part2pop import ParticlePopulation, make_particle
from part2pop.analysis.builder import build_variable
from part2pop.analysis.population.factory.dNdlnD import DNdlnDVar, build


@pytest.fixture(autouse=True)
def isolate_template(monkeypatch):
    monkeypatch.setattr(
        DNdlnDVar,
        "meta",
        replace(
            DNdlnDVar.meta,
            units="m$^{-3}$",
            long_label="Number size distribution",
            short_label=r"$dN/d\ln D$",
        ),
    )


@pytest.mark.parametrize("through_builder", [False, True])
@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("setting", ["normalize", "dry"])
def test_diameter_metadata_is_local_in_both_construction_orders(
    through_builder, changed_first, setting
):
    def make(cfg):
        return build_variable("dNdlnD", var_cfg=cfg) if through_builder else build(cfg)

    changed_cfg = {"normalize": True} if setting == "normalize" else {"wetsize": False}
    if changed_first:
        changed = make(changed_cfg)
        raw = make({"normalize": False, "wetsize": True})
    else:
        raw = make({"normalize": False, "wetsize": True})
        changed = make(changed_cfg)
    assert raw.meta is not changed.meta
    assert raw.meta is not DNdlnDVar.meta
    assert changed.meta is not DNdlnDVar.meta
    assert raw.meta.units == DNdlnDVar.meta.units == "m$^{-3}$"
    assert (
        raw.meta.long_label == DNdlnDVar.meta.long_label == "Number size distribution"
    )
    assert raw.meta.short_label == DNdlnDVar.meta.short_label == r"$dN/d\ln D$"
    if setting == "normalize":
        assert changed.meta.units == ""
    else:
        assert changed.meta.long_label == "Dry number size distribution"
        assert changed.meta.short_label == r"$dN/d\ln D_{\rm dry}$"


@pytest.mark.parametrize("method", ["hist", "kde"])
def test_constructing_other_metadata_preserves_raw_distribution_values(method):
    particle = make_particle(100e-9, ["SO4", "H2O"], [0.1, 0.9])
    pop = ParticlePopulation(
        particle.species, np.array([particle.masses]), np.array([100.0]), [0]
    )
    raw = build({"method": method, "edges": np.geomspace(1e-8, 1e-6, 10)})
    before = raw.compute(pop)
    build({"normalize": True, "wetsize": False})
    np.testing.assert_array_equal(raw.compute(pop), before)
    assert raw.meta.units == "m$^{-3}$"
    assert raw.meta.long_label == "Number size distribution"
    if method == "hist":
        assert np.sum(before * np.diff(np.log(raw.cfg["edges"]))) == pytest.approx(
            100.0
        )
