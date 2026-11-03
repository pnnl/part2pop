"""Unexpected data I/O errors must not become inactive-material results."""
import numpy as np
import pytest

from part2pop import make_particle
from part2pop.freezing.builder import build_freezing_particle
from part2pop.freezing.factory import homogeneous


def test_active_area_does_not_swallow_data_io_error(monkeypatch):
    particle = make_particle(1e-7, ["BC"], [1.])
    def unreadable(*args, **kwargs):
        raise OSError("freezing table is unreadable")
    monkeypatch.setattr(homogeneous, "retrieve_Jhet_val", unreadable)
    with pytest.raises(OSError, match="freezing table is unreadable"):
        build_freezing_particle(particle, {"morphology": "homogeneous"})


@pytest.mark.parametrize("both", [False, True])
def test_documented_legacy_solute_classification(both):
    particle = make_particle(1e-7, ["OC", "H2O"], [.5, .5])
    coefficients = {"m_log10_Jhet": np.nan}
    if both:
        coefficients["b_log10_Jhet"] = np.nan
    frozen = build_freezing_particle(particle, {
        "morphology": "homogeneous", "species_modifications": {"OC": coefficients}})
    assert frozen.INSA == 0
    expected = (np.exp(-.01801528 * particle.masses[0]
                       / particle.species[0].molar_mass / particle.masses[1])
                if both else 1.)
    assert frozen.get_aw() == pytest.approx(expected, rel=1e-12, abs=0)
