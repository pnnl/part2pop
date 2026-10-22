import numpy as np
import pytest
from part2pop.aerosol_particle import make_particle


@pytest.mark.parametrize("names,fracs", [
    (["SO4", "H2O"], [.5, .5]),
    (["H2O", "SO4"], [.5, .5]),
    (["OC", "H2O", "SO4"], [.2, .5, .3]),
])
def test_dry_diameter_and_requested_water_mass_fraction(names, fracs):
    d = 140e-9
    p = make_particle(d, names, fracs, D_is_wet=False)
    assert p.get_Ddry() == pytest.approx(d, rel=1e-10)
    water = next(i for i, s in enumerate(p.species) if s.name == 'H2O')
    assert p.masses[water] / p.masses.sum() == pytest.approx(fracs[names.index('H2O')])


def test_dry_diameter_with_no_water_remains_unchanged():
    p = make_particle(100e-9, ["SO4"], [1.0], D_is_wet=False)
    assert p.get_Ddry() == pytest.approx(100e-9)
    assert p.get_mass_h2o() == 0


def test_pure_water_invalid_as_dry_particle():
    with pytest.raises(ValueError, match="positive dry mass fraction"):
        make_particle(100e-9, ["H2O"], [1.0], D_is_wet=False)
