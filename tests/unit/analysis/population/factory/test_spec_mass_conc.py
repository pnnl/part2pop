import numpy as np

from part2pop.analysis.population.factory.spec_mass_conc import SpecMassConc, build


def test_build_returns_spec_mass_conc():
    var = build({})
    assert isinstance(var, SpecMassConc)
    assert var.meta.axis_names == ("species",)


def test_compute_uses_population_species(simple_population):
    var = build({})
    out = var.compute(simple_population)
    expected = np.array([simple_population.get_mass_conc(spec.name) for spec in simple_population.species])
    assert np.allclose(out, expected)


def test_compute_with_specific_species(simple_population):
    var = build({"species_names": "SO4"})
    out = var.compute(simple_population, as_dict=True)
    assert out["mass_conc"].shape == (1,)
    assert np.isclose(out["mass_conc"][0], simple_population.get_mass_conc("SO4"))
