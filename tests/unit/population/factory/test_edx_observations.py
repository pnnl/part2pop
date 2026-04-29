import numpy as np

from part2pop import ParticlePopulation, build_population
from part2pop.population.factory.helpers.assembly import (
    assemble_population_from_mass_fractions as real_assemble_population_from_mass_fractions,
)
from part2pop.population.factory import edx_observations


def _write_edx_csv(tmp_path):
    csv_path = tmp_path / "edx_small.csv"
    # Keep columns aligned with current reader expectations:
    # - element columns by direct names
    # - one diameter column containing "diam"
    # - one class/type column containing "class"
    csv_path.write_text(
        "diam_um,class,C,N,O,Na,Mg,Al,Si,P,S,Cl,K,Ca,Mn,Fe,Zn\n"
        # biological row tuned to current mapping logic:
        # - sums to 1.0
        # - keeps Al/Si near zero so OIN contribution stays small
        # - leaves substantial C/N/O/P/S pool for biological fraction
        "0.5,biological,0.35,0.12,0.33,0.04,0.04,0.00,0.00,0.04,0.04,0.04,0.00,0.00,0.00,0.00,0.00\n"
        "",
        encoding="utf-8",
    )
    return csv_path


def test_edx_observations_smoke_build_population(tmp_path):
    edx_csv = _write_edx_csv(tmp_path)
    cfg = {
        "type": "edx_observations",
        "edx_file": str(edx_csv),
    }

    pop = build_population(cfg)
    assert isinstance(pop, ParticlePopulation)

    # Characterize current behavior only
    assert hasattr(pop, "classes")
    assert len(pop.classes) == len(pop.ids)
    assert len(pop.ids) == 1
    assert len(pop.classes) == 1

    # Number concentrations and diameters should be positive/plausible
    assert np.all(pop.num_concs > 0)
    d = pop.get_particle_var("Ddry")
    assert np.all(np.isfinite(d))
    assert np.all(d > 0)


def test_edx_observations_calls_assembly_helper_directly(tmp_path, monkeypatch):
    edx_csv = _write_edx_csv(tmp_path)
    cfg = {
        "type": "edx_observations",
        "edx_file": str(edx_csv),
    }

    captured = {"count": 0, "kwargs": None}

    def _capture_and_build(**kwargs):
        captured["count"] += 1
        captured["kwargs"] = kwargs
        return real_assemble_population_from_mass_fractions(**kwargs)

    monkeypatch.setattr(
        edx_observations,
        "assemble_population_from_mass_fractions",
        _capture_and_build,
    )

    pop = build_population(cfg)

    assert captured["count"] == 1
    assert isinstance(pop, ParticlePopulation)
    assert pop.spec_masses.ndim == 2

    kwargs = captured["kwargs"]
    diameters = kwargs["diameters"]
    number_concentrations = kwargs["number_concentrations"]
    species_names = kwargs["species_names"]
    mass_fractions = kwargs["mass_fractions"]
    classes = kwargs["classes"]

    assert len(diameters) == len(number_concentrations)
    assert len(species_names) == len(diameters)
    assert len(mass_fractions) == len(diameters)
    assert classes is not None
    assert len(classes) == len(diameters)
