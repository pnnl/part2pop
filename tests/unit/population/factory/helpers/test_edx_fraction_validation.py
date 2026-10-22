import numpy as np
import pandas as pd
import pytest

from part2pop.population.factory.helpers import edx

ELEMENTS = np.array(["C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Mn", "Fe", "Cu", "Zn"])
SPECIES = ["SO4", "OIN", "OC", "Na", "Cl", "biological"]
SAMPLERS = [edx.sample_particle, edx.sample_bio_particle, edx.sample_carbonaceous_particle]


@pytest.mark.parametrize("sample", SAMPLERS)
@pytest.mark.parametrize("bad", [-.1, np.nan, np.inf, -np.inf])
def test_invalid_element_is_rejected_before_reconstruction(sample, bad):
    row = np.zeros(len(ELEMENTS))
    row[0], row[3] = 1.1, bad  # Negative Na used to survive with total one.
    with pytest.raises(ValueError, match="finite and nonnegative"):
        sample(SPECIES, row, ELEMENTS)


@pytest.mark.parametrize("sample", SAMPLERS)
def test_zero_row_is_not_promoted_to_a_composition(sample):
    with pytest.raises(ValueError, match="sum to"):
        sample(SPECIES, np.zeros(len(ELEMENTS)), ELEMENTS)


@pytest.mark.parametrize("bad", [-.1, np.nan, np.inf])
def test_csv_invalid_values_fail_before_percent_inference(monkeypatch, bad):
    data = pd.DataFrame({"diameter": [1.], "class": ["inorganics"],
                         **{name: [0.] for name in ELEMENTS}})
    data["C"], data["Na"] = 1.1, bad
    monkeypatch.setattr(pd, "read_csv", lambda path: data)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        edx.read_edx_file({"edx_file": "input.csv"}, list(ELEMENTS))


@pytest.mark.parametrize("percent", [False, True])
def test_valid_fraction_and_percent_csv_retain_existing_composition(monkeypatch, percent):
    scale = 100. if percent else 1.
    data = pd.DataFrame({"diameter": [1.], "class": ["inorganics"],
                         **{name: [0.] for name in ELEMENTS}})
    data["Na"], data["Cl"] = .4 * scale, .6 * scale
    monkeypatch.setattr(pd, "read_csv", lambda path: data)
    raw = edx.read_edx_file({"edx_file": "input.csv"}, list(ELEMENTS))
    result = edx.sample_particle(SPECIES, raw.mass_fractions[0], raw.elements)
    np.testing.assert_allclose(result, [0., 0., 0., .4, .6, 0.], rtol=1e-12, atol=0)
