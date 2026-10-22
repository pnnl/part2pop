import numpy as np
import pytest

from part2pop.population.factory.helpers.edx import (
    EdxElementMassFractions,
    _DEFAULT_EDX_TARGET_SPECIES,
    _normalize_or_raise,
    read_edx_file,
    reconstruct_edx_species_mass_fractions,
    sample_bio_particle,
    sample_particle,
)


def _write_three_row_edx_csv(tmp_path):
    csv_path = tmp_path / "edx_three_rows.csv"
    csv_path.write_text(
        "diam_um,class,C,N,O,Na,Mg,Al,Si,P,S,Cl,K,Ca,Mn,Fe,Zn,Cu\n"
        "0.5,biological,0.30,0.12,0.33,0.04,0.04,0.00,0.00,0.04,0.04,0.04,0.00,0.00,0.00,0.00,0.00,0.05\n"
        # Keep synthetic row within the existing EDX mass-balance guard.
        "0.7,carbonaceous,0.30,0.10,0.34,0.05,0.03,0.02,0.02,0.03,0.03,0.05,0.01,0.005,0.005,0.005,0.005,0.00\n"
        # Keep synthetic row within the existing EDX mass-balance guard.
        "1.0,dust,0.20,0.10,0.38,0.05,0.03,0.02,0.02,0.03,0.03,0.05,0.01,0.02,0.01,0.02,0.02,0.01\n",
        encoding="utf-8",
    )
    return csv_path


def test_read_edx_file_parses_minimal_realistic_csv(tmp_path):
    edx_csv = _write_three_row_edx_csv(tmp_path)
    elements = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Zn']
    raw = read_edx_file({"edx_file": str(edx_csv)}, elements)

    assert isinstance(raw, EdxElementMassFractions)
    assert raw.mass_fractions.shape == (3, len(elements))
    np.testing.assert_array_equal(raw.elements, np.array(elements))
    np.testing.assert_allclose(raw.D, np.array([0.5e-6, 0.7e-6, 1.0e-6]))
    assert raw.ptype.tolist() == ["biological", "carbonaceous", "dust"]
    np.testing.assert_allclose(
        raw.mass_fractions[0],
        np.array([0.30, 0.12, 0.33, 0.04, 0.04, 0.00, 0.00, 0.04, 0.04, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00]),
    )


def test_reconstruct_edx_species_mass_fractions_dispatch_and_alignment(tmp_path):
    edx_csv = _write_three_row_edx_csv(tmp_path)
    elements = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Zn', 'Cu']
    raw = read_edx_file({"edx_file": str(edx_csv)}, elements)
    aero_spec_names = list(_DEFAULT_EDX_TARGET_SPECIES)
    assert aero_spec_names == ['SO4', 'OIN', 'OC', 'Na', 'Cl', 'biological']

    masses, classes = reconstruct_edx_species_mass_fractions(raw, aero_spec_names)
    biological_idx = aero_spec_names.index("biological")
    so4_idx = aero_spec_names.index("SO4")
    class_to_row = {ptype: i for i, ptype in enumerate(classes)}

    assert masses.shape == (3, len(aero_spec_names))
    assert classes == raw.ptype.tolist()
    assert np.all(np.isfinite(masses))
    assert np.allclose(np.sum(masses, axis=1), np.ones(3), atol=1e-10)

    assert masses[class_to_row["biological"], biological_idx] > 0
    assert masses[class_to_row["carbonaceous"], so4_idx] == 0
    assert masses[class_to_row["dust"], so4_idx] > 0

    # Regression values captured from the pre-refactor reconstruction.
    expected = np.array([
        [0.0, 0.0, 0.0, 0.04, 0.04, 0.92],
        [0.0, 0.1679621788498662, 0.7320378211501337, 0.05, 0.05, 0.0],
        [0.08987837205675969, 0.24437017524473217, 0.5657514526985081, 0.05, 0.05, 0.0],
    ])
    np.testing.assert_allclose(masses, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "ptype",
    [
        "carbonaceous",
        "carbonaceous mixed dust",
        "dust-carbonaceous",
        "organics",
        "mixed organics",
    ],
)
def test_reconstruct_edx_species_mass_fractions_carbonaceous_aliases(tmp_path, ptype):
    edx_csv = _write_three_row_edx_csv(tmp_path)
    elements = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Zn', 'Cu']
    raw = read_edx_file({"edx_file": str(edx_csv)}, elements)
    raw.ptype = np.array([raw.ptype[0], ptype, raw.ptype[2]], dtype=str)

    masses, classes = reconstruct_edx_species_mass_fractions(
        raw,
        list(_DEFAULT_EDX_TARGET_SPECIES),
    )

    assert classes[1] == ptype
    np.testing.assert_allclose(
        masses[1],
        np.array([0.0, 0.1679621788498662, 0.7320378211501337, 0.05, 0.05, 0.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_read_edx_file_non_csv_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be .csv"):
        read_edx_file({"edx_file": str(bad)}, ["C", "O"])


def test_read_edx_file_percent_and_wt_prefix_columns(tmp_path):
    csv_path = tmp_path / "edx_wt.csv"
    csv_path.write_text(
        "diam_um,class,wt_C,wt_N,wt_O,wt_Na,wt_Mg,wt_Al,wt_Si,wt_P,wt_S,wt_Cl,wt_K,wt_Ca,wt_Mn,wt_Fe,wt_Zn\n"
        "1.0,dust,20,10,39,5,3,2,2,3,3,5,1,2,1,2,2\n",
        encoding="utf-8",
    )
    elems = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Zn']
    out = read_edx_file({"edx_file": str(csv_path)}, elems)
    assert np.isclose(out.D[0], 1e-6)
    assert np.isclose(np.sum(out.mass_fractions[0]), 1.0, atol=1e-12)


def test_sample_particle_oxygen_limited_and_missing_species_raises():
    elements = np.array(['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn'])
    mass_fracs = np.array([0.2, 0.05, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.15, 0.05, 0.01, 0.02, 0.01, 0.02, 0.01, 0.01])
    with pytest.raises(ValueError, match="sum to"):
        sample_particle(['SO4', 'OIN', 'OC', 'Na', 'Cl'], mass_fracs, elements)

    with pytest.raises(ValueError, match="Could not find SO4"):
        sample_particle(['OIN', 'OC', 'Na', 'Cl'], mass_fracs, elements)


def test_sample_bio_particle_oxygen_limited_and_normalize_guard():
    elements = np.array(['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn'])
    mass_fracs = np.array([0.2, 0.1, 0.02, 0.05, 0.04, 0.08, 0.08, 0.02, 0.1, 0.05, 0.04, 0.05, 0.03, 0.02, 0.00, 0.02])
    with pytest.raises(ValueError, match="sum to"):
        sample_bio_particle(['OIN', 'biological', 'Na', 'Cl'], mass_fracs, elements)

    with pytest.raises(ValueError, match="sum to"):
        _normalize_or_raise(np.array([0.2, 0.2]))


@pytest.mark.parametrize(
    ("ptype", "carbonaceous"),
    [
        ("inorganics", False),
        ("inorganic", False),
        ("sulfate", False),
        ("organics", True),
        ("mixed organics", True),
        ("carbonaceous", True),
    ],
)
def test_reconstruction_routes_class_labels_by_whole_word(ptype, carbonaceous):
    # A substring test sent "inorganics" to the carbonaceous branch, which puts
    # all sulfur into OC: SO4 0.749 / OC 0.201 became SO4 0 / OC 0.950.
    elements = np.array(['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                         'K', 'Ca', 'Mn', 'Fe', 'Zn', 'Cu'])
    row = np.zeros(len(elements))
    for element, fraction in {"C": 0.05, "N": 0.15, "O": 0.50, "Na": 0.05, "S": 0.25}.items():
        row[list(elements).index(element)] = fraction
    raw = EdxElementMassFractions(
        np.array([1e-6]), elements, row[None, :].copy(), np.array([ptype])
    )
    species = list(_DEFAULT_EDX_TARGET_SPECIES)

    masses, _ = reconstruct_edx_species_mass_fractions(raw, species)
    so4 = masses[0, species.index("SO4")]
    oc = masses[0, species.index("OC")]

    if carbonaceous:
        assert so4 == 0.0
        assert oc == pytest.approx(0.95)
    else:
        assert so4 == pytest.approx(0.749, abs=5e-4)
        assert oc == pytest.approx(0.201, abs=5e-4)
