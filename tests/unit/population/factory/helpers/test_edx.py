import numpy as np

from part2pop.population.factory.helpers.edx import (
    Population_MassFracs,
    read_edx_file,
    reconstruct_edx_species_mass_fractions,
)


def _write_three_row_edx_csv(tmp_path):
    csv_path = tmp_path / "edx_three_rows.csv"
    csv_path.write_text(
        "diam_um,class,C,N,O,Na,Mg,Al,Si,P,S,Cl,K,Ca,Mn,Fe,Zn\n"
        "0.5,biological,0.35,0.12,0.33,0.04,0.04,0.00,0.00,0.04,0.04,0.04,0.00,0.00,0.00,0.00,0.00\n"
        # Keep synthetic row within the existing EDX mass-balance guard.
        "0.7,carbonaceous,0.30,0.10,0.34,0.05,0.03,0.02,0.02,0.03,0.03,0.05,0.01,0.005,0.005,0.005,0.005\n"
        # Keep synthetic row within the existing EDX mass-balance guard.
        "1.0,dust,0.20,0.10,0.39,0.05,0.03,0.02,0.02,0.03,0.03,0.05,0.01,0.02,0.01,0.02,0.02\n",
        encoding="utf-8",
    )
    return csv_path


def test_read_edx_file_parses_minimal_realistic_csv(tmp_path):
    edx_csv = _write_three_row_edx_csv(tmp_path)
    elements = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Zn']
    raw = read_edx_file({"edx_file": str(edx_csv)}, elements)

    assert isinstance(raw, Population_MassFracs)
    assert raw.mass_fractions.shape == (3, len(elements))
    assert np.isclose(raw.D[0], 0.5e-6)
    assert np.isclose(raw.D[1], 0.7e-6)
    assert raw.ptype.tolist() == ["biological", "carbonaceous", "dust"]


def test_reconstruct_edx_species_mass_fractions_dispatch_and_alignment(tmp_path):
    edx_csv = _write_three_row_edx_csv(tmp_path)
    elements = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Mn', 'Fe', 'Zn']
    raw = read_edx_file({"edx_file": str(edx_csv)}, elements)
    aero_spec_names = ['SO4', 'OIN', 'OC', 'Na', 'Cl', 'biological']

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
