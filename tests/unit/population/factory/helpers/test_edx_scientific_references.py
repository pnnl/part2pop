"""Stoichiometric characterization of the EDX elemental-to-species reconstruction.

Reference inputs are calculated independently from pure-compound formulas,
normalizing the masses of EDX-measurable elements (hydrogen is omitted).
These are *synthetic stoichiometric reference cases*, not laboratory EDX data.

Passing the tests demonstrates existing reconstruction behavior, NOT chemical
speciation accuracy. EDX does not uniquely identify chemical compounds.

Known scientific limitations of the current default reconstruction:
  * Zn is allocated to OIN without allocating its oxide oxygen; ZnO oxygen
    therefore appears in the reconstructed OC bin.
  * Carbon in inorganic carbonate (e.g., CaCO3), and part of its oxygen,
    are allocated to OC rather than to an inorganic carbonate.
  * Nitrogen in ammonium sulfate is allocated to OC; sulfate and ammonium
    cannot be represented as distinct compounds by the present scheme.
  * Nitrate is reconstructed as OC: all N and the oxygen not used by sulfate
    or oxides become organic carbon (NH4NO3 becomes 100% OC).
  * Phosphorus and its remaining oxygen are reconstructed as OC (for example
    most of a calcium phosphate particle).
  * Particle class affects the sulfate/organic allocation, and EDX alone
    cannot establish whether a particle is biological or carbonaceous.

Path forward (separate scientific change; NOT part of refactor PR #108):
  1. Define and document alternative, selectable reconstruction schemes
     for oxides (including ZnO), carbonates, and ammonium-bearing salts.
  2. Track oxygen/elemental mass balance and chemical ambiguity explicitly;
     retain the legacy behavior for reproducibility and update these
     characterization expectations only when selecting a new scheme.
  3. Validate those schemes against measured reference aerosols with known
     composition, blank/substrate corrections, and independent chemical
     speciation (for example, sulfate/ammonium via ion chromatography).
"""

import numpy as np
import pytest

from part2pop.population.factory.helpers.edx import sample_particle


ELEMENTS = np.array([
    "C", "N", "O", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "K", "Ca", "Mn", "Fe", "Cu", "Zn",
])
SPECIES = ["SO4", "OIN", "OC", "Na", "Cl", "biological"]

# Atomic masses here are independent test inputs, not imported from edx.py.
ATOMIC_MASS = {
    "C": 12.011, "N": 14.007, "O": 16.0,
    "Na": 22.99, "Cl": 35.45, "Si": 28.085,
    "Fe": 55.85, "Zn": 65.38, "Ca": 40.078,
    "S": 32.065, "P": 30.974,
}


def reconstruct_formula(formula):
    """Reconstruct from a pure compound's detected elemental mass fractions."""
    masses = {
        element: count * ATOMIC_MASS[element]
        for element, count in formula.items()
    }
    total = sum(masses.values())
    fractions = np.array([
        masses.get(element, 0.0) / total
        for element in ELEMENTS
    ])

    output = sample_particle(SPECIES, fractions, ELEMENTS)
    assert np.all(np.isfinite(output))
    assert np.all(output >= -1e-12)
    assert output.sum() == pytest.approx(1.0)
    return dict(zip(SPECIES, output))


def test_sodium_chloride():
    result = reconstruct_formula({"Na": 1, "Cl": 1})
    assert result["Na"] == pytest.approx(22.99 / 58.44)
    assert result["Cl"] == pytest.approx(35.45 / 58.44)
    assert result["OC"] == pytest.approx(0)


def test_quartz():
    result = reconstruct_formula({"Si": 1, "O": 2})
    assert result["OIN"] == pytest.approx(1.0)
    assert result["OC"] == pytest.approx(0)


def test_iron_oxide():
    result = reconstruct_formula({"Fe": 2, "O": 3})
    assert result["OIN"] == pytest.approx(1.0)
    assert result["OC"] == pytest.approx(0)


def test_known_limitation_zinc_oxide():
    # Known incorrect interpretation, recorded so a later scheme can fix it.
    # ZnO is inorganic; the present scheme wrongly assigns Zn's O to OC.
    result = reconstruct_formula({"Zn": 1, "O": 1})
    assert result["OC"] == pytest.approx(16 / 81.38)


def test_known_limitation_calcium_carbonate():
    # CaCO3 is inorganic but its C plus 2 O atoms are reconstructed as OC.
    result = reconstruct_formula({"Ca": 1, "C": 1, "O": 3})
    expected_false_oc = (12.011 + 32) / 100.089
    assert result["OC"] == pytest.approx(expected_false_oc)


def test_known_limitation_ammonium_sulfate():
    # Hydrogen is not detected by EDX and is excluded from normalization.
    # The current sulfate-allocation model wrongly treats ammonium N as OC.
    result = reconstruct_formula({"N": 2, "S": 1, "O": 4})
    expected_false_oc = 28.014 / (28.014 + 32.065 + 64)
    assert result["OC"] == pytest.approx(expected_false_oc)


def test_known_limitation_ammonium_nitrate():
    # Hydrogen is undetected; every N and O atom of NH4NO3 ends up in OC.
    result = reconstruct_formula({"N": 2, "O": 3})
    assert result["OC"] == pytest.approx(1.0)
    assert result["SO4"] == pytest.approx(0)


def test_known_limitation_calcium_phosphate():
    # Ca3(PO4)2: Ca takes CaO oxygen into OIN; P and the remaining five O
    # atoms are reconstructed as organic carbon.
    total = 3 * 40.078 + 2 * 30.974 + 8 * 16.0
    result = reconstruct_formula({"Ca": 3, "P": 2, "O": 8})
    assert result["OIN"] == pytest.approx((3 * 40.078 + 3 * 16.0) / total)
    assert result["OC"] == pytest.approx((2 * 30.974 + 5 * 16.0) / total)
