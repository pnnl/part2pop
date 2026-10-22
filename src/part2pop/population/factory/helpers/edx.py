"""EDX-specific helpers for elemental-to-aerosol-species reconstruction.

EDX input composition is measured on an elemental mass-fraction basis. This
module reconstructs approximate aerosol-species mass fractions using a named
set of stoichiometric and grouping assumptions required by downstream part2pop
calculations. The reconstruction is not unique and should not be interpreted as
a direct measurement of aerosol material species.

Assumptions of the ``sulfate_oxide_organic_allocation`` scheme:

* Input. Each row gives element mass fractions for the configured elements.
  All 16 default elements (C, N, O, Na, Mg, Al, Si, P, S, Cl, K, Ca, Mn, Fe,
  Zn, Cu) must be present, and all six target species (SO4, OIN, OC, Na, Cl,
  biological) must be requested. Rows are treated as percent when the mean
  row sum is within ``np.isclose`` of 100 (about +/-0.001); the reconstructed
  species fractions must then sum to 0.99-1.01 or a ValueError is raised.
  Input values must be finite and nonnegative. Negative/noisy measurements
  require explicit preprocessing; they are never silently clipped. All-zero
  rows still fail the existing reconstructed-total check. Unlisted elements
  (for example Ti or Br) are not read and are not renormalized away.
* Hydrogen is not measured by EDX, so organic and ammonium mass is
  under-counted relative to the true particle mass.
* Default branch: all S is sulfate (S + 4 O, capped by measured O). Mg, Al,
  Si, K, Ca, Fe, Mn and Cu take fixed oxide oxygen (MgO, Al2O3, SiO2, K2O,
  CaO, Fe2O3, MnO, CuO), capped by the O left after sulfate; Zn joins OIN
  without oxide oxygen. Remaining C, N, P and O become OC, so carbonate
  carbon, ammonium and nitrate nitrogen, and phosphate phosphorus are all
  reconstructed as organic carbon.
* Biological branch: only Al and Si form an oxide coating; S, K, Mg, Ca, Fe,
  Mn, Zn and Cu join the biological species and no sulfate is formed.
* Carbonaceous branch (the listed carbonaceous classes, or labels containing
  the whole word "organics"): same oxides as the default branch, no sulfate;
  S joins OC.
* Na and Cl are assigned directly to the Na and Cl species in every branch.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict
import warnings

import numpy as np
import pandas as pd


_MASS_SUM_MIN = 0.99
_MASS_SUM_MAX = 1.01
_PERCENT_SUM_TARGET = 100.0
_CARBONACEOUS_CLASSES = ("carbonaceous", "carbonaceous mixed dust", "dust-carbonaceous")
_CLASS_LABEL_SEPARATORS = re.compile(r"[\s_\-/,;]+")
# Existing behavior groups all of these elemental masses into reconstructed OIN.
# Stoichiometric oxygen is allocated for Mg/Al/Si/K/Ca/Fe/Mn/Cu, while Zn is
# included in OIN without an additional oxygen allocation.
_DEFAULT_DUST_ELEMENTS = ("Mg", "Al", "Si", "K", "Ca", "Fe", "Mn", "Zn", "Cu")
_BIO_DUST_ELEMENTS = ("Al", "Si")

# Elemental molar masses used by EDX stoichiometric reconstruction.
# Only elements needed by the oxygen-allocation formulas are listed here;
# these are not AerosolSpecies properties.
_ELEMENT_MOLAR_MASS_KG_PER_MOL = {
    "O": 16.0e-3,
    "Mn": 54.94e-3,
    "Fe": 55.85e-3,
    "Mg": 24.305e-3,
    "Al": 27.0e-3,
    "Si": 28.085e-3,
    "K": 39.09e-3,
    "Ca": 40.078e-3,
    "S": 32.065e-3,
    "Cu": 63.55e-3,
}

_DEFAULT_EDX_RECONSTRUCTION_SCHEME = "sulfate_oxide_organic_allocation"
_EDX_TARGET_SPECIES = {
    "sulfate": "SO4",
    "dust": "OIN",
    "organic": "OC",
    "sodium": "Na",
    "chloride": "Cl",
    "biological": "biological",
}
_DEFAULT_EDX_TARGET_SPECIES = (
    _EDX_TARGET_SPECIES["sulfate"],
    _EDX_TARGET_SPECIES["dust"],
    _EDX_TARGET_SPECIES["organic"],
    _EDX_TARGET_SPECIES["sodium"],
    _EDX_TARGET_SPECIES["chloride"],
    _EDX_TARGET_SPECIES["biological"],
)


@dataclass(eq=False)
class EdxElementMassFractions:
    """Elemental EDX composition and particle attributes read from an input file."""

    D: np.ndarray
    elements: np.ndarray
    mass_fractions: np.ndarray
    ptype: np.ndarray


def _validate_csv_extension(filename: str) -> None:
    extension = filename.split('.')[-1]
    if extension != 'csv':
        raise ValueError(f"edx_file must be .csv format supplied format is .{extension}")


def _extract_element_mass_fractions(data: pd.DataFrame, elements: list[str], filename: str):
    particle_massfracs = np.zeros((len(data), len(elements)))
    elements = np.array(elements)

    # Find element columns by exact column name or suffix convention (*_Element).
    for jj, (spec) in enumerate(elements):
        idx = None
        for ii, (column) in enumerate(data.keys()):
            if column.split("_")[-1] == spec:
                idx = ii
                particle_massfracs[:, jj] = np.array(data[column])
            elif column == spec:
                idx = ii
                particle_massfracs[:, jj] = np.array(data[column])
        if not idx:
            # TODO: Preserve existing behavior for Issue #40 narrow pass; idx==0 quirk
            # can be addressed in follow-up without scientific changes.
            raise KeyError(f"Could not identify column for {spec} in {filename}")
    return particle_massfracs, elements


def _convert_percent_to_fraction_if_needed(particle_massfracs):
    # Convert to fractional units if rows are reported in percent and sum ~100.
    if np.isclose(np.average(np.sum(particle_massfracs, axis=1)), _PERCENT_SUM_TARGET):
        particle_massfracs /= 100.0
    return particle_massfracs


def _extract_particle_diameters(data: pd.DataFrame, filename: str):
    # Find diameter column and convert from micrometers to meters.
    matches = [k for k in data if "diam" in k.lower()]
    if len(matches) == 0:
        raise KeyError(f"Could not identify diameter column in {filename}")
    elif len(matches) > 1:
        warnings.warn(
            f"Possible diameter columns identified: {matches}. Proceeding using {matches[0]}.",
            UserWarning,
        )
    return 1e-6 * np.array(data[matches[0]])


def _extract_particle_classes(data: pd.DataFrame, filename: str):
    # Find class/label/type column and normalize classes to lowercase.
    matches = [k for k in data if any(term in k.lower() for term in ("label", "class", "type"))]
    if len(matches) == 0:
        raise KeyError(f"Could not identify classification column in {filename}")
    elif len(matches) > 1:
        warnings.warn(
            f"Possible classification columns identified: {matches}. Proceeding using {matches[0]}.",
            UserWarning,
        )
    ptypes = np.array(data[matches[0]], dtype='str')
    ptypes = np.char.lower(ptypes)  # make everything lowercase
    return ptypes


def _validate_fractions(values, label):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{label} must be finite and nonnegative; preprocess noisy EDX data explicitly")
    return values


def _element_mass_fraction_dict(elements, mass_fraction) -> dict:
    values = _validate_fractions(mass_fraction, "Element mass fractions")
    if values.ndim != 1 or len(values) != len(elements):
        raise ValueError("Element mass fractions must be a 1D row matching elements")
    return dict(zip(elements, values))


def _sulfate_oxygen_from_sulfur(data_dict: dict) -> float:
    masses = _ELEMENT_MOLAR_MASS_KG_PER_MOL
    return data_dict['S'] * ((4 * masses['O']) / masses['S'])


def _oxide_oxygen_for_default_dust(data_dict: dict) -> float:
    masses = _ELEMENT_MOLAR_MASS_KG_PER_MOL
    return (
        data_dict['Mg'] * (masses['O'] / masses['Mg'])
        + data_dict['Al'] * ((3 * masses['O']) / (2 * masses['Al']))
        + data_dict['Si'] * ((2 * masses['O']) / masses['Si'])
        + data_dict['K'] * (masses['O'] / (2 * masses['K']))
        + data_dict['Ca'] * (masses['O'] / masses['Ca'])
        + data_dict['Fe'] * ((3 * masses['O']) / (2 * masses['Fe']))
        + data_dict['Mn'] * (masses['O'] / masses['Mn'])
        + data_dict['Cu'] * (masses['O'] / masses['Cu'])
    )


def _oxide_oxygen_for_biological_dust_coating(data_dict: dict) -> float:
    masses = _ELEMENT_MOLAR_MASS_KG_PER_MOL
    return (
        + data_dict['Al'] * ((3 * masses['O']) / (2 * masses['Al']))
        + data_dict['Si'] * ((2 * masses['O']) / masses['Si'])
    )


def _dust_mass_fraction(data_dict: dict, dust_oxygen_fraction: float, dust_elements: tuple[str, ...]) -> float:
    dust_mass_fraction = dust_oxygen_fraction
    for kk in dust_elements:
        dust_mass_fraction += data_dict[kk]
    return dust_mass_fraction


def _assign_species_or_raise(sampled_masses, aerospecs, species_name, value, missing_message):
    try:
        idx = np.where(aerospecs == species_name)[0][0]
        sampled_masses[idx] = value
    except:
        raise ValueError(missing_message)


def _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict):
    sodium = _EDX_TARGET_SPECIES["sodium"]
    chloride = _EDX_TARGET_SPECIES["chloride"]
    try:
        idx = np.where(aerospecs == sodium)[0][0]
        sampled_masses[idx] = data_dict['Na']
        idx = np.where(aerospecs == chloride)[0][0]
        sampled_masses[idx] = data_dict['Cl']
    except:
        raise ValueError(f"Could not find Na or Cl in provided aerospecs: {aerospecs}")


def _normalize_or_raise(sampled_masses):
    total = np.sum(sampled_masses)
    if total > _MASS_SUM_MIN and total < _MASS_SUM_MAX:
        sampled_masses /= total
    else:
        raise ValueError(f"Sampled mass fractions sum to {np.sum(sampled_masses)}.")


def read_edx_file(config: Dict[str, Any], elements: list[str]) -> EdxElementMassFractions:
    filename = config["edx_file"]
    _validate_csv_extension(filename)
    data = pd.read_csv(filename)
    particle_massfracs, elements = _extract_element_mass_fractions(data, elements, filename)
    _validate_fractions(particle_massfracs, "Element mass fractions")
    particle_massfracs = _convert_percent_to_fraction_if_needed(particle_massfracs)
    particle_diameters = _extract_particle_diameters(data, filename)
    ptypes = _extract_particle_classes(data, filename)

    return EdxElementMassFractions(particle_diameters, elements, particle_massfracs, ptypes)


def sample_particle(aerospecs: list[str], mass_fraction: np.ndarray, elements: np.ndarray) -> np.ndarray:
    aerospecs = np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = _element_mass_fraction_dict(elements, mass_fraction)

    # Assume sulfur is sulfate (SO4) and cap sulfate oxygen by available oxygen.
    sulfate_O_fraction = _sulfate_oxygen_from_sulfur(data_dict)
    if sulfate_O_fraction <= data_dict['O']:
        sulfate_mass_fraction = data_dict['S'] + sulfate_O_fraction
    else:
        sulfate_mass_fraction = data_dict['S'] + data_dict['O']
        sulfate_O_fraction = data_dict['O']
    sulfate = _EDX_TARGET_SPECIES["sulfate"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        sulfate,
        sulfate_mass_fraction,
        f"Could not find {sulfate} in provided aerospecs: {aerospecs}",
    )

    # Mg/Al/Si/K/Ca/Fe/Mn/Cu receive stoichiometric oxide oxygen; Zn is
    # grouped into OIN without additional oxygen. Total dust oxygen is capped
    # by measured oxygen remaining after sulfate allocation.
    dust_O_fraction = _oxide_oxygen_for_default_dust(data_dict)
    if sulfate_O_fraction + dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O'] - sulfate_O_fraction
    dust_mass_fraction = _dust_mass_fraction(data_dict, dust_O_fraction, _DEFAULT_DUST_ELEMENTS)
    dust = _EDX_TARGET_SPECIES["dust"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        dust,
        dust_mass_fraction,
        f"Could not find {dust} in provided aerospecs: {aerospecs}",
    )

    # Remaining C/N/P/O mass is treated as organic carbon (OC).
    organic_mass_fraction = (
        data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['O'] - sulfate_O_fraction - dust_O_fraction
    )
    organic = _EDX_TARGET_SPECIES["organic"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        organic,
        organic_mass_fraction,
        f"Could not find {organic} in provided aerospecs: {aerospecs}",
    )

    # Na and Cl are assigned directly to Na/Cl species masses.
    _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict)

    _normalize_or_raise(sampled_masses)

    return sampled_masses


def sample_bio_particle(aerospecs: list[str], mass_fraction: np.ndarray, elements: np.ndarray) -> np.ndarray:
    aerospecs = np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = _element_mass_fraction_dict(elements, mass_fraction)

    # Al/Si are treated as a dust coating (oxides), with oxygen allocation
    # capped by the measured oxygen available to the particle.
    dust_O_fraction = _oxide_oxygen_for_biological_dust_coating(data_dict)
    if dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']
    dust_mass_fraction = _dust_mass_fraction(data_dict, dust_O_fraction, _BIO_DUST_ELEMENTS)
    dust = _EDX_TARGET_SPECIES["dust"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        dust,
        dust_mass_fraction,
        f"Could not find {dust} in provided aerospecs: {aerospecs}",
    )

    # Na and Cl are assigned directly to Na/Cl species masses.
    _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict)

    # Biological remainder: C/N/P/S/K/Mg/Ca/Fe/Mn/Zn/Cu plus oxygen not
    # allocated to the Al/Si dust coating.
    bio_mass_fraction = (
        + data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['S'] + data_dict['K'] + data_dict['Mg']
        + data_dict['Ca'] + data_dict['Fe'] + data_dict['Mn']
        + data_dict['Zn'] + data_dict['Cu'] + data_dict['O'] - dust_O_fraction
    )
    biological = _EDX_TARGET_SPECIES["biological"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        biological,
        bio_mass_fraction,
        f"Could not find bio in provided aerospecs: {aerospecs}",
    )

    _normalize_or_raise(sampled_masses)

    return sampled_masses


def sample_carbonaceous_particle(aerospecs: list[str], mass_fraction: np.ndarray, elements: np.ndarray) -> np.ndarray:
    aerospecs = np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = _element_mass_fraction_dict(elements, mass_fraction)

    # The carbonaceous branch uses the same OIN elemental grouping and oxide
    # allocation as the default branch, but no oxygen is reserved for sulfate.
    # Dust oxygen is therefore capped only by the particle's measured oxygen.
    dust_O_fraction = _oxide_oxygen_for_default_dust(data_dict)
    if dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']
    dust_mass_fraction = _dust_mass_fraction(data_dict, dust_O_fraction, _DEFAULT_DUST_ELEMENTS)
    dust = _EDX_TARGET_SPECIES["dust"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        dust,
        dust_mass_fraction,
        f"Could not find {dust} in provided aerospecs: {aerospecs}",
    )

    # Current behavior: sulfur is not assigned to SO4 in this branch; it is included in OC.
    organic_mass_fraction = (
        data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['S'] + data_dict['O'] - dust_O_fraction
    )
    organic = _EDX_TARGET_SPECIES["organic"]
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        organic,
        organic_mass_fraction,
        f"Could not find {organic} in provided aerospecs: {aerospecs}",
    )

    # Na and Cl are assigned directly to Na/Cl species masses.
    _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict)

    _normalize_or_raise(sampled_masses)

    return sampled_masses


def _is_carbonaceous_class(ptype: str) -> bool:
    """Return True for particle classes routed to the carbonaceous branch.

    Matches the listed carbonaceous classes exactly, or any label containing
    the whole word "organics" (for example "organics" or "mixed organics").
    Whole-word matching keeps labels such as "inorganics" on the default
    branch; a substring test sent them to the carbonaceous branch, which
    assigns sulfur to OC instead of SO4.
    """
    if ptype in _CARBONACEOUS_CLASSES:
        return True
    return "organics" in _CLASS_LABEL_SEPARATORS.split(ptype.strip())


def reconstruct_edx_species_mass_fractions(
    raw_population: EdxElementMassFractions,
    aero_spec_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Reconstruct aerosol-species mass fractions from elemental EDX input."""
    particle_classes = []
    aero_spec_masses = np.zeros((len(raw_population.ptype), len(aero_spec_names)))
    for ii, (ptype, mass_fracs) in enumerate(zip(raw_population.ptype, raw_population.mass_fractions)):
        particle_classes.append(ptype)

        if ptype == 'biological':
            aero_spec_masses[ii] = sample_bio_particle(aero_spec_names, mass_fracs, raw_population.elements)
        elif _is_carbonaceous_class(ptype):
            aero_spec_masses[ii] = sample_carbonaceous_particle(aero_spec_names, mass_fracs, raw_population.elements)
        else:
            aero_spec_masses[ii] = sample_particle(aero_spec_names, mass_fracs, raw_population.elements)
    return aero_spec_masses, particle_classes
