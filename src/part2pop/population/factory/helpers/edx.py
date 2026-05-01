"""EDX-specific helper functions for population reconstruction."""

from typing import Any, Dict
import pandas as pd
import numpy as np
import warnings


_MASS_SUM_MIN = 0.99
_MASS_SUM_MAX = 1.01
_PERCENT_SUM_TARGET = 100.0
_CARBONACEOUS_CLASSES = ("carbonaceous", "carbonaceous mixed dust", "dust-carbonaceous")
_DEFAULT_DUST_ELEMENTS = ("Mg", "Al", "Si", "K", "Ca", "Fe", "Mn", "Zn")
_BIO_DUST_ELEMENTS = ("Al", "Si")


class ElementMasses:
    """Element molar masses used by EDX reconstruction assumptions."""

    def __init__(self):
        self.O = 16.0e-3
        self.Mn = 54.94e-3
        self.Fe = 55.85e-3
        self.Mg = 24.305e-3
        self.Al = 27.0e-3
        self.Si = 28.085e-3
        self.K = 39.09e-3
        self.Ca = 40.078e-3
        self.S = 32.065e-3
        self.Na = 22.99e-3
        self.Cl = 35.45e-3


class Population_MassFracs:
    def __init__(self, diameters, elements, mass_fractions, ptypes):
        self.D = diameters
        self.elements = elements
        self.mass_fractions = mass_fractions
        self.ptype = ptypes


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


def _element_mass_fraction_dict(elements, mass_fraction) -> dict:
    return dict(zip(elements, mass_fraction))


def _default_dust_oxygen_fraction(data_dict: dict, molec_masses: ElementMasses) -> float:
    return (
        data_dict['Mg'] * (molec_masses.O / molec_masses.Mg)
        + data_dict['Al'] * ((3 * molec_masses.O) / (2 * molec_masses.Al))
        + data_dict['Si'] * ((2 * molec_masses.O) / molec_masses.Si)
        + data_dict['K'] * (molec_masses.O / (2 * molec_masses.K))
        + data_dict['Ca'] * (molec_masses.O / molec_masses.Ca)
        + data_dict['Fe'] * ((3 * molec_masses.O) / (2 * molec_masses.Fe))
        + data_dict['Mn'] * (molec_masses.O / molec_masses.Mn)
    )


def _bio_dust_oxygen_fraction(data_dict: dict, molec_masses: ElementMasses) -> float:
    return (
        + data_dict['Al'] * ((3 * molec_masses.O) / (2 * molec_masses.Al))
        + data_dict['Si'] * ((2 * molec_masses.O) / molec_masses.Si)
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
    try:
        idx = np.where(aerospecs == 'Na')[0][0]
        sampled_masses[idx] = data_dict['Na']
        idx = np.where(aerospecs == 'Cl')[0][0]
        sampled_masses[idx] = data_dict['Cl']
    except:
        raise ValueError(f"Could not find Na or Cl in provided aerospecs: {aerospecs}")


def _normalize_or_raise(sampled_masses):
    total = np.sum(sampled_masses)
    if total > _MASS_SUM_MIN and total < _MASS_SUM_MAX:
        sampled_masses /= total
    else:
        raise ValueError(f"Sampled mass fractions sum to {np.sum(sampled_masses)}.")


def read_edx_file(config: Dict[str, Any], elements: list[str]) -> Population_MassFracs:
    filename = config["edx_file"]
    _validate_csv_extension(filename)
    data = pd.read_csv(filename)
    particle_massfracs, elements = _extract_element_mass_fractions(data, elements, filename)
    particle_massfracs = _convert_percent_to_fraction_if_needed(particle_massfracs)
    particle_diameters = _extract_particle_diameters(data, filename)
    ptypes = _extract_particle_classes(data, filename)

    return Population_MassFracs(particle_diameters, elements, particle_massfracs, ptypes)


def sample_particle(aerospecs: list[str], mass_fraction: np.ndarray, elements: np.ndarray) -> np.ndarray:
    aerospecs = np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = _element_mass_fraction_dict(elements, mass_fraction)
    molec_masses = ElementMasses()

    # Assume sulfur is sulfate (SO4) and cap sulfate oxygen by available oxygen.
    sulfate_O_fraction = data_dict['S'] * ((4 * molec_masses.O) / molec_masses.S)
    if sulfate_O_fraction <= data_dict['O']:
        sulfate_mass_fraction = data_dict['S'] + sulfate_O_fraction
    else:
        sulfate_mass_fraction = data_dict['S'] + data_dict['O']
        sulfate_O_fraction = data_dict['O']
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'SO4',
        sulfate_mass_fraction,
        f"Could not find SO4 in provided aerospecs: {aerospecs}",
    )

    # Dust oxides from Mg/Al/Si/K/Ca/Fe/Mn are oxygen-limited by remaining oxygen.
    dust_O_fraction = _default_dust_oxygen_fraction(data_dict, molec_masses)
    if sulfate_O_fraction + dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O'] - sulfate_O_fraction
    dust_mass_fraction = _dust_mass_fraction(data_dict, dust_O_fraction, _DEFAULT_DUST_ELEMENTS)
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'OIN',
        dust_mass_fraction,
        f"Could not find OIN in provided aerospecs: {aerospecs}",
    )

    # Remaining C/N/P/O mass is treated as organic carbon (OC).
    organic_mass_fraction = (
        data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['O'] - sulfate_O_fraction - dust_O_fraction
    )
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'OC',
        organic_mass_fraction,
        f"Could not find OC in provided aerospecs: {aerospecs}",
    )

    # Na and Cl are assigned directly to Na/Cl species masses.
    _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict)

    _normalize_or_raise(sampled_masses)

    return sampled_masses


def sample_bio_particle(aerospecs: list[str], mass_fraction: np.ndarray, elements: np.ndarray) -> np.ndarray:
    aerospecs = np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = _element_mass_fraction_dict(elements, mass_fraction)
    molec_masses = ElementMasses()

    # Al/Si are treated as dust coating (oxides), capped by available oxygen.
    dust_O_fraction = _bio_dust_oxygen_fraction(data_dict, molec_masses)
    if dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']
    dust_mass_fraction = _dust_mass_fraction(data_dict, dust_O_fraction, _BIO_DUST_ELEMENTS)
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'OIN',
        dust_mass_fraction,
        f"Could not find OIN in provided aerospecs: {aerospecs}",
    )

    # Na and Cl are assigned directly to Na/Cl species masses.
    _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict)

    # Biological remainder: C/N/P/S/K/Mg/Ca/Fe/Mn/Zn plus remaining oxygen.
    bio_mass_fraction = (
        + data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['S'] + data_dict['K'] + data_dict['Mg']
        + data_dict['Ca'] + data_dict['Fe'] + data_dict['Mn']
        + data_dict['Zn'] + data_dict['O'] - dust_O_fraction
    )
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'biological',
        bio_mass_fraction,
        f"Could not find bio in provided aerospecs: {aerospecs}",
    )

    _normalize_or_raise(sampled_masses)

    return sampled_masses


def sample_carbonaceous_particle(aerospecs: list[str], mass_fraction: np.ndarray, elements: np.ndarray) -> np.ndarray:
    aerospecs = np.array(aerospecs)
    sampled_masses = np.zeros(len(aerospecs))
    data_dict = _element_mass_fraction_dict(elements, mass_fraction)
    molec_masses = ElementMasses()

    # Dust oxides are formed similarly to default particles, capped by oxygen.
    dust_O_fraction = _default_dust_oxygen_fraction(data_dict, molec_masses)
    if dust_O_fraction > data_dict['O']:
        dust_O_fraction = data_dict['O']
    dust_mass_fraction = _dust_mass_fraction(data_dict, dust_O_fraction, _DEFAULT_DUST_ELEMENTS)
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'OIN',
        dust_mass_fraction,
        f"Could not find OIN in provided aerospecs: {aerospecs}",
    )

    # Current behavior: sulfur is not assigned to SO4 in this branch; it is included in OC.
    organic_mass_fraction = (
        data_dict['C'] + data_dict['N'] + data_dict['P']
        + data_dict['S'] + data_dict['O'] - dust_O_fraction
    )
    _assign_species_or_raise(
        sampled_masses,
        aerospecs,
        'OC',
        organic_mass_fraction,
        f"Could not find OC in provided aerospecs: {aerospecs}",
    )

    # Na and Cl are assigned directly to Na/Cl species masses.
    _assign_nacl_or_raise(sampled_masses, aerospecs, data_dict)

    _normalize_or_raise(sampled_masses)

    return sampled_masses


def reconstruct_edx_species_mass_fractions(raw_population, aero_spec_names):
    particle_classes = []
    aero_spec_masses = np.zeros((len(raw_population.ptype), len(aero_spec_names)))
    for ii, (ptype, mass_fracs) in enumerate(zip(raw_population.ptype, raw_population.mass_fractions)):
        particle_classes.append(ptype)
        if ptype == 'biological':
            aero_spec_masses[ii] = sample_bio_particle(aero_spec_names, mass_fracs, raw_population.elements)
        elif ptype in _CARBONACEOUS_CLASSES:
            aero_spec_masses[ii] = sample_carbonaceous_particle(aero_spec_names, mass_fracs, raw_population.elements)
        else:
            aero_spec_masses[ii] = sample_particle(aero_spec_names, mass_fracs, raw_population.elements)
    return aero_spec_masses, particle_classes
