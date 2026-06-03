from __future__ import annotations

import numpy as np

from part2pop import make_particle
from part2pop.population.base import ParticlePopulation
from part2pop.species.registry import get_species


def assemble_population_from_mass_fractions(
    *,
    diameters,
    number_concentrations,
    species_names,
    mass_fractions,
    ids=None,
    classes=None,
    species_modifications=None,
    D_is_wet=False,
    specdata_path=None,
):
    species_modifications = species_modifications or {}

    D = np.asarray(diameters, dtype=float)
    N = np.asarray(number_concentrations, dtype=float)

    if D.ndim != 1 or N.ndim != 1:
        raise ValueError("diameters and number_concentrations must be 1D")
    if len(D) != len(N):
        raise ValueError("diameters and number_concentrations must have equal length")
    if len(species_names) != len(D) or len(mass_fractions) != len(D):
        raise ValueError("species_names and mass_fractions must match particle count")

    if ids is None:
        ids = list(range(len(D)))
    if len(ids) != len(D):
        raise ValueError("ids must match particle count")
    if classes is not None and len(classes) != len(D):
        raise ValueError("classes must match particle count when provided")

    pop_species_names = []
    normalized_species_rows = []
    normalized_fraction_rows = []

    for i in range(len(D)):
        part_names = [str(name) for name in list(species_names[i])]
        part_fracs_raw = np.asarray(mass_fractions[i], dtype=float)
        if part_fracs_raw.ndim != 1:
            raise ValueError("per-particle mass_fractions must be a flat 1D row")
        part_fracs = part_fracs_raw.ravel()

        if len(part_names) != len(part_fracs):
            raise ValueError("per-particle species_names and mass_fractions lengths must match")

        normalized_species_rows.append(part_names)
        normalized_fraction_rows.append(part_fracs)

        for name in part_names:
            if name not in pop_species_names:
                pop_species_names.append(name)

    species_list = tuple(
        get_species(spec_name, specdata_path, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    population = ParticlePopulation(
        species=species_list,
        spec_masses=[],
        num_concs=[],
        ids=[],
        classes=classes,
        species_modifications=species_modifications,
    )

    for i in range(len(D)):
        frac_map = {
            k: float(v)
            for k, v in zip(normalized_species_rows[i], normalized_fraction_rows[i])
        }
        aligned_fracs = [frac_map.get(name, 0.0) for name in pop_species_names]
        aligned_fracs = np.asarray(aligned_fracs, dtype=float).ravel()
        if aligned_fracs.ndim != 1:
            raise ValueError("aligned mass fractions must be 1D")
        if len(aligned_fracs) != len(pop_species_names):
            raise ValueError("aligned mass fractions must match master species length")

        particle = make_particle(
            D[i],
            species_list,
            aligned_fracs.copy(),
            species_modifications=species_modifications,
            specdata_path=specdata_path,
            D_is_wet=D_is_wet,
        )
        population.set_particle(particle, int(ids[i]), N[i])

    return population
