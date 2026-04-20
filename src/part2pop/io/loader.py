from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from ..population.base import ParticlePopulation
from ..species.registry import get_species

__all__ = ["load_population"]


def _recover_metadata(raw: np.ndarray) -> Dict[str, Any]:
    return json.loads(str(raw.tolist()))


def load_population(
    filepath: Union[str, Path],
    include_particle_data: bool = False,
) -> Tuple[
    ParticlePopulation,
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
] | Tuple[
    ParticlePopulation,
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
]:
    with np.load(Path(filepath), allow_pickle=False) as archive:
        metadata = _recover_metadata(archive["metadata"])
        species_mods = metadata.get("species_modifications", {})

        species_names = metadata.get("species_names", [])
        species = tuple(
            get_species(name, **species_mods.get(name, {})) for name in species_names
        )

        spec_masses = np.asarray(archive["spec_masses"])
        num_concs = np.asarray(archive["num_concs"])
        ids = tuple(np.asarray(archive["ids"]).tolist())

        population = ParticlePopulation(
            species=species,
            spec_masses=spec_masses,
            num_concs=num_concs,
            ids=list(ids),
            species_modifications=species_mods,
        )

        for attr in metadata.get("extra_attrs", []):
            key = f"extra__{attr}"
            if key in archive:
                setattr(population, attr, np.array(archive[key]))

        analysis_records: Dict[str, Dict[str, Any]] = {}
        for record in metadata.get("analysis", []):
            var_name = record["name"]
            result_keys = record.get("result_keys", [])
            result_data: Dict[str, Any] = {}
            for key in result_keys:
                array_key = f"analysis__{var_name}__{key}"
                if array_key in archive:
                    result_data[key] = np.array(archive[array_key])
            analysis_records[var_name] = {
                "config": record.get("config", {}),
                "result": result_data,
            }

        particle_records: Dict[str, Dict[str, Any]] = {}
        for record in metadata.get("particle_variables", []):
            var_name = record["name"]
            result_keys = record.get("result_keys", [])
            result_data: Dict[str, Any] = {}
            for key in result_keys:
                array_key = f"particle__{var_name}__{key}"
                if array_key in archive:
                    result_data[key] = np.array(archive[array_key])
            particle_records[var_name] = {
                "config": record.get("config", {}),
                "result": result_data,
            }

        if include_particle_data:
            return population, analysis_records, metadata, particle_records

        return population, analysis_records, metadata
