from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

from ..population.base import ParticlePopulation
from ._utils import ensure_path, make_json_safe, serialize_metadata

__all__ = ["save_population"]

_DATA_FORMAT_VERSION = "0.1"


def save_population(
    filepath: Union[str, Path],
    population: ParticlePopulation,
    analysis_results: Mapping[str, Mapping[str, Any]] | None = None,
    particle_results: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    path = ensure_path(filepath)
    data_store: Dict[str, np.ndarray] = {}

    data_store["spec_masses"] = np.asarray(population.spec_masses, dtype=float)
    data_store["num_concs"] = np.asarray(population.num_concs, dtype=float)
    data_store["ids"] = np.asarray(population.ids, dtype=int)

    base_keys = {"species", "spec_masses", "num_concs", "ids", "species_modifications"}
    extra_attrs: list[str] = []
    for key, value in population.__dict__.items():
        if key in base_keys or value is None:
            continue
        extra_attrs.append(key)
        data_store[f"extra__{key}"] = np.asarray(value)

    analysis_info: list[Dict[str, Any]] = []
    if analysis_results:
        for var_name, entry in analysis_results.items():
            cfg = entry.get("config", {})
            result = entry.get("result", entry)
            if not isinstance(result, Mapping):
                result = {"value": result}
            keys: list[str] = []
            for res_key, res_val in result.items():
                array_key = f"analysis__{var_name}__{res_key}"
                data_store[array_key] = np.asarray(res_val)
                keys.append(res_key)
            analysis_info.append(
                {
                    "name": var_name,
                    "config": make_json_safe(cfg),
                    "result_keys": sorted(keys),
                }
            )

    particle_info: list[Dict[str, Any]] = []
    if particle_results:
        for var_name, entry in particle_results.items():
            cfg = entry.get("config", {})
            result = entry.get("result", entry)
            if not isinstance(result, Mapping):
                result = {"value": result}
            keys: list[str] = []
            for res_key, res_val in result.items():
                array_key = f"particle__{var_name}__{res_key}"
                data_store[array_key] = np.asarray(res_val)
                keys.append(res_key)
            particle_info.append(
                {
                    "name": var_name,
                    "config": make_json_safe(cfg),
                    "result_keys": sorted(keys),
                }
            )

    payload: Dict[str, Any] = {
        "format_version": _DATA_FORMAT_VERSION,
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "species_names": [spec.name for spec in population.species],
        "species_modifications": make_json_safe(population.species_modifications),
        "extra_attrs": extra_attrs,
        "analysis": analysis_info,
        "particle_variables": particle_info,
        "user_metadata": make_json_safe(dict(metadata)) if metadata else {},
    }

    data_store["metadata"] = serialize_metadata(payload)
    np.savez_compressed(path, **data_store)
    return path
