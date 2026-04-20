from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Union

import numpy as np

from ..population.base import ParticlePopulation
from ..species.registry import get_species

__all__ = ["load_population", "save_population"]

_DATA_FORMAT_VERSION = "0.1"


def _make_json_safe(value: Any) -> Any:
    """Convert common numpy objects into JSON-serializable equivalents."""
    if isinstance(value, Mapping):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _make_json_safe(value.tolist())
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _ensure_path(path: Union[Path, str]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def _serialize_metadata(metadata: Dict[str, Any]) -> np.ndarray:
    payload = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
    return np.array(payload, dtype=np.str_)


def save_population(
    filepath: Union[str, Path],
    population: ParticlePopulation,
    analysis_results: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a particle population plus analysis data into a compact NPZ file."""
    path = _ensure_path(filepath)
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

    # Store analysis results arrays
    analysis_info: list[Dict[str, Any]] = []
    if analysis_results:
        for var_name, entry in analysis_results.items():
            cfg = entry.get("config", {})
            result = entry.get("result", entry)
            if not isinstance(result, Mapping):
                result = {"value": result}
            keys = []
            for res_key, res_val in result.items():
                array_key = f"analysis__{var_name}__{res_key}"
                data_store[array_key] = np.asarray(res_val)
                keys.append(res_key)
            analysis_info.append(
                {
                    "name": var_name,
                    "config": _make_json_safe(cfg),
                    "result_keys": sorted(keys),
                }
            )

    payload: Dict[str, Any] = {
        "format_version": _DATA_FORMAT_VERSION,
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "species_names": [spec.name for spec in population.species],
        "species_modifications": _make_json_safe(population.species_modifications),
        "extra_attrs": extra_attrs,
        "analysis": analysis_info,
        "user_metadata": _make_json_safe(dict(metadata)) if metadata else {},
    }

    data_store["metadata"] = _serialize_metadata(payload)
    np.savez_compressed(path, **data_store)
    return path


def load_population(filepath: Union[str, Path]) -> tuple[ParticlePopulation, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load a saved population and its analysis data."""
    with np.load(Path(filepath), allow_pickle=False) as archive:
        metadata_raw = archive["metadata"]
        metadata = json.loads(str(metadata_raw.tolist()))
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

        return population, analysis_records, metadata
