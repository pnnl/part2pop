from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

__all__ = ["make_json_safe", "serialize_metadata", "ensure_path"]
_POPULATION_METADATA_ARRAY_TAG = "__part2pop_ndarray__"
_POPULATION_METADATA_MAPPING_TAG = "__part2pop_mapping__"
_POPULATION_METADATA_SCALAR_TAG = "__part2pop_numpy_scalar__"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _encode_population_metadata(value: Any, array_store: Dict[str, np.ndarray]) -> Any:
    """Encode supported population metadata into JSON-safe descriptors.

    Mappings use string keys, tuples/lists become lists, NumPy scalars become
    Python scalars where possible. Extended-precision scalars whose item()
    remains a NumPy scalar use a tagged zero-dimensional array, preserving
    their dtype and precision. Non-object NumPy arrays are stored in the
    NPZ array store for lossless restoration. Object-dtype arrays are rejected
    because loading uses ``allow_pickle=False``; masked arrays are rejected
    because their mask cannot be stored losslessly here. Other unsupported
    objects are stringified, matching :func:`make_json_safe`. Non-finite
    Python floats are written with JSON's non-standard ``NaN``/``Infinity``
    tokens, which part2pop reads back but strict JSON parsers may reject.
    """
    if isinstance(value, Mapping):
        return {
            _POPULATION_METADATA_MAPPING_TAG: {
                str(k): _encode_population_metadata(v, array_store)
                for k, v in value.items()
            }
        }
    if isinstance(value, (list, tuple)):
        return [_encode_population_metadata(v, array_store) for v in value]
    if isinstance(value, np.ma.MaskedArray):
        # np.asarray would silently drop the mask, restoring masked entries
        # as ordinary values on load.
        raise TypeError(
            "population metadata masked arrays are not supported; "
            "fill them (np.ma.filled) or store the mask as a separate array"
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("population metadata arrays with object dtype are not supported")
        array_key = f"population_metadata__array_{len(array_store)}"
        array_store[array_key] = np.asarray(value)
        return {_POPULATION_METADATA_ARRAY_TAG: array_key}
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):
            # longdouble/clongdouble cannot always become Python scalars.
            # Recursing on item would never make progress on those platforms.
            return {_POPULATION_METADATA_SCALAR_TAG:
                    _encode_population_metadata(np.asarray(value), array_store)}
        return _encode_population_metadata(item, array_store)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _decode_population_metadata(value: Any, array_source: Any) -> Any:
    """Decode population metadata and restore tagged NumPy arrays."""
    if isinstance(value, Mapping):
        if set(value) == {_POPULATION_METADATA_SCALAR_TAG}:
            array = _decode_population_metadata(value[_POPULATION_METADATA_SCALAR_TAG], array_source)
            if not isinstance(array, np.ndarray) or array.ndim != 0 or array.dtype.hasobject:
                raise ValueError("invalid encoded population metadata scalar")
            return array[()]
        if set(value) == {_POPULATION_METADATA_MAPPING_TAG}:
            items = value[_POPULATION_METADATA_MAPPING_TAG]
            if not isinstance(items, Mapping):
                raise ValueError("invalid encoded population metadata mapping")
            return {
                str(k): _decode_population_metadata(v, array_source)
                for k, v in items.items()
            }

        if set(value) == {_POPULATION_METADATA_ARRAY_TAG}:
            array_key = value[_POPULATION_METADATA_ARRAY_TAG]
            if not isinstance(array_key, str) or not array_key.startswith("population_metadata__array_"):
                raise ValueError("invalid population metadata array reference")
            if array_key not in array_source:
                raise KeyError(f"missing population metadata array: {array_key}")
            return np.array(array_source[array_key])

        # Backward-compatible fallback for plain JSON mappings.
        return {str(k): _decode_population_metadata(v, array_source) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_population_metadata(v, array_source) for v in value]
    return value

def ensure_path(path: Union[Path, str]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def serialize_metadata(metadata: Dict[str, Any]) -> np.ndarray:
    payload = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
    return np.array(payload, dtype=np.str_)
