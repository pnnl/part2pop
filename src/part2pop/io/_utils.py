from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

__all__ = ["make_json_safe", "serialize_metadata", "ensure_path"]


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


def ensure_path(path: Union[Path, str]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def serialize_metadata(metadata: Dict[str, Any]) -> np.ndarray:
    payload = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
    return np.array(payload, dtype=np.str_)
