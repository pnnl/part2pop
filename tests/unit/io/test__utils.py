import importlib

import numpy as np

from part2pop.io import _utils


def test_import_io__utils():
    importlib.import_module("part2pop.io._utils")


def test_make_json_safe_handles_nested_values():
    payload = {
        "array": np.array([1, 2]),
        "mapping": {"pi": np.float64(3.14)},
        "sequence": (np.int32(4), "text"),
    }
    safe = _utils.make_json_safe(payload)

    assert isinstance(safe["array"], list)
    assert safe["mapping"]["pi"] == 3.14
    assert safe["sequence"][0] == 4


def test_make_json_safe_handles_custom_objects():
    class Custom:
        def __str__(self):
            return "custom"

    payload = {
        "array": np.array([1, 2]),
        "nested": {"value": np.float64(3.0)},
        "tuple": (np.int32(4), "text"),
        "custom": Custom(),
    }
    safe = _utils.make_json_safe(payload)

    assert isinstance(safe["array"], list)
    assert safe["nested"]["value"] == 3.0
    assert safe["tuple"][0] == 4
    assert safe["custom"] == "custom"


def test_serialize_metadata_emits_numpy_string():
    metadata = {"tag": "roundtrip", "value": 42}
    serialized = _utils.serialize_metadata(metadata)

    assert isinstance(serialized, np.ndarray)
    assert serialized.dtype.type == np.str_
    assert "tag\":\"roundtrip" in serialized.item()


def test_ensure_path_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "data" / "population"
    result = _utils.ensure_path(target)

    assert result == target
    assert result.parent.exists()
