# tests/unit/analysis/test_base.py

import pytest

from pyparticle.analysis.base import VariableMeta, Variable


def test_variable_meta_fields():
    meta = VariableMeta(
        name="foo",
        axis_names=("x", "y"),
        description="foo var",
        long_label="Foo [unit]",
        short_label="Foo",
        default_cfg={"a": 1},
        units={"foo": "1"},
    )

    assert meta.name == "foo"
    assert "x" in meta.axis_names
    assert meta.default_cfg["a"] == 1
    assert meta.units["foo"] == "1"


def test_variable_rescale_not_implemented():
    meta = VariableMeta(
        name="bar",
        axis_names=("x",),
        description="bar var",
        units={"bar": "unit"},
    )

    v = Variable(meta, cfg={})
    # no-op if units match
    v.rescale({"bar": "unit"})
    # otherwise raises
    with pytest.raises(NotImplementedError):
        v.rescale({"bar": "other"})
