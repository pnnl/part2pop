import importlib

import pytest

from part2pop.analysis.base import Variable, VariableMeta


def test_import_base():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.base")


class _DummyVariable(Variable):
    meta = VariableMeta(
        name="dummy",
        axis_names=("x",),
        description="a dummy variable",
        units="m",
        default_cfg={"scale": "linear"},
    )

    def compute(self, population):
        return population


def test_variable_rescale_always_raises():
    var = _DummyVariable(cfg={})
    with pytest.raises(NotImplementedError):
        var.rescale("cm")
