import importlib

import pytest

from part2pop.analysis.base import Variable, VariableMeta


def test_import_base():
    # Smoke test: module should import successfully
    importlib.import_module("part2pop.analysis.base")


def test_variable_meta_and_rescale():
    class DummyVariable(Variable):
        meta = VariableMeta(
            name="dummy",
            axis_names=(),
            description="desc",
            units="m",
        )

        def compute(self, population):
            return None

    var = DummyVariable(cfg={})
    with pytest.raises(NotImplementedError, match="km"):
        var.rescale("km")
