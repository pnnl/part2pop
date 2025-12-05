# tests/unit/freezing/factory/test_registry.py

from pyparticle.freezing.factory.registry import discover_morphology_types


def test_discover_morphology_types_returns_callables():
    morphs = discover_morphology_types()
    assert isinstance(morphs, dict)
    assert len(morphs) > 0
    for name, builder in morphs.items():
        assert callable(builder)
