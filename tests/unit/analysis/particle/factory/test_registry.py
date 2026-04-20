import sys
import pytest

from part2pop.analysis.base import VariableMeta
from part2pop.analysis.particle.base import ParticleVariable
from part2pop.analysis.particle.factory import registry as reg


def test_particle_registry_exposes_known_variables():
    names = reg.list_particle_variables(include_aliases=True)
    assert "Dwet" in names
    assert "wet_diameter" in names
    assert "kappa" in names  # alias for the hygroscopicity variable


def test_resolve_alias_returns_canonical_name():
    assert reg.resolve_particle_name("wet_diameter") == "Dwet"
    assert reg.resolve_particle_name("kappa") == "kappa"


def test_particle_builder_instantiation_provides_meta():
    builder = reg.get_particle_builder("Dwet")
    inst = builder({})
    assert inst.meta.name == "Dwet"
    axis_names = inst.meta.axis_names
    if isinstance(axis_names, str):
        axis_names = (axis_names,)
    assert axis_names == ("rh_grid",)


def test_unknown_particle_variable_suggests_close_match():
    with pytest.raises(reg.UnknownParticleVariableError) as excinfo:
        reg.resolve_particle_name("Dwetty")
    assert "Dwet" in str(excinfo.value)


def test_describe_particle_variable_returns_meta():
    @reg.register_particle_variable("describe_test")
    class DescribeTestVar(ParticleVariable):
        meta = VariableMeta(
            name="describe_test",
            axis_names=("rh_grid",),
            description="a descriptive test variable",
            default_cfg={},
            units={"m": "meters"},
        )

        def compute_one(self, population, part_id):
            return 0.0

        def compute(self, population):
            return 0.0

    try:
        info = reg.describe_particle_variable("describe_test")
        assert info["name"] == "describe_test"
        assert info["value_key"] == "describe_test"
        assert info["axis_keys"] == ["rh_grid"]
        assert info["description"] == "a descriptive test variable"
        assert info["defaults"] == {}
        assert info["units"] == {"m": "meters"}
    finally:
        reg._PARTICLE_REG.pop("describe_test", None)


def test_register_alias_collision_raises(monkeypatch):
    # Ensure registry is populated before checking alias collisions
    reg.resolve_particle_name("wet_diameter")

    alias = "wet_diameter"
    with pytest.raises(KeyError):
        @reg.register_particle_variable("dummy_conflict")
        class DummyConflict(ParticleVariable):
            meta = VariableMeta(
                name="dummy_conflict",
                description="conflict",
                axis_names=("rh_grid",),
                default_cfg={},
                aliases=(alias,),
            )

    reg._PARTICLE_REG.pop("dummy_conflict", None)


def test_discover_particle_factory_handles_import_failure(monkeypatch, tmp_path):
    prev_discovered = reg._PARTICLE_DISCOVERED
    reg._PARTICLE_DISCOVERED = False

    module_name = "tmp_particle_plugin"
    file_path = tmp_path / f"{module_name}.py"
    file_path.write_text("def build(cfg=None):\n    return 'plugin'\n")

    def fake_iter_modules(paths):
        yield (None, module_name, False)

    monkeypatch.setattr(reg.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(reg.os.path, "dirname", lambda _: str(tmp_path))

    original_import = reg.importlib.import_module

    def failing_import(name, *args, **kwargs):
        if name.endswith(module_name):
            raise ImportError("simulated failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(reg.importlib, "import_module", failing_import)

    try:
        reg._PARTICLE_REG.pop(module_name, None)
        reg._discover_particle_factories()
        assert module_name in reg._PARTICLE_REG
    finally:
        reg._PARTICLE_REG.pop(module_name, None)
        fullname = f"{reg.__package__}.{module_name}"
        sys.modules.pop(fullname, None)
        reg._PARTICLE_DISCOVERED = prev_discovered
