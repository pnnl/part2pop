from part2pop.species import resolve_species_name, resolve_species_names
from part2pop.species.resolution import resolve_species_name_rows


def test_default_aliases_resolve_expected_labels():
    assert resolve_species_name("dust") == "OIN"
    assert resolve_species_name("Dust") == "OIN"
    assert resolve_species_name("soot") == "BC"
    assert resolve_species_name("black carbon") == "BC"
    assert resolve_species_name("Org") == "OC"
    assert resolve_species_name("POA") == "OC"


def test_alias_normalization_handles_underscores_and_hyphens():
    assert resolve_species_name("black_carbon") == "BC"
    assert resolve_species_name("BLACK-CARBON") == "BC"


def test_canonical_names_pass_through_when_not_aliased():
    assert resolve_species_name("SO4") == "SO4"
    assert resolve_species_name("OC") == "OC"
    assert resolve_species_name("BC") == "BC"


def test_user_aliases_extend_defaults_for_single_call():
    aliases = {"foo dust": "OIN"}
    assert resolve_species_name("foo_dust", aliases=aliases) == "OIN"
    # defaults still apply while extended
    assert resolve_species_name("dust", aliases=aliases) == "OIN"


def test_user_aliases_override_defaults_for_single_call():
    aliases = {"dust": "OC"}
    assert resolve_species_name("dust", aliases=aliases) == "OC"


def test_resolve_species_names_resolves_sequence_in_order():
    resolved = resolve_species_names(["Dust", "soot", "SO4", "black_carbon"])
    assert resolved == ["OIN", "BC", "SO4", "BC"]


def test_resolve_species_name_rows_resolves_nested_rows_preserving_order_and_shape():
    rows = [["Dust", "SO4"], ["soot", "Org", "NH4"]]
    resolved = resolve_species_name_rows(rows)
    assert resolved == [["OIN", "SO4"], ["BC", "OC", "NH4"]]
    assert len(resolved) == len(rows)
    assert [len(r) for r in resolved] == [len(r) for r in rows]


def test_resolve_species_name_rows_canonical_and_alias_mixed_and_duplicates_preserved():
    rows = [["BC", "black carbon", "BC"], ["OC", "poa", "SO4"]]
    resolved = resolve_species_name_rows(rows)
    assert resolved == [["BC", "BC", "BC"], ["OC", "OC", "SO4"]]
