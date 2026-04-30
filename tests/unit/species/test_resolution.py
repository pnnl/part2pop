from part2pop.species import resolve_species_name, resolve_species_names


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
