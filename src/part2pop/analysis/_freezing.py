"""Shared species-override policy for freezing analysis variables."""


def resolve_freezing_config(population, cfg):
    """Copy config with omitted/None overrides inherited and mappings replaced.

    In particular, an explicit empty mapping clears population defaults. Callers
    use the selected mapping read-only without saving it on the population.
    """
    resolved = dict(cfg)
    if resolved.get("species_modifications") is None:
        resolved["species_modifications"] = getattr(
            population, "species_modifications", {}
        )
    return resolved
