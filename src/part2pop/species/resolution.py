"""Species-name resolution helpers.

This module maps source/instrument labels to canonical part2pop species names
without defining any species physical properties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re


_SEPARATOR_RE = re.compile(r"[\s_\-]+")


def _normalize_label(name: str) -> str:
    """Normalize a species label for alias lookup.

    Normalization is case-insensitive and treats spaces, underscores, and
    hyphens consistently.
    """
    text = str(name).strip().casefold()
    text = _SEPARATOR_RE.sub(" ", text).strip()
    return text


_DEFAULT_ALIASES: dict[str, str] = {
    _normalize_label("dust"): "OIN",
    _normalize_label("soot"): "BC",
    _normalize_label("black carbon"): "BC",
    _normalize_label("org"): "OC",
    _normalize_label("poa"): "OC",
}


def resolve_species_name(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Resolve one source label to a canonical species name.

    Parameters
    ----------
    name
        Input species label from a source or builder config.
    aliases
        Optional call-scoped aliases that extend/override defaults.

    Returns
    -------
    str
        Canonical species name if matched; otherwise the stripped original
        name (pass-through behavior).
    """
    stripped = str(name).strip()
    merged = dict(_DEFAULT_ALIASES)

    if aliases:
        for alias_key, canonical in aliases.items():
            merged[_normalize_label(alias_key)] = str(canonical)

    return merged.get(_normalize_label(stripped), stripped)


def resolve_species_names(
    names: Sequence[str], aliases: Mapping[str, str] | None = None
) -> list[str]:
    """Resolve a sequence of source labels in order."""
    return [resolve_species_name(name, aliases=aliases) for name in names]


def resolve_species_name_rows(
    name_rows: Sequence[Sequence[str]], aliases: Mapping[str, str] | None = None
) -> list[list[str]]:
    """Resolve a nested sequence of source labels while preserving shape/order."""
    return [resolve_species_names(row, aliases=aliases) for row in name_rows]
