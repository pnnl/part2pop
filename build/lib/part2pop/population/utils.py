#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for population building.

Provide expansion of compound-like species names into tracked species
before particle construction. Examples: "NaCl", "(NH4)2SO4", "NH4(SO4)2".

Notes:
- Only factories should call this; core classes remain unchanged.
- H2O is handled separately by make_particle and is not added here.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from part2pop.species.registry import get_species
from part2pop.data import open_dataset

def _read_available_species_tokens() -> List[str]:
    """Read available species names from datasets/species_data/aero_data.dat.

    Returns a list of known tokens (species names) sorted by length (desc)
    to enable greedy longest-match parsing.
    """
    #species_file = Path(data_path) / "species_data" / "aero_data.dat"
    tokens: List[str] = []
    with open_dataset("aero_data.dat") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                tokens.append(parts[0])
    tokens.sort(key=len, reverse=True)
    return tokens


def _parse_formula(formula: str, known_tokens: List[str]) -> Dict[str, int]:
    """Parse a limited chemical-like formula into counts of known tokens.

    Supports parentheses groups and integer multipliers, and matches tokens
    greedily from the provided known_tokens list. Raises ValueError for
    unknown tokens or malformed strings.
    """
    s = formula.strip()
    n = len(s)
    i = 0

    def parse_group() -> Dict[str, int]:
        nonlocal i
        counts: Dict[str, int] = {}
        while i < n and s[i] != ')':
            if s[i] == '(':
                i += 1  # consume '('
                inner = parse_group()
                if i >= n or s[i] != ')':
                    raise ValueError(f"Unmatched '(' in formula: {formula}")
                i += 1  # consume ')'
                # optional multiplier
                start = i
                while i < n and s[i].isdigit():
                    i += 1
                mult = int(s[start:i]) if i > start else 1
                for k, v in inner.items():
                    counts[k] = counts.get(k, 0) + v * mult
                continue

            # match a known token greedily
            matched = None
            for tok in known_tokens:
                if s.startswith(tok, i):
                    matched = tok
                    break
            if matched is None:
                raise ValueError(
                    f"Unknown token starting at '{s[i:]}' in formula '{formula}'"
                )
            i += len(matched)
            # optional integer immediately after the token
            start = i
            while i < n and s[i].isdigit():
                i += 1
            mult = int(s[start:i]) if i > start else 1
            counts[matched] = counts.get(matched, 0) + mult
        return counts

    result = parse_group()
    if i != n:
        if i < n and s[i] == ')':
            raise ValueError(f"Unmatched ')' in formula: {formula}")
        raise ValueError(f"Unexpected trailing characters in formula: '{s[i:]}'")
    return result


def _mass_fraction_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert token counts into mass fractions using get_species(...).molar_mass."""
    total = 0.0
    molars: Dict[str, float] = {}
    for token, count in counts.items():
        mm = get_species(token).molar_mass
        molars[token] = mm
        total += count * mm
    if total <= 0.0:
        raise ValueError("Computed non-positive molar mass for counts")
    return {token: (counts[token] * molars[token]) / total for token in counts}


def _normalize(fracs: List[float]) -> List[float]:
    s = float(sum(fracs))
    if s <= 0:
        return fracs
    return [f / s for f in fracs]


def expand_compounds_for_population(
    names_list: List[List[str]],
    fracs_list: List[List[float]]
) -> Tuple[List[List[str]], List[List[float]]]:
    """Expand compound species names into constituent species and compute mass fractions.

    Per-mode, expands entries like "NaCl" or "(NH4)2SO4" into tracked species
    tokens using molar masses. If a name is already a known species token, it is
    passed through unchanged. Duplicates are merged and fractions normalized.
    """
    if len(names_list) != len(fracs_list):
        raise ValueError("names_list and fracs_list must have the same length")

    known_tokens = _read_available_species_tokens()

    out_names_list: List[List[str]] = []
    out_fracs_list: List[List[float]] = []

    for names, fracs in zip(names_list, fracs_list):
        if len(names) != len(fracs):
            raise ValueError("Each names sublist must match length of fracs sublist")

        merged: "OrderedDict[str, float]" = OrderedDict()

        for name, frac in zip(names, fracs):
            name = str(name)
            frac = float(frac)
            if frac == 0.0:
                continue

            if name in known_tokens:
                if name not in merged:
                    merged[name] = 0.0
                merged[name] += frac
                continue

            counts = _parse_formula(name, known_tokens)
            mass_fracs = _mass_fraction_from_counts(counts)
            for token, mf in mass_fracs.items():
                if token not in merged:
                    merged[token] = 0.0
                merged[token] += frac * mf

        out_names = list(merged.keys())
        out_fracs = _normalize(list(merged.values()))
        out_names_list.append(out_names)
        out_fracs_list.append(out_fracs)

    return out_names_list, out_fracs_list

    result = parse_group()
    if i != n:
        # Should only be extra ')' which implies mismatch
        if i < n and s[i] == ')':
            raise ValueError(f"Unmatched ')' in formula: {formula}")
        # otherwise trailing garbage
        raise ValueError(f"Unexpected trailing characters in formula: '{s[i:]}'")
    return result


def _mass_fraction_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert token counts into mass fractions using get_species(...).molar_mass."""
    # total molar mass
    total = 0.0
    molars: Dict[str, float] = {}
    for token, count in counts.items():
        mm = get_species(token).molar_mass
        molars[token] = mm
        total += count * mm
    if total <= 0.0:
        raise ValueError("Computed non-positive molar mass for counts")
    return {token: (counts[token] * molars[token]) / total for token in counts}


def _normalize(fracs: List[float]) -> List[float]:
    s = float(sum(fracs))
    if s <= 0:
        return fracs
    return [f / s for f in fracs]


def expand_compounds_for_population(
    names_list: List[List[str]],
    fracs_list: List[List[float]]
) -> Tuple[List[List[str]], List[List[float]]]:
    """Expand compound species names into constituent species and compute mass fractions.

    For each mode's list of names and fractions, expands entries like "NaCl"
    or "(NH4)2SO4" into tracked species tokens using molar masses. If a name
    is already a known species token, it is passed through unchanged.

    Returns new lists of names and fractions per mode with duplicates merged
    and fractions normalized to sum to 1.0.
    """
    if len(names_list) != len(fracs_list):
        raise ValueError("names_list and fracs_list must have the same length")

    known_tokens = _read_available_species_tokens()

    out_names_list: List[List[str]] = []
    out_fracs_list: List[List[float]] = []
    for names, fracs in zip(names_list, fracs_list):
        if len(names) != len(fracs):
            raise ValueError("Each names sublist must match length of fracs sublist")

        merged: "OrderedDict[str, float]" = OrderedDict()

        for name, frac in zip(names, fracs):
            name = str(name)
            frac = float(frac)
            if frac == 0.0:
                # skip zero mass fraction entries
                continue

            if name in known_tokens:
                # already a tracked species
                if name not in merged:
                    merged[name] = 0.0
                merged[name] += frac
                continue

            # attempt to parse as a compound formula
            counts = _parse_formula(name, known_tokens)
            mass_fracs = _mass_fraction_from_counts(counts)

            for token, mf in mass_fracs.items():
                if token not in merged:
                    merged[token] = 0.0
                merged[token] += frac * mf

        out_names = list(merged.keys())
        out_fracs = list(merged.values())
        out_fracs = _normalize(out_fracs)

        out_names_list.append(out_names)
        out_fracs_list.append(out_fracs)

    return out_names_list, out_fracs_list
