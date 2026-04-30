# ADR 0003: GUI Metadata Source of Truth

## Status

Accepted (directional, post-Priority-1 execution)

## Context

- GUI-related metadata currently includes hard-coded structures (e.g., `viewer/metadata.py`).
- This can drift from actual registered package capabilities.

## Decision

- Long-term, GUI code should consume package-level `describe_*` / `list_*` APIs.
- `viewer/metadata.py` should not remain the long-term source of truth.
- Priority 1 does **not** require completing full GUI migration.
- Priority 1 should document this direction and prepare registry metadata contracts.
- Current registry metadata APIs are stable enough to support future GUI metadata migration work.

## Consequences

- Metadata is generated from the same source that drives runtime behavior.
- Drift risk between GUI options and core builders is reduced.
- GUI migration can proceed incrementally after core registry APIs are stabilized.
