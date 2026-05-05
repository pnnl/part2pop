# Contributing to part2pop

Thank you for your interest in improving `part2pop`.

## Support and feature requests

- Report bugs using GitHub Issues.
- Request new features using GitHub Issues.

## Contribution workflow

- Use pull requests for all contributions.
- Create a branch for your change and keep each pull request focused on a single topic/scope.

## Pre-PR checks

Before opening a pull request, run:

```bash
pytest tests/unit -q
pytest tests/integration -q
```

For docs-only changes, also run:

```bash
git diff --check
```
