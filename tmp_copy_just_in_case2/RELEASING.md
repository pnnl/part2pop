# Releasing part2pop

This project uses **GitHub Actions + PyPI Trusted Publishing** to automate releases.
Once set up, releasing a new version requires only creating and pushing a Git tag.

No PyPI API tokens are used. Do **not** upload manually with `twine`.

---

## Preconditions (one-time setup)

These steps are assumed to already be complete:

- A GitHub Actions workflow exists at:
  ```
  .github/workflows/release.yml
  ```
- PyPI Trusted Publishing is configured for:
  - Repository: `pnnl/part2pop`
  - Workflow filename: `release.yml`
- The release workflow has permission:
  ```
  id-token: write
  ```

---

## Release process

### 1. Choose the new version number

Follow semantic versioning:

- Patch: `0.1.2` → `0.1.3`
- Minor: `0.1.x` → `0.2.0`
- Major: `1.0.0`

PyPI versions are immutable. Never reuse a version number.

---

### 2. Update the version in the code

Update the version in the appropriate place (e.g. `pyproject.toml`):

```toml
[project]
version = "X.Y.Z"
```

---

### 3. Commit and push the version bump

```bash
git add pyproject.toml
git commit -m "Release vX.Y.Z"
git push
```

The commit **must be pushed before tagging**.

---

### 4. Create and push the release tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Pushing the tag triggers the release workflow.

---

### 5. Automated publishing

After the tag is pushed, GitHub Actions will automatically:

1. Check out the tagged commit
2. Build source and wheel distributions
3. Authenticate to PyPI via Trusted Publishing (OIDC)
4. Upload the new release to PyPI

No manual intervention is required.

---

### 6. Verify the release

After the workflow completes successfully:

- Visit: https://pypi.org/project/part2pop/
- Confirm the new version appears
- Confirm the README renders correctly

Optionally create a GitHub Release pointing to the tag.

---

## If something goes wrong

- Do **not** reuse the same version number
- Fix the issue
- Bump the version again (e.g. `X.Y.(Z+1)`)
- Repeat the release steps

---

## Important notes

- Do not upload releases manually using `twine`
- Do not store PyPI API tokens
- All official releases must be created via Git tags
- The tag name must match the version number exactly (`vX.Y.Z`)

---

## Summary (TL;DR)

```bash
# update version
git commit -am "Release vX.Y.Z"
git push
git tag vX.Y.Z
git push origin vX.Y.Z
```

That’s it.
