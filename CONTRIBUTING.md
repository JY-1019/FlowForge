# Contributing to FlowForge

Thanks for helping make FlowForge better. The project is still alpha, so small,
well-scoped changes with clear tests are the easiest to review and merge.

## Development Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all]"
```

## Local Checks

For normal changes, start with the tests closest to the behavior you touched:

```bash
python -m pytest tests/test_annotations.py -q
```

Before opening a pull request, run the full regression suite and packaging
checks:

```bash
python -m pytest tests/ -x -q
python -m build
python -m twine check dist/*
python -m check_manifest
```

## Packaging Notes

FlowForge ships runtime data in the wheel, including:

- bundled MkDocs documentation under `flowforge/_docs/`
- curated Anthropic Agent Skills under `flowforge/skills/anthropic/`
- `flowforge/py.typed` for PEP 561 type discovery

When adding new non-Python files used at runtime, update both `pyproject.toml`
and `MANIFEST.in`, then add or update a packaging test.

## Pull Request Guidelines

- Keep edits focused on the requested behavior.
- Add tests for public API changes, packaging changes, and regressions.
- Document new user-facing behavior in `README.md` or `flowforge/_docs/`.
- Do not commit generated build outputs such as `build/`, `dist/`,
  `*.egg-info/`, `.flowforge_cache/`, or MkDocs `site/`.

