# Changelog

All notable changes to FlowForge will be documented in this file.

The format is based on Keep a Changelog, and this project uses semantic
versioning after the initial public release.

## [Unreleased]

### Added

- Open-source packaging metadata for PyPI publication.
- Source distribution manifest covering examples, tests, docs, and bundled
  runtime assets.
- Packaging tests for bundled docs, bundled skills, PEP 561 marker, and package
  version metadata.
- GitHub Actions workflows for CI and PyPI release publishing.

## [0.1.0] - 2026-04-30

### Added

- Initial annotation-based FlowForge API with `@global_config`, `@flow`,
  `@task`, `@step`, and `FlowForge.compile()`.
- DAG compilation, deterministic execution, route filtering, task loops,
  branch dispatching, tool integration, run traces, and local documentation.

