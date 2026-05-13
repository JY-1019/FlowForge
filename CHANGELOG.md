# Changelog

All notable changes to FlowForge will be documented in this file.

The format is based on Keep a Changelog, and this project uses semantic
versioning after the initial public release.

## [Unreleased]

## [0.1.1] - 2026-05-13

### Added

- Explicit tool input schemas for `FunctionTool`, `HTTPTool`, and `MCPServer`.
- Direct `ctx.call_tool()` support for HTTP, MCP, function, and registry adapter tools.
- Bounded tool-result context via `LLMConfig.max_tool_result_chars`.

### Changed

- `ctx.call_llm()` now assembles global, flow, task, and step prompts into a
  hierarchical system prompt.
- Structured `ctx.call_llm()` responses are validated against step
  `output_schema` before returning to step code.
- Dynamic tool catalogs now include schema-aware call examples and compact
  automatically when needed.

### Security

- Session memory and step history are injected as untrusted context, not as
  executable instructions.

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
