# Security Policy

## Supported Versions

FlowForge is pre-1.0. Security fixes are applied to the latest released
version only unless a maintainer announces otherwise.

## Reporting a Vulnerability

Please do not report security vulnerabilities in a public issue.

Use GitHub private vulnerability reporting or open a private security advisory
for the repository when available. Include:

- affected FlowForge version or commit
- a minimal reproduction
- impact and any known workaround
- whether the issue is already public

The project aims to acknowledge valid reports within 7 days.

## Scope

FlowForge can execute user-provided Python functions, dynamic generated code,
local tools, HTTP tools, and MCP servers. Treat agent definitions and generated
flows as trusted application code unless your application adds a sandbox around
them.

