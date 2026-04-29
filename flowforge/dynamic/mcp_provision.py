"""Phase 3 — MCP auto-provisioning + ``_artifact`` persistence.

When a :class:`~flowforge.dynamic.capability.CapabilitySelection` picks
``mode='mcp'`` for any planned step, we:

1. resolve the server's command / URL / tool list from
   ``DynamicRunOptions``,
2. write a JSON record to ``<generated_dir>/_artifact/mcp/<server>.json``
   so the auto-provisioned server is visible to operators and future
   compilations,
3. extend the dynamic ``manifest.json`` with an ``mcp_servers`` section so
   subsequent compiles know which MCP servers the persisted flow needs.

The actual server start and tool registration are still performed at
**runtime** by ``ctx.call_tool("mcp_register_server", ...)`` calls
generated in Phase 4 — this module only persists the spec.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from flowforge.dynamic.capability import CapabilitySelection
    from flowforge.dynamic.plan import WorkflowPlan
    from flowforge.types import DynamicRunOptions

logger = logging.getLogger(__name__)


_ARTIFACT_DIR_NAME = "_artifact"
_ARTIFACT_MCP_SUBDIR = "mcp"


class McpProvisionRecord(BaseModel):
    """One ``_artifact/mcp/<server>.json`` entry."""

    server_name: str
    url: str = ""
    command: list[str] = Field(default_factory=list)
    headers_keys: list[str] = Field(default_factory=list)
    declared_tools: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    used_by_steps: list[str] = Field(default_factory=list)
    created_by_flow: str = ""
    created_at: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_root(options: "DynamicRunOptions") -> Path:
    """Return ``<generated_dir>/_artifact/mcp`` (created on demand)."""
    from flowforge.dynamic.manifest import resolve_generated_dir

    root = resolve_generated_dir(options) / _ARTIFACT_DIR_NAME / _ARTIFACT_MCP_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _collect_mcp_uses(
    selection: "CapabilitySelection",
) -> dict[str, dict[str, set[str]]]:
    """Group capability selections by MCP server.

    Returns ``{server_name: {"tools": {...}, "steps": {...}}}``.
    """
    grouped: dict[str, dict[str, set[str]]] = {}
    for sel in selection.selections:
        if sel.mode != "mcp" or not sel.mcp_server_name:
            continue
        bucket = grouped.setdefault(
            sel.mcp_server_name, {"tools": set(), "steps": set()},
        )
        bucket["tools"].update(sel.tool_names)
        bucket["steps"].add(sel.step_name)
    return grouped


def provision_mcp_servers(
    *,
    selection: "CapabilitySelection",
    plan: "WorkflowPlan",
    options: "DynamicRunOptions | None",
) -> list[McpProvisionRecord]:
    """Persist a spec record for every MCP server referenced by *selection*.

    Returns the list of provisioned records (possibly empty).  When
    ``options`` is ``None`` or no MCP-mode steps exist, this is a no-op.
    """
    if options is None:
        return []

    grouped = _collect_mcp_uses(selection)
    if not grouped:
        return []

    records: list[McpProvisionRecord] = []
    artifact_root = _artifact_root(options)

    commands_map = getattr(options, "mcp_server_commands", {}) or {}
    urls_map = getattr(options, "mcp_server_urls", {}) or {}
    tools_map = getattr(options, "mcp_server_tools", {}) or {}
    headers_map = getattr(options, "mcp_server_headers", {}) or {}

    for server_name, usage in grouped.items():
        record = McpProvisionRecord(
            server_name=server_name,
            url=urls_map.get(server_name, ""),
            command=list(commands_map.get(server_name, []) or []),
            headers_keys=sorted(
                (headers_map.get(server_name, {}) or {}).keys()
            ),
            declared_tools=list(tools_map.get(server_name, []) or []),
            selected_tools=sorted(usage["tools"]),
            used_by_steps=sorted(usage["steps"]),
            created_by_flow=plan.flow_name,
            created_at=_utc_now(),
        )
        artifact_path = artifact_root / f"{server_name}.json"
        artifact_path.write_text(
            json.dumps(record.model_dump(), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "MCP artefact persisted: %s (server=%s, tools=%d, used_by=%s)",
            artifact_path, server_name,
            len(record.selected_tools), record.used_by_steps,
        )
        records.append(record)

    if records and getattr(options, "persist_generated", False):
        _merge_into_manifest(records, options)

    return records


def _merge_into_manifest(
    records: list[McpProvisionRecord],
    options: "DynamicRunOptions",
) -> None:
    """Extend the dynamic manifest with an ``mcp_servers`` section."""
    from flowforge.dynamic.manifest import (
        _manifest_lock,
        load_manifest,
        save_manifest,
    )

    with _manifest_lock(options):
        manifest = load_manifest(options)
        existing = getattr(manifest, "mcp_servers", None)
        # ``DynamicManifest`` is extended below to include ``mcp_servers``;
        # be defensive in case an older manifest file is on disk.
        if existing is None:
            existing = []
        names_to_replace = {record.server_name for record in records}
        merged = [
            entry for entry in existing
            if getattr(entry, "server_name", "") not in names_to_replace
        ] + records
        manifest.mcp_servers = merged
        save_manifest(options, manifest)


__all__ = [
    "McpProvisionRecord",
    "provision_mcp_servers",
]
