"""Persistence helpers for dynamically generated flows and tools."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from flowforge.types import DynamicRunOptions, FunctionTool, ToolConfig

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"


class GeneratedFlowRecord(BaseModel):
    """Manifest entry for one generated ``@flow`` file."""

    name: str
    file: str
    class_name: str = ""
    inject_before: str = ""
    inject_after: str = ""
    downstream_flow_route: str = ""
    bridge: Literal["contract", "shared_data", ""] = ""
    created_at: str = ""


class GeneratedToolRecord(BaseModel):
    """Manifest entry for one generated tool file."""

    name: str
    file: str
    symbol: str = ""
    kind: str = "function"
    dependencies: list[str] = Field(default_factory=list)
    created_at: str = ""


class DynamicManifest(BaseModel):
    """Manifest persisted under the dynamic generated directory."""

    version: int = 1
    flows: list[GeneratedFlowRecord] = Field(default_factory=list)
    tools: list[GeneratedToolRecord] = Field(default_factory=list)
    # ``mcp_servers`` is populated by Phase 3 of dynamic generation.  It
    # records auto-provisioned MCP servers so subsequent compiles know
    # which external services the persisted flows depend on.  The element
    # type is a forward reference (``Any`` here) to avoid an import cycle
    # with ``flowforge.dynamic.mcp_provision``; that module's
    # ``McpProvisionRecord`` matches this shape.
    mcp_servers: list[Any] = Field(default_factory=list)


def resolve_project_root(options: DynamicRunOptions) -> Path:
    """Return the project root used for generated assets."""
    return Path(options.project_root or Path.cwd()).expanduser().resolve()


def resolve_generated_dir(options: DynamicRunOptions) -> Path:
    """Return the generated directory, constrained inside project root."""
    root = resolve_project_root(options)
    raw = Path(options.generated_dir).expanduser()
    generated = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        generated.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"dynamic generated_dir must be inside project_root: {generated}"
        ) from exc
    return generated


def manifest_path(options: DynamicRunOptions) -> Path:
    return resolve_generated_dir(options) / _MANIFEST_NAME


# ---------------------------------------------------------------------------
# File locking — prevents concurrent manifest corruption
# ---------------------------------------------------------------------------

@contextmanager
def _manifest_lock(options: DynamicRunOptions):
    """Acquire an exclusive file lock around manifest reads/writes.

    Uses ``fcntl.flock`` (available on macOS and Linux).  The lock file is
    created next to the manifest as ``manifest.json.lock``.
    """
    generated = resolve_generated_dir(options)
    generated.mkdir(parents=True, exist_ok=True)
    lock_path = generated / f"{_MANIFEST_NAME}.lock"
    fd = lock_path.open("w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


# ---------------------------------------------------------------------------
# Manifest I/O — always use file locking
# ---------------------------------------------------------------------------

def load_manifest(options: DynamicRunOptions) -> DynamicManifest:
    path = manifest_path(options)
    if not path.exists():
        return DynamicManifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    return DynamicManifest.model_validate(data)


def save_manifest(options: DynamicRunOptions, manifest: DynamicManifest) -> Path:
    root = resolve_generated_dir(options)
    root.mkdir(parents=True, exist_ok=True)
    path = root / _MANIFEST_NAME
    path.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Persistence helpers — locked read-modify-write
# ---------------------------------------------------------------------------

def persist_flow_code(
    *,
    flow_name: str,
    code: str,
    options: DynamicRunOptions,
    class_name: str = "",
    inject_before: str = "",
    inject_after: str = "",
    downstream_flow_route: str = "",
    bridge: Literal["contract", "shared_data", ""] = "",
) -> GeneratedFlowRecord:
    """Write generated flow code and update ``manifest.json`` atomically."""
    with _manifest_lock(options):
        generated = resolve_generated_dir(options)
        flows_dir = generated / "flows"
        flows_dir.mkdir(parents=True, exist_ok=True)
        file_path = flows_dir / f"{flow_name}.py"
        file_path.write_text(code.rstrip() + "\n", encoding="utf-8")

        manifest = load_manifest(options)
        record = GeneratedFlowRecord(
            name=flow_name,
            file=str(file_path.relative_to(resolve_project_root(options))),
            class_name=class_name,
            inject_before=inject_before,
            inject_after=inject_after,
            downstream_flow_route=downstream_flow_route,
            bridge=bridge or ("contract" if downstream_flow_route else ""),
            created_at=_utc_now(),
        )
        manifest.flows = [item for item in manifest.flows if item.name != flow_name]
        manifest.flows.append(record)
        save_manifest(options, manifest)
    return record


def persist_tool_code(
    *,
    tool_name: str,
    code: str,
    options: DynamicRunOptions,
    symbol: str = "",
    dependencies: list[str] | None = None,
) -> GeneratedToolRecord:
    """Write generated tool code and update ``manifest.json`` atomically."""
    with _manifest_lock(options):
        generated = resolve_generated_dir(options)
        tools_dir = generated / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        file_path = tools_dir / f"{tool_name}.py"
        file_path.write_text(code.rstrip() + "\n", encoding="utf-8")

        manifest = load_manifest(options)
        record = GeneratedToolRecord(
            name=tool_name,
            file=str(file_path.relative_to(resolve_project_root(options))),
            symbol=symbol,
            dependencies=dependencies or [],
            created_at=_utc_now(),
        )
        manifest.tools = [item for item in manifest.tools if item.name != tool_name]
        manifest.tools.append(record)
        save_manifest(options, manifest)
    return record


def load_generated_assets(global_meta: Any, options: DynamicRunOptions) -> None:
    """Load generated tools and flows from the dynamic manifest into metadata."""
    manifest = load_manifest(options)
    root = resolve_project_root(options)

    existing_tool_names = {_tool_name(tool) for tool in global_meta.tools}
    for record in manifest.tools:
        try:
            tool = _load_tool_record(record, root)
        except Exception as exc:
            logger.warning("failed to load generated tool %s: %s", record.name, exc)
            continue
        name = _tool_name(tool)
        if name and name not in existing_tool_names:
            global_meta.tools.append(tool)
            existing_tool_names.add(name)

    existing_flow_names = {flow.name for flow in global_meta.flows}
    for record in manifest.flows:
        try:
            flow_meta = _load_flow_record(record, root)
        except Exception as exc:
            logger.warning("failed to load generated flow %s: %s", record.name, exc)
            continue
        if flow_meta.name not in existing_flow_names:
            from flowforge.dynamic.defaults import apply_dynamic_defaults

            apply_dynamic_defaults(flow_meta, options)
            source_path = root / record.file
            try:
                source_code = source_path.read_text(encoding="utf-8")
            except OSError:
                source_code = ""
            setattr(flow_meta, "__flowforge_dynamic_source_code__", source_code)
            setattr(flow_meta, "__flowforge_dynamic_manifest_record__", record)
            setattr(
                flow_meta,
                "__flowforge_dynamic_downstream_flow_route__",
                record.downstream_flow_route,
            )
            _insert_generated_flow(global_meta.flows, flow_meta, record)
            existing_flow_names.add(flow_meta.name)


def _insert_generated_flow(
    flows: list[Any],
    flow_meta: Any,
    record: GeneratedFlowRecord,
) -> None:
    """Insert generated bridge flows before their declared downstream flow."""

    target = (
        record.inject_before
        or record.downstream_flow_route
        or ""
    ).removeprefix("global.").split(".", 1)[0]
    if target:
        for index, existing in enumerate(flows):
            if getattr(existing, "name", "") == target:
                flows.insert(index, flow_meta)
                return
    flows.append(flow_meta)


def _load_flow_record(record: GeneratedFlowRecord, root: Path) -> Any:
    from flowforge.annotations.decorators import _FLOW_ATTR

    module = _import_file(root / record.file, f"_flowforge_generated_flow_{record.name}")
    if record.class_name:
        obj = getattr(module, record.class_name)
        return getattr(obj, _FLOW_ATTR)
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and hasattr(obj, _FLOW_ATTR):
            return getattr(obj, _FLOW_ATTR)
    raise ValueError(f"generated file for {record.name!r} contains no @flow class")


def _load_tool_record(record: GeneratedToolRecord, root: Path) -> ToolConfig:
    module = _import_file(root / record.file, f"_flowforge_generated_tool_{record.name}")
    if hasattr(module, "TOOL_CONFIG"):
        config = getattr(module, "TOOL_CONFIG")
        return config

    symbol = record.symbol or record.name
    func = getattr(module, symbol)
    return FunctionTool(func=func, name=record.name)


def _import_file(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import generated file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _tool_name(tool: Any) -> str:
    from flowforge.types import MCPServer, FunctionTool, HTTPTool, ClaudeSkill, AgentSkill

    if isinstance(tool, MCPServer):
        return tool.name
    if isinstance(tool, FunctionTool):
        return tool.name or (
            tool.func.__name__ if hasattr(tool.func, "__name__") else ""
        )
    if isinstance(tool, HTTPTool):
        return tool.name
    if isinstance(tool, ClaudeSkill):
        return tool.name or tool.skill_id
    if isinstance(tool, AgentSkill):
        return tool.name
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
