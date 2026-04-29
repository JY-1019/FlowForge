"""Phase 2 — Capability selection.

For each step in a :class:`~flowforge.dynamic.plan.WorkflowPlan`, decide
*how* it should be implemented:

* ``llm_only`` — pure ``ctx.call_llm(...)`` with no external tool.
* ``builtin_tool`` — call a registered FunctionTool / HTTPTool /
  generated tool by name (e.g. ``files_write_text``, ``web_fetch_url``).
* ``claude_skill`` — load an Anthropic Claude Skill into the model context.
* ``agent_skill`` — load a vendored ``SKILL.md`` for prompt-side guidance.
* ``mcp`` — invoke a tool exposed by a (possibly auto-provisioned) MCP
  server.

The catalog of registered capabilities is enforced as a strict whitelist:
the LLM can only pick names that are actually registered (or, for ``mcp``,
declared in ``DynamicRunOptions.mcp_server_commands`` /
``mcp_server_urls``).  Any unknown name causes a retry with a corrective
prompt.  This is the structural fix for hallucinated tool names that we
previously caught only after code generation.
"""
from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from flowforge.errors import PlannerError

if TYPE_CHECKING:
    from flowforge.dynamic.plan import WorkflowPlan
    from flowforge.types import DynamicRunOptions, LLMConfig, ToolConfig

logger = logging.getLogger(__name__)


_CAPABILITY_MAX_TOKENS = 6000
_CAPABILITY_RETRIES = 2

CapabilityMode = str  # one of {"llm_only","builtin_tool","claude_skill",
#                                "agent_skill","mcp"}

_VALID_MODES = {
    "llm_only",
    "builtin_tool",
    "claude_skill",
    "agent_skill",
    "mcp",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class StepCapability(BaseModel):
    """How a single planned step should be implemented."""

    step_name: str = Field(
        ...,
        description="Must match one PlannedStep.name in the input plan.",
    )
    mode: CapabilityMode = Field(
        ...,
        description=(
            "One of: llm_only, builtin_tool, claude_skill, agent_skill, mcp."
        ),
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description=(
            "Capability names this step uses.  Empty when mode='llm_only'.  "
            "Must reference the registered catalog verbatim — paraphrasing "
            "or invention is rejected.  When mode='mcp' the names are the "
            "MCP server's tool names (e.g. ['playwright_navigate'])."
        ),
    )
    mcp_server_name: str = Field(
        "",
        description=(
            "When mode='mcp', the registered MCP server identifier from "
            "DynamicRunOptions.mcp_server_commands/urls (e.g. 'playwright', "
            "'figma').  Empty for non-mcp modes."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=5,
        description="Short justification for this mode/tool choice.",
    )

    @field_validator("mcp_server_name", mode="before")
    @classmethod
    def _coerce_none_mcp_server_name(cls, value: Any) -> str:
        return value or ""

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, value: str) -> str:
        if value not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {value!r}"
            )
        return value


class CapabilitySelection(BaseModel):
    """Phase-2 output: one entry per planned step (in plan order)."""

    selections: list[StepCapability] = Field(..., min_length=1)

    def by_step_name(self) -> dict[str, StepCapability]:
        return {sel.step_name: sel for sel in self.selections}


# ---------------------------------------------------------------------------
# Catalog formatting
# ---------------------------------------------------------------------------


def _classify_tool_configs(
    tool_configs: list["ToolConfig"],
) -> dict[str, list[dict[str, str]]]:
    """Bucket registered tools into capability mode groups."""
    from flowforge.types import (
        AgentSkill, ClaudeSkill, FunctionTool, HTTPTool, MCPServer,
    )

    catalog: dict[str, list[dict[str, str]]] = {
        "builtin_tool": [],
        "claude_skill": [],
        "agent_skill": [],
        "mcp_pre_registered": [],
    }
    for tool in tool_configs:
        if isinstance(tool, FunctionTool):
            name = tool.name or (
                tool.func.__name__ if hasattr(tool.func, "__name__") else ""
            )
            if name:
                catalog["builtin_tool"].append({
                    "name": name,
                    "description": tool.description or "",
                })
        elif isinstance(tool, HTTPTool):
            if tool.name:
                catalog["builtin_tool"].append({
                    "name": tool.name,
                    "description": tool.description or "",
                })
        elif isinstance(tool, ClaudeSkill):
            name = tool.name or tool.skill_id
            if name:
                catalog["claude_skill"].append({
                    "name": name,
                    "description": tool.description or "",
                })
        elif isinstance(tool, AgentSkill):
            if tool.name:
                catalog["agent_skill"].append({
                    "name": tool.name,
                    "description": tool.description or "",
                })
        elif isinstance(tool, MCPServer):
            if tool.name:
                catalog["mcp_pre_registered"].append({
                    "name": tool.name,
                    "description": tool.description or "",
                })
    return catalog


def _mcp_options_summary(options: Any) -> dict[str, list[str]]:
    """Return MCP servers declared in DynamicRunOptions but not yet registered."""
    if options is None:
        return {}
    summary: dict[str, list[str]] = {}
    commands = getattr(options, "mcp_server_commands", {}) or {}
    urls = getattr(options, "mcp_server_urls", {}) or {}
    server_tools = getattr(options, "mcp_server_tools", {}) or {}
    for server in set(list(commands.keys()) + list(urls.keys())):
        summary[server] = list(server_tools.get(server, []) or [])
    return summary


def _format_catalog_for_prompt(
    catalog: dict[str, list[dict[str, str]]],
    mcp_options: dict[str, list[str]],
) -> str:
    sections: list[str] = []
    for label, key in (
        ("Built-in tools (FunctionTool / HTTPTool / generated)", "builtin_tool"),
        ("Claude Skills (Anthropic-hosted)", "claude_skill"),
        ("Agent Skills (local SKILL.md, prompt-injected)", "agent_skill"),
        ("Pre-registered MCP servers", "mcp_pre_registered"),
    ):
        entries = catalog.get(key) or []
        if not entries:
            continue
        sections.append(f"### {label}")
        for entry in entries:
            desc = entry.get("description") or "(no description)"
            sections.append(f"- {entry['name']}: {desc}")
        sections.append("")

    if mcp_options:
        sections.append(
            "### MCP servers declared in DynamicRunOptions (auto-provisioned "
            "on demand — pick mode='mcp' and use mcp_server_name=<key>)"
        )
        for server, tools in mcp_options.items():
            tools_text = ", ".join(tools) if tools else "(tools auto-discovered)"
            sections.append(f"- {server} → tools: {tools_text}")
        sections.append("")

    return "\n".join(sections).strip() or "(no capabilities registered)"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_selection(
    selection: CapabilitySelection,
    plan: "WorkflowPlan",
    catalog: dict[str, list[dict[str, str]]],
    mcp_options: dict[str, list[str]],
) -> str | None:
    """Return ``None`` when valid, else an error string for retry feedback."""
    plan_step_names = [step.name for step in plan.steps]
    selection_names = [sel.step_name for sel in selection.selections]

    missing = [name for name in plan_step_names if name not in selection_names]
    if missing:
        return (
            "Capability selection missing entries for planned steps: "
            f"{missing}.  Provide one entry per planned step name."
        )

    extra = [name for name in selection_names if name not in plan_step_names]
    if extra:
        return (
            f"Capability selection references unknown step names: {extra}.  "
            "Use only names from the supplied plan."
        )

    builtin_names = {entry["name"] for entry in catalog.get("builtin_tool", [])}
    claude_names = {entry["name"] for entry in catalog.get("claude_skill", [])}
    agent_names = {entry["name"] for entry in catalog.get("agent_skill", [])}
    pre_mcp = {entry["name"] for entry in catalog.get("mcp_pre_registered", [])}
    mcp_servers = set(mcp_options.keys()) | pre_mcp

    for sel in selection.selections:
        if sel.mode == "llm_only":
            if sel.tool_names:
                return (
                    f"Step {sel.step_name!r}: mode='llm_only' must not "
                    "list tool_names."
                )
            continue

        if not sel.tool_names:
            return (
                f"Step {sel.step_name!r}: mode={sel.mode!r} requires at "
                "least one entry in tool_names."
            )

        if sel.mode == "builtin_tool":
            unknown = [n for n in sel.tool_names if n not in builtin_names]
            if unknown:
                return (
                    f"Step {sel.step_name!r}: builtin_tool names not in "
                    f"registry: {unknown}.  Pick from: "
                    f"{sorted(builtin_names) or '(none registered)'}."
                )
        elif sel.mode == "claude_skill":
            unknown = [n for n in sel.tool_names if n not in claude_names]
            if unknown:
                return (
                    f"Step {sel.step_name!r}: claude_skill names not in "
                    f"registry: {unknown}.  Available: "
                    f"{sorted(claude_names) or '(none registered)'}."
                )
        elif sel.mode == "agent_skill":
            unknown = [n for n in sel.tool_names if n not in agent_names]
            if unknown:
                return (
                    f"Step {sel.step_name!r}: agent_skill names not in "
                    f"registry: {unknown}.  Available: "
                    f"{sorted(agent_names) or '(none registered)'}."
                )
        elif sel.mode == "mcp":
            if not sel.mcp_server_name:
                return (
                    f"Step {sel.step_name!r}: mode='mcp' requires "
                    "mcp_server_name to be set."
                )
            if sel.mcp_server_name not in mcp_servers:
                return (
                    f"Step {sel.step_name!r}: mcp_server_name "
                    f"{sel.mcp_server_name!r} is not declared.  "
                    f"Available: {sorted(mcp_servers) or '(none)'}."
                )
            declared_tools = set(mcp_options.get(sel.mcp_server_name, []))
            if declared_tools:
                unknown = [
                    n for n in sel.tool_names if n not in declared_tools
                ]
                if unknown:
                    return (
                        f"Step {sel.step_name!r}: MCP tool_names not in "
                        f"server {sel.mcp_server_name!r}'s catalog: "
                        f"{unknown}.  Declared: {sorted(declared_tools)}."
                    )
    return None


# ---------------------------------------------------------------------------
# LLM driver
# ---------------------------------------------------------------------------


_CAPABILITY_SYSTEM = textwrap.dedent("""\
    You are FlowForge's capability selector.  Given a workflow plan and the
    catalog of capabilities currently registered in the project, decide HOW
    each planned step should be implemented.

    PRIORITY ORDER (apply in sequence, pick the first that fits):
    1. ``llm_only`` — when the step's purpose is pure reasoning, summary,
       drafting, or analysis that an LLM can do without any external tool.
    2. ``builtin_tool`` — when a registered built-in tool already covers
       the work (HTTP fetches, file I/O, document creation, charts, JSON
       reshaping, package installation, shell, MCP utility helpers, etc.).
    3. ``claude_skill`` / ``agent_skill`` — when a registered skill
       provides specialised guidance the step needs.  Prefer Agent Skills
       (local SKILL.md prompt injection) when working in a code/design
       authoring step, since they work cross-provider.
    4. ``mcp`` — when no built-in tool fits and the catalog declares a
       relevant MCP server (e.g. Playwright for browser automation, Figma
       for design context).  The server is auto-provisioned at runtime.

    HARD RULES
    - Output exactly one entry per planned step, using the step's name.
    - ``tool_names`` MUST contain only names that appear in the catalog or
      the MCP server's declared tools.  Do NOT invent or paraphrase names.
    - For ``mcp`` mode, set ``mcp_server_name`` to a server key from the
      catalog and list its tools in ``tool_names``.
    - ``rationale`` is one short sentence — why this mode/tool fits.
    - When ``needs_llm_reasoning=False`` on the plan, prefer
      ``builtin_tool`` over ``llm_only`` if any tool covers the work.
    - When ``needs_llm_reasoning=True``, ``llm_only`` is a valid choice
      even if tools exist.

    Respond by calling the ``submit_capability_selection`` tool.  Do NOT
    respond with prose.
""")


_CAPABILITY_TOOL_SCHEMA: dict[str, Any] = {
    "name": "submit_capability_selection",
    "description": "Return the capability decision for every planned step.",
    "input_schema": {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "step_name": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": list(sorted(_VALID_MODES)),
                        },
                        "tool_names": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "mcp_server_name": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["step_name", "mode", "rationale"],
                },
            },
        },
        "required": ["selections"],
    },
}


def _format_plan_for_prompt(plan: "WorkflowPlan") -> str:
    lines = [
        f"flow_name: {plan.flow_name}",
        f"flow_prompt: {plan.flow_prompt}",
        f"task_name: {plan.task_name}",
        "steps:",
    ]
    for step in plan.steps:
        branch = ""
        if step.branch is not None:
            branch = (
                f"; branches on field '{step.branch.field}' to "
                f"{list(step.branch.targets.values())}"
            )
        lines.append(
            f"  - name: {step.name}\n"
            f"    order: {step.order}\n"
            f"    needs_llm_reasoning: {step.needs_llm_reasoning}\n"
            f"    purpose: {step.purpose}{branch}"
        )
    return "\n".join(lines)


async def select_capabilities(
    *,
    plan: "WorkflowPlan",
    tool_configs: list["ToolConfig"],
    dynamic_options: Any,
    llm_config: "LLMConfig",
) -> CapabilitySelection:
    """Run Phase 2 — pick one capability mode per planned step.

    Retries up to ``_CAPABILITY_RETRIES`` times when validation rejects
    the selection (unknown tool name, missing step entry, mode/tool
    inconsistency).  The error feedback is appended to the user prompt
    so the LLM can self-correct.
    """
    from flowforge.llm.caller import call_with_tool

    catalog = _classify_tool_configs(tool_configs)
    mcp_options = _mcp_options_summary(dynamic_options)
    catalog_text = _format_catalog_for_prompt(catalog, mcp_options)
    plan_text = _format_plan_for_prompt(plan)

    base_prompt = (
        f"## Workflow plan\n{plan_text}\n\n"
        f"## Capability catalog\n{catalog_text}\n\n"
        "Decide the implementation mode for every step.  Submit via the tool."
    )

    last_error: str | None = None
    for attempt in range(_CAPABILITY_RETRIES + 1):
        prompt = base_prompt
        if last_error:
            prompt += (
                f"\n\n## Previous attempt failed validation\n{last_error}\n"
                "Fix the listed issues and resubmit."
            )

        raw = await call_with_tool(
            prompt=prompt,
            tool_schema=_CAPABILITY_TOOL_SCHEMA,
            llm_config=llm_config,
            system_prompt=_CAPABILITY_SYSTEM,
            max_tokens=_CAPABILITY_MAX_TOKENS,
        )
        try:
            selection = CapabilitySelection.model_validate(raw)
        except Exception as exc:
            last_error = f"Schema validation error: {exc}"
            logger.warning(
                "capability selection attempt %d schema-invalid: %s",
                attempt + 1, exc,
            )
            continue

        err = _validate_selection(selection, plan, catalog, mcp_options)
        if err is None:
            logger.info(
                "capability selection: %s",
                json.dumps(
                    [s.model_dump() for s in selection.selections],
                    ensure_ascii=False,
                ),
            )
            return selection
        last_error = err
        logger.warning(
            "capability selection attempt %d invalid: %s",
            attempt + 1, err,
        )

    raise PlannerError(
        "capability selection validation failed after "
        f"{_CAPABILITY_RETRIES + 1} attempts: {last_error}"
    )


__all__ = [
    "CapabilityMode",
    "StepCapability",
    "CapabilitySelection",
    "select_capabilities",
]
