"""Builds structured prompts from DAG + docs for the AI Planner.

Token budget strategy
---------------------
The planner prompt must be as compact as possible while still giving the LLM
enough context to pick the right execution path.

**Flow-only planning**: Route selection operates at the FLOW level only.
Tasks and steps within a flow are determined by their ``order`` and
``branch`` configuration — the planner does not need to reason about them.

What we include (in decreasing token priority):

1. Global system prompt (truncated to 300 chars)
2. Flow hierarchy as an indented tree, each with a route path + summary
3. User request

What we deliberately omit:

- TASK / STEP nodes (execution-level detail, not planning-level)
- Raw DAG edge list (tree structure visible from indentation)
- Full input/output schema descriptions
"""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG, DAGNode
    from flowforge.schema.dag import NodeType
    from flowforge.doc.models import AnyDoc

_SUMMARY_MAX = 120   # chars per summary
_CAPS_MAX = 3        # capabilities per node
_PRE_MAX = 2         # preconditions per node
_GLOBAL_PROMPT_MAX = 300


class PromptBuilder:
    """Assembles the planner prompt from DAG metadata and docs."""

    @staticmethod
    def _shorten(text: str, max_len: int = _SUMMARY_MAX) -> str:
        text = (text or "").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…"

    def build(
        self,
        user_request: Any,
        dag: FlowForgeDAG,
        docs: dict[str, AnyDoc],
        global_prompt: str = "",
    ) -> str:
        from flowforge.schema.dag import NodeType

        parts: list[str] = []

        # 1. Compressed global context
        if global_prompt:
            truncated = global_prompt[:_GLOBAL_PROMPT_MAX]
            if len(global_prompt) > _GLOBAL_PROMPT_MAX:
                truncated += "…"
            parts.append(f"## System\n{truncated}")

        # 2. Flow tree — only FLOW nodes drive path selection.
        #    Tasks/steps are execution details handled by order/branch.
        #    Show as indented tree with route paths for easy selection.
        #    Internal flows (name starts with "_") are excluded — they are
        #    framework-managed and should never be selected by the planner.
        lines: list[str] = []
        for node in dag.get_all_nodes():
            if node.type == NodeType.GLOBAL:
                continue
            if node.type != NodeType.FLOW:
                continue
            # Skip internal flows (e.g. _dynamic_generator).
            if node.name.startswith("_"):
                continue

            # route path = node.id without "global." prefix
            route_path = node.id.removeprefix("global.")
            depth = route_path.count(".")
            indent = "  " * depth

            doc = docs.get(node.id)
            summary = ""
            caps_str = ""
            pre_str = ""
            if doc:
                raw_summary = getattr(doc, "summary", "")
                summary = self._shorten(raw_summary)
                raw_caps = getattr(doc, "capabilities", [])
                caps = raw_caps if isinstance(raw_caps, list) else [str(raw_caps)]
                caps = [str(cap) for cap in caps[:_CAPS_MAX] if str(cap).strip()]
                if caps:
                    caps_str = " | " + "; ".join(caps)
                raw_preconditions = getattr(doc, "preconditions", [])
                preconditions = (
                    raw_preconditions
                    if isinstance(raw_preconditions, list)
                    else [str(raw_preconditions)]
                )
                preconditions = [
                    str(item) for item in preconditions[:_PRE_MAX] if str(item).strip()
                ]
                if preconditions:
                    pre_str = " | pre: " + "; ".join(preconditions)

            if not summary:
                summary = self._shorten(getattr(node.meta, "prompt", ""))

            lines.append(f"{indent}{route_path}  — {summary}{caps_str}{pre_str}")

        routes_text = "\n".join(lines) if lines else "(no user-defined flows available)"
        parts.append("## Available Routes (Flow hierarchy)\n" + routes_text)

        # 3. User request
        if hasattr(user_request, "model_dump"):
            req_str = json.dumps(user_request.model_dump(), ensure_ascii=False)
        else:
            req_str = str(user_request)[:500]   # hard cap to avoid prompt explosion
        parts.append(f"## Request\n{req_str}")

        # 4. Instruction
        parts.append(
            "## Task\n"
            "First decompose the request into ordered requirements, then select "
            "the MINIMUM set of flow routes needed to fulfill them.\n"
            "Return dot-separated route paths (e.g. \"ml_platform.training\") in execution order.\n"
            "- Be SPECIFIC: prefer deeper paths when the request targets a narrow capability.\n"
            "- If the request spans multiple domains, return multiple routes.\n"
            "- Include `requirements[]` with one entry per ordered requirement; "
            "mark whether each is covered by an existing route, needs a new "
            "flow, or needs a new tool/dependency.\n"
            "- If there are no available user-defined routes that can satisfy "
            "the request, set `routes` to [], set `gap_detected` to true, and "
            "fill `suggested_flow_name` plus `suggested_flow_prompt` at the "
            "top level.\n"
            "- If some routes can run now but a missing flow is still required to fully satisfy the request, "
            "return the runnable routes AND report the missing flow as a gap.\n"
            "- When reporting a gap, ALSO set `downstream_flow_route` to the "
            "route of the FIRST flow that will consume the missing flow's output. "
            "The missing flow will be chained immediately BEFORE that flow; the "
            "framework uses this to align input/output schemas between them. "
            "If the missing flow has no downstream consumer, leave it empty.\n"
            "- Tasks and steps within each flow run automatically — do NOT include them.\n"
            "Use the plan_execution tool."
        )

        return "\n\n".join(parts)
