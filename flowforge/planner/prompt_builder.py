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
_GLOBAL_PROMPT_MAX = 300


class PromptBuilder:
    """Assembles the planner prompt from DAG metadata and docs."""

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
        lines: list[str] = []
        for node in dag.get_all_nodes():
            if node.type == NodeType.GLOBAL:
                continue
            if node.type != NodeType.FLOW:
                continue

            # route path = node.id without "global." prefix
            route_path = node.id.removeprefix("global.")
            depth = route_path.count(".")
            indent = "  " * depth

            doc = docs.get(node.id)
            summary = ""
            caps_str = ""
            if doc:
                raw_summary = getattr(doc, "summary", "")
                summary = raw_summary[:_SUMMARY_MAX] + ("…" if len(raw_summary) > _SUMMARY_MAX else "")
                caps: list[str] = getattr(doc, "capabilities", [])[:_CAPS_MAX]
                if caps:
                    caps_str = " | " + "; ".join(caps)
            lines.append(f"{indent}{route_path}  — {summary}{caps_str}")

        parts.append("## Available Routes (Flow hierarchy)\n" + "\n".join(lines))

        # 3. User request
        if hasattr(user_request, "model_dump"):
            req_str = json.dumps(user_request.model_dump(), ensure_ascii=False)
        else:
            req_str = str(user_request)[:500]   # hard cap to avoid prompt explosion
        parts.append(f"## Request\n{req_str}")

        # 4. Instruction
        parts.append(
            "## Task\n"
            "Select the MINIMUM set of flow routes needed to fulfill this request.\n"
            "Return dot-separated route paths (e.g. \"ml_platform.training\").\n"
            "- Be SPECIFIC: prefer deeper paths when the request targets a narrow capability.\n"
            "- If the request spans multiple domains, return multiple routes.\n"
            "- Tasks and steps within each flow run automatically — do NOT include them.\n"
            "Use the plan_execution tool."
        )

        return "\n\n".join(parts)
