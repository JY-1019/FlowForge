"""Builds structured prompts from DAG + docs for the AI Planner.

Token budget strategy
---------------------
The planner prompt must be as compact as possible while still giving the LLM
enough context to pick the right execution path.

What we include (in decreasing token priority):
1. Global system prompt (truncated to 300 chars)
2. Per-node one-liner: type + id + summary (no edges — tree shape is implied by ids)
3. Capabilities only for FLOW and TASK nodes (3 items max)
4. User request

What we deliberately omit:
- Raw DAG edge list (redundant: the dotted IDs encode the tree already)
- Step/branch summaries in the overview (implementation detail, not path-selection level)
- Full input/output schema descriptions (available inside doc if needed at runtime)
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

        # 2. Node registry — only FLOW and TASK nodes drive path selection.
        #    STEP/BRANCH are execution details, not planning-level choices.
        planning_types = {NodeType.GLOBAL, NodeType.FLOW, NodeType.TASK}
        lines: list[str] = []
        for node in dag.get_all_nodes():
            if node.type not in planning_types:
                continue
            doc = docs.get(node.id)
            summary = ""
            caps_str = ""
            if doc:
                raw_summary = getattr(doc, "summary", "")
                summary = raw_summary[:_SUMMARY_MAX] + ("…" if len(raw_summary) > _SUMMARY_MAX else "")
                caps: list[str] = getattr(doc, "capabilities", [])[:_CAPS_MAX]
                if caps:
                    caps_str = " | " + "; ".join(caps)
            lines.append(f"[{node.type.value}] {node.id}  {summary}{caps_str}")

        parts.append("## Available Nodes\n" + "\n".join(lines))

        # 3. User request
        if hasattr(user_request, "model_dump"):
            req_str = json.dumps(user_request.model_dump(), ensure_ascii=False)
        else:
            req_str = str(user_request)[:500]   # hard cap to avoid prompt explosion
        parts.append(f"## Request\n{req_str}")

        # 4. Instruction
        parts.append(
            "## Task\n"
            "Return the ordered list of node IDs that should execute to fulfill this request. "
            "Only include FLOW and TASK node IDs. Use the plan_execution tool."
        )

        return "\n\n".join(parts)
