"""LLM-based doc generation for FlowForge nodes."""
from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from flowforge.doc.cache import DocCache
from flowforge.doc.models import (
    StepDoc,
    BranchDoc,
    TaskDoc,
    FlowDoc,
    GlobalDoc,
    AnyDoc,
)
from flowforge.llm.caller import call_with_tool
from flowforge.schema.dag import DAGNode, NodeType
from flowforge.errors import DocGenerationError

if TYPE_CHECKING:
    from flowforge.types import LLMConfig

# Default concurrency limit for parallel doc generation.
_DEFAULT_CONCURRENCY = 5

_DOC_TOOL_SCHEMA = {
    "name": "generate_doc",
    "description": "Generate structured documentation for a FlowForge node",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "1-2 sentence summary"},
            "input_schema_desc": {
                "type": "object",
                "description": "Field-by-field description of the input schema",
                "additionalProperties": {"type": "string"},
            },
            "output_schema_desc": {
                "type": "object",
                "description": "Field-by-field description of the output schema",
                "additionalProperties": {"type": "string"},
            },
            "preconditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions required before executing this node",
            },
            "capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What this node can do",
            },
            "children_overview": {
                "type": "string",
                "description": "Overview of child nodes (flows/tasks only)",
            },
            "routing_hints": {
                "type": "object",
                "description": "Branch-specific: when to choose which path",
                "additionalProperties": {"type": "string"},
            },
            "agent_overview": {
                "type": "string",
                "description": "Global agent overview",
            },
        },
        "required": ["summary"],
    },
}

# Tighter system prompt saves ~40 tokens per LLM call
_SYSTEM_PROMPT = (
    "You are a technical documentation assistant for an AI agent framework. "
    "Generate concise structured docs. Be specific and brief."
)


class DocGenerator:
    """Generates AI documentation for each node.

    Supports all three LLM providers — Anthropic, OpenAI, and Google Gemini —
    by delegating to :func:`flowforge.llm.caller.call_with_tool`.  The active
    provider is determined by ``llm_config.provider``.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        cache: DocCache | None = None,
        force_regenerate: bool = False,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        self._llm_config = llm_config
        self._cache = cache or DocCache()
        self._force = force_regenerate
        self._concurrency = concurrency

    async def generate_all(
        self,
        nodes: list[DAGNode],
        *,
        node_types: set[NodeType] | None = None,
    ) -> dict[str, AnyDoc]:
        """Generate docs for provided nodes, optionally filtered by type.

        Parameters
        ----------
        nodes:
            All DAG nodes.
        node_types:
            When set, only generate docs for nodes whose type is in this set.
            For planning-only doc generation, pass ``{NodeType.GLOBAL, NodeType.FLOW}``
            to skip task/step nodes and dramatically reduce LLM calls.

        Returns node_id → doc mapping.
        """
        targets = nodes
        if node_types is not None:
            targets = [n for n in nodes if n.type in node_types]

        # Use concurrent generation with a semaphore to limit parallelism.
        sem = asyncio.Semaphore(self._concurrency)
        docs: dict[str, AnyDoc] = {}

        async def _gen(node: DAGNode) -> tuple[str, AnyDoc | None]:
            async with sem:
                doc = await self.generate_node(node)
                return node.id, doc

        results = await asyncio.gather(*[_gen(n) for n in targets])
        for result in results:
            node_id, doc = result
            if doc is not None:
                docs[node_id] = doc
                # Also attach to the node object
                node_obj = next((n for n in targets if n.id == node_id), None)
                if node_obj:
                    node_obj.doc = doc
        return docs

    async def generate_node(self, node: DAGNode) -> AnyDoc | None:
        """Generate doc for a single node, using cache when possible."""
        prompt = self._build_prompt(node)

        if not self._force:
            cached = self._cache.get(prompt)
            if cached:
                return self._dict_to_doc(node.type, cached)

        raw = await self._call_llm(prompt, node)
        self._cache.set(prompt, raw)
        return self._dict_to_doc(node.type, raw)

    def _build_prompt(self, node: DAGNode) -> str:
        meta = node.meta
        parts = [
            f"type: {node.type.value}",
            f"name: {node.name}",
            f"prompt: {getattr(meta, 'prompt', '')}",
        ]

        for schema_attr in ("input_schema", "output_schema"):
            schema = getattr(meta, schema_attr, None)
            if schema is not None:
                try:
                    parts.append(f"{schema_attr}: {json.dumps(schema.model_json_schema())}")
                except Exception:
                    pass

        if getattr(meta, "condition", None):
            parts.append(f"branch_field: {meta.condition.field}")
            parts.append(f"branch_options: {meta.condition.enum}")

        if getattr(meta, "child_flows", None):
            parts.append(f"child_flows: {[f.name for f in meta.child_flows]}")

        if getattr(meta, "tasks", None):
            parts.append(f"tasks: {[t.name for t in meta.tasks]}")

        if getattr(meta, "steps", None):
            parts.append(f"steps: {[(s.order, s.prompt) for s in meta.steps]}")

        return "\n".join(parts)

    async def _call_llm(self, prompt: str, node: DAGNode) -> dict[str, Any]:
        user_msg = f"Document this FlowForge node:\n\n{prompt}"
        try:
            return await call_with_tool(
                prompt=user_msg,
                tool_schema=_DOC_TOOL_SCHEMA,
                llm_config=self._llm_config,
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=512,
            )
        except Exception as e:
            raise DocGenerationError(
                f"LLM call failed for node '{node.id}' "
                f"(provider={self._llm_config.provider}): {e}"
            ) from e

    def _dict_to_doc(self, node_type: NodeType, data: dict[str, Any]) -> AnyDoc:
        model_map = {
            NodeType.STEP: StepDoc,
            NodeType.TASK: TaskDoc,
            NodeType.FLOW: FlowDoc,
        }
        model = model_map.get(node_type, GlobalDoc)
        return model(**{k: v for k, v in data.items() if k in model.model_fields})
