"""LLM-based doc generation for FlowForge nodes."""
from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._llm_config = llm_config
        self._cache = cache or DocCache()
        self._force = force_regenerate

    async def generate_all(self, nodes: list[DAGNode]) -> dict[str, AnyDoc]:
        """Generate docs for all provided nodes. Returns node_id → doc mapping."""
        docs: dict[str, AnyDoc] = {}
        for node in nodes:
            doc = await self.generate_node(node)
            if doc is not None:
                docs[node.id] = doc
                node.doc = doc
        return docs

    async def generate_node(self, node: DAGNode) -> AnyDoc | None:
        """Generate doc for a single node, using cache when possible."""
        prompt = self._build_prompt(node)

        if not self._force:
            cached = self._cache.get(prompt)
            if cached:
                return self._dict_to_doc(node.type, cached)

        try:
            raw = await self._call_llm(prompt, node)
            self._cache.set(prompt, raw)
            return self._dict_to_doc(node.type, raw)
        except Exception as e:
            logger.warning(
                "doc generation failed for node '%s' (provider=%s), "
                "using prompt-only fallback: %s",
                node.id, self._llm_config.provider, e,
            )
            return self._fallback_doc(node)

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

    def _fallback_doc(self, node: DAGNode) -> AnyDoc:
        raw = getattr(node.meta, "prompt", "") or f"{node.type.value}: {node.name}"
        return self._dict_to_doc(node.type, {"summary": raw[:200]})
