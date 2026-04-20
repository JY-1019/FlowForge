"""Path selection strategies: Deterministic, Autonomous, Hybrid."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG, DAGNode
    from flowforge.doc.models import AnyDoc
    from flowforge.planner.base import ExecutionPlan


class DeterministicSelector:
    """Select all nodes in topological order — fully predictable."""

    def select(self, dag: FlowForgeDAG, docs: dict[str, AnyDoc]) -> list[str]:
        ordered = dag.topological_order()
        return [n.id for n in ordered]


class AutonomousSelector:
    """Let the LLM freely compose the execution path."""

    def select_from_llm_response(self, node_ids: list[str]) -> list[str]:
        return node_ids


class HybridSelector:
    """Use predefined paths but allow LLM to fill in branch choices."""

    def select(
        self,
        dag: FlowForgeDAG,
        docs: dict[str, AnyDoc],
        llm_overrides: dict[str, str] | None = None,
    ) -> list[str]:
        """Return default topological order; LLM overrides apply to branches."""
        ordered = dag.topological_order()
        node_ids = [n.id for n in ordered]
        return node_ids
