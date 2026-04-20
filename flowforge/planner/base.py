"""Abstract planner and ExecutionPlan model."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.doc.models import AnyDoc
    from flowforge.types import LLMConfig


@dataclass
class ExecutionPlan:
    """The plan produced by a planner — an ordered list of node IDs to execute."""

    node_ids: list[str] = field(default_factory=list)
    mode: str = "deterministic"
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractPlanner(abc.ABC):
    """Base class for all planners."""

    @abc.abstractmethod
    async def plan(
        self,
        user_request: Any,
        dag: FlowForgeDAG,
        docs: dict[str, AnyDoc],
        llm_config: LLMConfig,
    ) -> ExecutionPlan:
        """Produce an ExecutionPlan from user request + DAG + docs."""
