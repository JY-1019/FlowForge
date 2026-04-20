"""Execution trace — records every node that ran during a single engine.run() call."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Repr helper
# ---------------------------------------------------------------------------

def _safe_repr(value: Any, max_len: int = 160) -> str:
    """Compact repr that never raises."""
    try:
        if value is None:
            return "None"
        if hasattr(value, "model_dump"):
            text = str(value.model_dump())
        else:
            text = repr(value)
        return text[:max_len] + ("…" if len(text) > max_len else "")
    except Exception:
        return "<unrepresentable>"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NodeTrace:
    """Trace record for a single node execution."""

    node_id: str                      # Full dotted DAG path
    node_type: str                    # "flow" | "task" | "step" | "branch"
    name: str
    execution_order: int              # 1-based, global across the run
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    input_repr: str = ""
    output_repr: str = ""
    error: str | None = None
    selected_branch: str | None = None  # branch nodes only

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1000

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.finished_at is not None


@dataclass
class RunTrace:
    """Complete trace of a single engine.run() call."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    nodes: list[NodeTrace] = field(default_factory=list)
    input_repr: str = ""
    output_repr: str = ""
    error: str | None = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def executed_node_ids(self) -> set[str]:
        """Node IDs that completed successfully."""
        return {n.node_id for n in self.nodes if n.succeeded}

    @property
    def all_visited_node_ids(self) -> set[str]:
        """All node IDs that were started (including failed ones)."""
        return {n.node_id for n in self.nodes}

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1000

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.finished_at is not None

    def get_node_trace(self, node_id: str) -> NodeTrace | None:
        """Return the NodeTrace for a given node_id (last attempt if retried)."""
        # Walk backwards so that the last attempt is returned on retry
        for nt in reversed(self.nodes):
            if nt.node_id == node_id:
                return nt
        return None

    def nodes_by_type(self, node_type: str) -> list[NodeTrace]:
        return [n for n in self.nodes if n.node_type == node_type]


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class RunTracer:
    """Collects trace events during a single agent run.

    Attach an instance to GlobalContext before calling engine.run().
    Runners call start_node() / finish_node() / error_node() transparently.
    """

    def __init__(self, run_input: Any = None) -> None:
        self._trace = RunTrace(input_repr=_safe_repr(run_input))
        self._counter = 0
        # node_id -> the most recent NodeTrace for that id (for finish/error calls)
        self._active: dict[str, NodeTrace] = {}

    @property
    def trace(self) -> RunTrace:
        return self._trace

    def start_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        input_data: Any = None,
    ) -> None:
        self._counter += 1
        nt = NodeTrace(
            node_id=node_id,
            node_type=node_type,
            name=name,
            execution_order=self._counter,
            input_repr=_safe_repr(input_data),
        )
        self._active[node_id] = nt
        self._trace.nodes.append(nt)

    def finish_node(
        self,
        node_id: str,
        output_data: Any = None,
        selected_branch: str | None = None,
    ) -> None:
        nt = self._active.get(node_id)
        if nt is not None:
            nt.finished_at = time.time()
            nt.output_repr = _safe_repr(output_data)
            if selected_branch:
                nt.selected_branch = selected_branch

    def error_node(self, node_id: str, error: str) -> None:
        nt = self._active.get(node_id)
        if nt is not None:
            nt.finished_at = time.time()
            nt.error = error

    def finish_run(
        self,
        output_data: Any = None,
        error: str | None = None,
    ) -> RunTrace:
        self._trace.finished_at = time.time()
        self._trace.output_repr = _safe_repr(output_data)
        self._trace.error = error
        return self._trace
