"""DAG node/edge models and the FlowForgeDAG wrapper around networkx.

FlowForge compiles the annotated agent class hierarchy into a
Directed Acyclic Graph (DAG).  Each decorated class or function becomes a
``DAGNode``; relationships between them become ``DAGEdge`` instances.

Node types
----------
``GLOBAL``  — one per agent; root of the DAG.
``FLOW``    — execution unit (may contain child flows and tasks).
              A flow with ``meta.is_branch == True`` is a *branch dispatcher*
              that selects one child flow at runtime.
``TASK``    — execution unit owned by a flow (may contain steps or child tasks).
              A task with ``meta.is_branch == True`` is a *branch dispatcher*
              that selects one child task at runtime.
``STEP``    — atomic execution unit owned by a leaf task.
              A step with ``meta.is_branch == True`` acts as a *branch
              dispatcher*: it evaluates a condition and calls one of several
              handler functions instead of its own body.

Note on branch dispatching
~~~~~~~~~~~~~~~~~~~~~~~~~~
There is no separate ``BRANCH`` node type.  Branch behaviour is a *property*
of any FLOW, TASK, or STEP node — indicated by ``node.meta.is_branch``.
This design keeps the DAG uniform and the execution engine simple: every node
runner checks ``meta.is_branch`` and handles it internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

import networkx as nx

from flowforge.errors import CycleDetectedError


class NodeType(str, Enum):
    """Enumeration of all DAG node categories.

    Values are used as display strings in renderers (Mermaid, Graphviz) and
    in trace records — keep them lowercase and stable.
    """

    GLOBAL = "global"
    FLOW   = "flow"
    TASK   = "task"
    STEP   = "step"


@dataclass
class DAGNode:
    """A single node in the FlowForge DAG.

    Attributes
    ----------
    id:
        Fully-qualified dotted path, e.g.
        ``"global.research.search.execute_search.optimize_query[1]"``.
        Constructed by the compiler as ``parent_id + "." + name + "[order]"``
        for steps, or ``parent_id + "." + name`` for flows/tasks.
    type:
        Structural category of this node (GLOBAL, FLOW, TASK, or STEP).
    name:
        Short human-readable label — used in visualisations and error messages.
    meta:
        The raw annotation metadata (``GlobalMeta`` / ``FlowMeta`` /
        ``TaskMeta`` / ``StepMeta``).  Cast to the appropriate type when
        needed.
    parent_id:
        The ``id`` of the parent node.  ``None`` only for the root GLOBAL node.
    doc:
        AI-generated documentation for this node.  ``None`` until
        ``agent.generate_docs()`` is called.
    """

    id: str
    type: NodeType
    name: str
    meta: Any        # GlobalMeta | FlowMeta | TaskMeta | StepMeta
    parent_id: str | None = None
    doc: Any = None  # Populated by doc/generator.py after compilation


@dataclass
class DAGEdge:
    """A directed edge in the FlowForge DAG.

    Attributes
    ----------
    source_id:
        ``id`` of the origin node.
    target_id:
        ``id`` of the destination node.
    edge_type:
        Semantic type of the relationship:

        ``"parent_child"``
            Standard containment: the target is a direct child of the source
            (flow → child flow, flow → task, task → step, etc.).
        ``"depends_on"``
            Execution dependency: the source flow must complete before the
            target flow starts.  Added by ``@flow(depends_on=[...])`` and
            processed by ``_add_depends_on_edges`` in the compiler.
    """

    source_id: str
    target_id: str
    edge_type: str = "parent_child"  # "parent_child" | "depends_on"


class FlowForgeDAG:
    """Thin wrapper over a networkx ``DiGraph`` with typed FlowForge nodes.

    Responsibilities
    ----------------
    * Store ``DAGNode`` and ``DAGEdge`` objects with O(1) lookup by node ID.
    * Delegate graph algorithms (topological sort, cycle detection) to
      networkx.
    * Provide type-filtered node queries used by the compiler, runner, and
      planner.

    The underlying networkx graph is accessible via the ``networkx_graph``
    property for advanced algorithms not exposed here.
    """

    def __init__(self) -> None:
        # networkx DiGraph is the authority for edge relationships and
        # graph-algorithm operations.
        self._graph: nx.DiGraph = nx.DiGraph()

        # Secondary index for O(1) node lookup by ID.
        self._nodes: dict[str, DAGNode] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: DAGNode) -> None:
        """Add a node to both the index and the underlying networkx graph."""
        self._nodes[node.id] = node
        self._graph.add_node(node.id, node=node)

    def add_edge(self, edge: DAGEdge) -> None:
        """Add a directed edge to the underlying networkx graph."""
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> DAGNode | None:
        """Return the node with the given ID, or ``None`` if not found."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[DAGNode]:
        """Return all nodes in insertion order."""
        return list(self._nodes.values())

    def get_children(self, node_id: str) -> list[DAGNode]:
        """Return all direct successor nodes of the given node."""
        return [
            self._nodes[nid]
            for nid in self._graph.successors(node_id)
            if nid in self._nodes
        ]

    def get_parents(self, node_id: str) -> list[DAGNode]:
        """Return all direct predecessor nodes of the given node."""
        return [
            self._nodes[nid]
            for nid in self._graph.predecessors(node_id)
            if nid in self._nodes
        ]

    def topological_order(self) -> list[DAGNode]:
        """Return all nodes in a valid topological (dependency-respecting) order.

        The root GLOBAL node is always first.

        Raises
        ------
        CycleDetectedError
            When the DAG contains a cycle (which would make topological sort
            impossible).
        """
        try:
            order = list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            cycle = nx.find_cycle(self._graph)
            raise CycleDetectedError([e[0] for e in cycle])
        return [self._nodes[nid] for nid in order if nid in self._nodes]

    def detect_cycles(self) -> list[list[str]]:
        """Return all cycles in the graph, each as a list of node IDs.

        Returns an empty list when the graph is acyclic.
        """
        cycles = list(nx.simple_cycles(self._graph))
        return cycles  # type: ignore[return-value]

    def nodes_by_type(self, node_type: NodeType) -> list[DAGNode]:
        """Return all nodes with the given ``NodeType``."""
        return [n for n in self._nodes.values() if n.type == node_type]

    def get_descendants(self, node_id: str) -> list[DAGNode]:
        """Return all descendant nodes (children, grandchildren, …) of *node_id*."""
        result: list[DAGNode] = []
        visited: set[str] = set()
        queue = list(self._graph.successors(node_id))
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            if nid in self._nodes:
                result.append(self._nodes[nid])
            queue.extend(self._graph.successors(nid))
        return result

    def resolve_route(self, route: str) -> set[str]:
        """Resolve a user-friendly route string to a set of node IDs.

        Accepts dotted paths at any level of specificity::

            "web_analysis"                       → flow + all descendants
            "web_analysis.scrape_and_analyze"    → task + all descendants
            "web_analysis.scrape_and_analyze.validate_url"  → just that step

        The resolution walks the DAG starting from ``"global"`` and matches
        each segment against child node names.  The matched node and **all
        its descendants** are included, plus every ancestor up to ``"global"``
        (so the runner can reach the target).

        Parameters
        ----------
        route:
            Dot-separated path of node names (without the ``"global."`` prefix).

        Returns
        -------
        set[str]
            Full DAG node IDs to include in ``planned_node_ids``.

        Raises
        ------
        ValueError
            When a segment cannot be resolved.
        """
        segments = route.strip().split(".")
        current_id = "global"
        matched_ids: set[str] = {"global"}

        for seg in segments:
            children = self.get_children(current_id)
            match = None
            for child in children:
                if child.name == seg:
                    match = child
                    break
            if match is None:
                available = [c.name for c in children]
                raise ValueError(
                    f"Route segment '{seg}' not found under '{current_id}'. "
                    f"Available: {available}"
                )
            current_id = match.id
            matched_ids.add(current_id)

        # Include all descendants of the resolved node.
        for desc in self.get_descendants(current_id):
            matched_ids.add(desc.id)

        return matched_ids

    def branch_nodes(self) -> list[DAGNode]:
        """Return all nodes that act as branch dispatchers.

        A branch dispatcher is any FLOW, TASK, or STEP node whose metadata
        has ``is_branch == True``.  These nodes select a child target at
        runtime instead of executing their own body.
        """
        return [
            n for n in self._nodes.values()
            if getattr(n.meta, "is_branch", False)
        ]

    # ------------------------------------------------------------------
    # Python protocols
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[DAGNode]:
        """Iterate over all nodes in insertion order."""
        return iter(self._nodes.values())

    def __len__(self) -> int:
        """Return the total number of nodes."""
        return len(self._nodes)

    @property
    def networkx_graph(self) -> nx.DiGraph:
        """Direct access to the underlying networkx ``DiGraph``.

        Use this for graph algorithms not exposed by ``FlowForgeDAG`` (e.g.
        shortest-path queries, subgraph extraction, …).
        """
        return self._graph
