"""Subtree visualization — renders which nodes executed during a single run.

Three output modes
------------------
- mermaid   : Zero-dependency text output (Mermaid classDef styling).
              Always available.  Paste into https://mermaid.live.

- graphviz  : Full DAG rendered to SVG/PNG.
              Executed nodes are colored + show execution order + duration.
              Skipped nodes are grayed out with a dashed border.
              Edges between executed nodes are bold.
              Requires ``pip install flowforge[viz]`` + system ``dot`` binary.

- terminal  : Rich table printed to stdout.  Always available.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.viz.run_trace import RunTrace, NodeTrace


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Execution-path colors keyed by NodeType value string.
# Branching steps/tasks/flows use their normal color; which branch was taken
# is shown via the "→ <selected_branch>" label appended by _node_label().
_EXEC_COLORS: dict[str, tuple[str, str]] = {
    "global": ("#1C2833", "white"),
    "flow":   ("#154360", "white"),
    "task":   ("#145A32", "white"),
    "step":   ("#512E5F", "white"),
}
_EXEC_ERROR_FILL = "#C0392B"
_EXEC_ERROR_FONT = "white"
_SKIPPED_FILL    = "#EAECEE"
_SKIPPED_FONT    = "#95A5A6"


def _node_label(node_type: str, name: str, nt: NodeTrace | None) -> str:
    """Build the node label for graphviz."""
    if nt is None:
        return name
    parts = [f"#{nt.execution_order}  {name}"]
    if nt.duration_ms is not None:
        parts.append(f"{nt.duration_ms:.0f} ms")
    if nt.selected_branch:
        parts.append(f"→ {nt.selected_branch}")
    if nt.error:
        parts.append(f"✗ {nt.error[:40]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Graphviz renderer
# ---------------------------------------------------------------------------

def render_run_graphviz(
    dag: FlowForgeDAG,
    trace: RunTrace,
    output_path: str | Path,
    fmt: str = "svg",
) -> Path:
    """Render the full DAG, highlighting the executed subtree.

    Args:
        dag:         Compiled FlowForgeDAG.
        trace:       RunTrace returned by engine.run_traced().
        output_path: Destination file (extension added automatically).
        fmt:         "svg" | "png" | "pdf".

    Returns:
        Path to the rendered file.
    """
    from flowforge.viz.renderer import _GRAPHVIZ_INSTALL_HINT
    try:
        import graphviz as gv
    except ImportError as e:
        raise ImportError(_GRAPHVIZ_INSTALL_HINT) from e

    executed  = trace.executed_node_ids
    all_visited = trace.all_visited_node_ids
    by_id     = {n.node_id: n for n in trace.nodes}

    dur_str = f"  {trace.duration_ms:.0f} ms" if trace.duration_ms else ""
    status  = "✓" if trace.succeeded else "✗ FAILED"
    title   = f"Run {trace.run_id}  {status}{dur_str}"

    dot = gv.Digraph(
        name="FlowForge Run",
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica Neue",
            "fontsize": "13",
            "label": title,
            "labelloc": "t",
            "bgcolor": "#FDFEFE",
            "pad": "0.4",
        },
        node_attr={"fontname": "Helvetica Neue", "fontsize": "10"},
    )

    for node in dag.get_all_nodes():
        node_id = node.id
        nt = by_id.get(node_id)
        is_exec = node_id in executed
        is_error = node_id in all_visited and not is_exec

        if is_error and nt:
            fill, font = _EXEC_ERROR_FILL, _EXEC_ERROR_FONT
            label = _node_label(node.type.value, node.name, nt)
            style = "rounded,filled"
            penwidth = "2.5"
        elif is_exec and nt:
            fill, font = _EXEC_COLORS.get(node.type.value, ("#2C3E50", "white"))
            label = _node_label(node.type.value, node.name, nt)
            style = "rounded,filled"
            penwidth = "2.0"
        else:
            fill, font = _SKIPPED_FILL, _SKIPPED_FONT
            label = node.name
            style = "rounded,filled,dashed"
            penwidth = "0.5"

        dot.node(
            node_id,
            label=label,
            shape="box",
            style=style,
            fillcolor=fill,
            fontcolor=font,
            penwidth=penwidth,
            tooltip=f"in: {nt.input_repr}\nout: {nt.output_repr}" if nt else "",
        )

    for node in dag.get_all_nodes():
        for child in dag.get_children(node.id):
            both = node.id in executed and child.id in executed
            dot.edge(
                node.id,
                child.id,
                penwidth="2.2" if both else "0.6",
                color="#2C3E50" if both else "#BDC3C7",
            )

    output_path = Path(output_path)
    try:
        dot.render(
            filename=str(output_path.with_suffix("")),
            format=fmt,
            cleanup=True,
        )
    except Exception as e:
        # Python package present but system 'dot' binary is missing.
        raise ImportError(_GRAPHVIZ_INSTALL_HINT) from e
    return output_path


# ---------------------------------------------------------------------------
# Mermaid renderer
# ---------------------------------------------------------------------------

def render_run_mermaid(dag: FlowForgeDAG, trace: RunTrace) -> str:
    """Return a Mermaid diagram string with execution highlighting."""
    executed    = trace.executed_node_ids
    all_visited = trace.all_visited_node_ids
    by_id       = {n.node_id: n for n in trace.nodes}

    def safe(s: str) -> str:
        return s.replace(".", "_").replace("[", "_").replace("]", "_")

    lines = [
        "graph TD",
        "  %%  FlowForge run subtree — executed nodes are colored, skipped nodes are gray",
        "  %%  Branching steps show '→ <selected_branch>' in their label.",
        "  classDef exec_global  fill:#1C2833,stroke:#111,color:white,stroke-width:2px",
        "  classDef exec_flow    fill:#154360,stroke:#0E2F44,color:white,stroke-width:2px",
        "  classDef exec_task    fill:#145A32,stroke:#0B3D21,color:white,stroke-width:2px",
        "  classDef exec_step    fill:#512E5F,stroke:#3B1F44,color:white,stroke-width:2px",
        "  classDef exec_error   fill:#C0392B,stroke:#922B21,color:white,stroke-width:2px",
        "  classDef skipped      fill:#EAECEE,stroke:#AEB6BF,color:#95A5A6,"
        "stroke-width:1px,stroke-dasharray:4",
        "",
    ]

    # Nodes
    for node in dag.get_all_nodes():
        node_id = node.id
        sid     = safe(node_id)
        nt      = by_id.get(node_id)
        is_exec = node_id in executed
        is_err  = node_id in all_visited and not is_exec

        if (is_exec or is_err) and nt:
            dur     = f" {nt.duration_ms:.0f}ms" if nt.duration_ms is not None else ""
            branch  = f" → {nt.selected_branch}" if nt.selected_branch else ""
            err_tag = f" ✗" if nt.error else ""
            label   = f"#{nt.execution_order} {node.name}{dur}{branch}{err_tag}"
        else:
            label = node.name

        # Escape double-quotes inside the label
        label = label.replace('"', "'")
        lines.append(f'  {sid}["{label}"]')

    lines.append("")

    # Edges
    for node in dag.get_all_nodes():
        sid = safe(node.id)
        for child in dag.get_children(node.id):
            csid = safe(child.id)
            if node.id in executed and child.id in executed:
                lines.append(f"  {sid} ==> {csid}")   # bold arrow for executed path
            else:
                lines.append(f"  {sid} --> {csid}")

    lines.append("")

    # Class assignments
    for node in dag.get_all_nodes():
        node_id = node.id
        sid     = safe(node_id)
        nt      = by_id.get(node_id)
        is_err  = node_id in all_visited and node_id not in executed

        if is_err:
            cls = "exec_error"
        elif node_id in executed:
            cls = f"exec_{node.type.value}"
        else:
            cls = "skipped"

        lines.append(f"  class {sid} {cls}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Terminal summary (no extra deps)
# ---------------------------------------------------------------------------

def print_run_summary(trace: RunTrace) -> None:
    """Print a human-readable execution summary to stdout."""
    W = 72
    sep = "─" * W

    dur   = f"{trace.duration_ms:.1f} ms" if trace.duration_ms else "—"
    status = "✓  OK" if trace.succeeded else f"✗  FAILED — {trace.error}"

    print(f"\n{sep}")
    print(f"  Run {trace.run_id}   {status}")
    print(f"  Duration : {dur}   Nodes executed : {len(trace.nodes)}")
    print(sep)
    print(f"  {'#':>3}  {'type':<8}  {'name':<28}  {'dur':>8}  branch / error")
    print(f"  {'─'*3}  {'─'*8}  {'─'*28}  {'─'*8}  {'─'*18}")

    for nt in trace.nodes:
        dur_s    = f"{nt.duration_ms:.0f} ms" if nt.duration_ms is not None else "—"
        extra    = nt.selected_branch or (f"✗ {nt.error[:30]}" if nt.error else "")
        icon     = "✗" if nt.error else "✓"
        print(
            f"  {nt.execution_order:>3}  {nt.node_type:<8}  {nt.name:<28}  "
            f"{dur_s:>8}  {extra}  {icon}"
        )

    print(sep)
    print(f"  Input  : {trace.input_repr[:60]}")
    print(f"  Output : {trace.output_repr[:60]}")
    print(sep + "\n")
