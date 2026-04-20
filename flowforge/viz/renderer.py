"""DAG rendering to SVG/PNG/Mermaid.

Graphviz is **optional**.  When the ``graphviz`` Python package or the
system ``dot`` binary is not installed, ``render_graphviz`` raises a
helpful ``ImportError`` that tells the user exactly how to opt in.

Zero-dependency alternative
---------------------------
``render_mermaid()`` always works — no extra packages required.
Paste the output into https://mermaid.live to see the diagram.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG

_GRAPHVIZ_INSTALL_HINT = (
    "SVG/PNG rendering requires the graphviz Python package AND the "
    "system 'dot' binary.\n\n"
    "  pip install flowforge[viz]          # Python wrapper\n"
    "  brew install graphviz               # macOS\n"
    "  sudo apt install graphviz           # Debian / Ubuntu\n"
    "  choco install graphviz              # Windows\n\n"
    "Alternatively, use engine.mermaid() — it works with no extra dependencies."
)


def render_graphviz(
    dag: FlowForgeDAG,
    output_path: str | Path,
    show_docs: bool = False,
    fmt: str = "svg",
) -> Path:
    """Render the DAG to an SVG/PNG file using graphviz.

    Requires ``pip install flowforge[viz]`` **and** the system ``dot``
    binary.  Raises a descriptive ``ImportError`` when either is missing.

    For a zero-dependency alternative use :func:`render_mermaid` instead.
    """
    try:
        import graphviz as gv
    except ImportError as e:
        raise ImportError(_GRAPHVIZ_INSTALL_HINT) from e

    color_map = {
        "global": "#4A90D9",
        "flow":   "#27AE60",
        "task":   "#F39C12",
        "step":   "#8E44AD",
    }

    dot = gv.Digraph(
        name="FlowForge",
        graph_attr={"rankdir": "TB", "fontname": "Helvetica"},
        node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica"},
    )

    for node in dag.get_all_nodes():
        color = color_map.get(node.type.value, "#95A5A6")
        label = node.name
        if show_docs and node.doc and hasattr(node.doc, "summary"):
            label += f"\n{node.doc.summary[:50]}..."
        dot.node(
            node.id,
            label=label,
            fillcolor=color,
            fontcolor="white",
        )

    for node in dag.get_all_nodes():
        for child in dag.get_children(node.id):
            dot.edge(node.id, child.id)

    output_path = Path(output_path)
    try:
        dot.render(
            filename=str(output_path.with_suffix("")),
            format=fmt,
            cleanup=True,
        )
    except Exception as e:
        # The Python package is installed but the system 'dot' binary is missing.
        raise ImportError(_GRAPHVIZ_INSTALL_HINT) from e

    return output_path


def render_mermaid(dag: FlowForgeDAG) -> str:
    """Render the DAG as a Mermaid diagram string.

    No extra packages required — works out of the box.
    Paste the output into https://mermaid.live to view the diagram.
    """
    lines = ["graph TD"]

    for node in dag.get_all_nodes():
        safe_id = node.id.replace(".", "_").replace("[", "_").replace("]", "_")
        lines.append(f'    {safe_id}["{node.type.value}: {node.name}"]')

    for node in dag.get_all_nodes():
        safe_id = node.id.replace(".", "_").replace("[", "_").replace("]", "_")
        for child in dag.get_children(node.id):
            safe_child = child.id.replace(".", "_").replace("[", "_").replace("]", "_")
            lines.append(f"    {safe_id} --> {safe_child}")

    return "\n".join(lines)
