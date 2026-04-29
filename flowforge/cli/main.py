"""FlowForge CLI — flowforge viz | validate | run | doc-generate."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="flowforge",
    help="FlowForge — Annotation-Based AI Agent Framework",
    no_args_is_help=True,
)
console = Console()


def _load_agent_module(agent_file: Path):
    """Dynamically import the agent module and return the global_config class."""
    spec = importlib.util.spec_from_file_location("_agent_module", agent_file)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Cannot load module from {agent_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_agent_module"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _find_agent_class(module):
    """Find the class decorated with @global_config."""
    from flowforge.annotations.decorators import _GLOBAL_ATTR

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, _GLOBAL_ATTR):
            return obj
    return None


def _compile(agent_file: Path, dynamic_options=None):
    """Load module, find agent class, compile DAG."""
    from flowforge import FlowForge

    module = _load_agent_module(agent_file)
    cls = _find_agent_class(module)
    if cls is None:
        console.print("[red]No @global_config class found in the file.[/red]")
        raise typer.Exit(1)
    return FlowForge.compile(cls, dynamic_options=dynamic_options)


@app.command()
def validate(
    agent_file: Path = typer.Argument(..., help="Path to the agent Python file"),
) -> None:
    """Validate the DAG structure (cycle detection, order uniqueness, I/O compatibility)."""
    console.print(f"[bold]Validating[/bold] {agent_file} ...")
    try:
        engine = _compile(agent_file)
        console.print(f"[green]✓ Valid DAG with {len(engine.dag)} nodes.[/green]")

        # Check for cycles
        cycles = engine.dag.detect_cycles()
        if cycles:
            console.print(f"[red]✗ Cycles detected: {cycles}[/red]")
            raise typer.Exit(1)

        # Print summary table
        table = Table(title="DAG Nodes")
        table.add_column("Node ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Name", style="green")
        for node in engine.dag.get_all_nodes():
            table.add_row(node.id, node.type.value, node.name)
        console.print(table)
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def viz(
    agent_file: Path = typer.Argument(..., help="Path to the agent Python file"),
    output: Path = typer.Option(Path("dag.svg"), "--output", "-o", help="Output file path"),
    show_docs: bool = typer.Option(False, "--show-docs", help="Include doc summaries in nodes"),
    fmt: str = typer.Option("svg", "--format", "-f", help="Output format: svg | png | pdf"),
    mermaid: bool = typer.Option(False, "--mermaid", help="Print Mermaid diagram to stdout"),
    save_md: Optional[Path] = typer.Option(None, "--save-md", help="Save full DAG Mermaid to a .md file"),
) -> None:
    """Render the DAG to an SVG/PNG/PDF file.

    \b
    Examples
    --------
    flowforge viz agent.py --mermaid               # print Mermaid to stdout
    flowforge viz agent.py --save-md dag.md        # save Mermaid to file
    flowforge viz agent.py --output dag.svg        # render SVG (needs graphviz)
    flowforge viz agent.py --output dag.svg --show-docs
    """
    engine = _compile(agent_file)

    if mermaid:
        from flowforge.viz.renderer import render_mermaid
        print(render_mermaid(engine.dag))
        return

    if save_md:
        from flowforge.viz.renderer import render_mermaid
        md_content = "# FlowForge — Full DAG Structure\n\n```mermaid\n" + render_mermaid(engine.dag) + "\n```\n"
        save_md.write_text(md_content)
        console.print(f"[green]✓ Mermaid DAG saved to {save_md}[/green]")
        return

    try:
        from flowforge.viz.renderer import render_graphviz
        result = render_graphviz(engine.dag, output, show_docs=show_docs, fmt=fmt)
        console.print(f"[green]✓ DAG rendered to {result}[/green]")
    except ImportError:
        console.print("[yellow]graphviz not available — printing Mermaid instead:[/yellow]")
        from flowforge.viz.renderer import render_mermaid
        print(render_mermaid(engine.dag))


@app.command()
def run(
    agent_file: Path = typer.Argument(..., help="Path to the agent Python file"),
    query: str = typer.Option(..., "--query", "-q", help="User query string"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Print execution trace table"),
    viz_run: bool = typer.Option(False, "--viz", help="Render subtree SVG after run"),
    viz_output: Path = typer.Option(Path("run.svg"), "--viz-output", help="Subtree SVG output path"),
    viz_fmt: str = typer.Option("svg", "--viz-fmt", help="svg | png | pdf"),
    viz_mermaid: bool = typer.Option(False, "--viz-mermaid", help="Print executed-path Mermaid to stdout"),
    compare: bool = typer.Option(False, "--compare", help="Print full DAG + executed path side by side"),
    compare_output: Optional[Path] = typer.Option(None, "--compare-output", help="Save comparison to a .md file"),
    planning_mode: str = typer.Option("deterministic", "--planning-mode", help="deterministic | autonomous | hybrid"),
    allow_dynamic: bool = typer.Option(False, "--allow-dynamic", help="Allow dynamic flow/tool generation for this run"),
    dynamic_dir: Path = typer.Option(Path("flowforge/generated"), "--dynamic-dir", help="Directory for generated flow/tool files"),
) -> None:
    """Run the agent with a user query.

    \b
    Examples
    --------
    flowforge run agent.py -q "hello" --trace
    flowforge run agent.py -q "hello" --viz-mermaid          # executed path only
    flowforge run agent.py -q "hello" --compare              # full DAG + executed path
    flowforge run agent.py -q "hello" --compare-output viz.md
    flowforge run agent.py -q "hello" --viz --viz-output run.svg
    """
    from flowforge import DynamicRunOptions

    dynamic_options = DynamicRunOptions(
        enabled=allow_dynamic,
        project_root=str(agent_file.resolve().parent),
        generated_dir=str(dynamic_dir),
    )
    engine = _compile(agent_file, dynamic_options=dynamic_options)

    async def _run():
        result, run_trace = await engine.run_traced(
            query,
            planning_mode=planning_mode,
            dynamic_options=dynamic_options,
        )

        console.print("[bold green]Result:[/bold green]")
        console.print(result)

        if trace:
            from flowforge.viz.subtree import print_run_summary
            print_run_summary(run_trace)

        if viz_mermaid:
            from flowforge.viz.subtree import render_run_mermaid
            mmd = render_run_mermaid(engine.dag, run_trace)
            console.print("\n[bold]Executed path (Mermaid):[/bold]")
            print(mmd)

        if compare or compare_output:
            md = engine.compare_mermaid(run_trace)
            if compare_output:
                compare_output.write_text(md)
                console.print(f"[green]✓ Comparison saved to {compare_output}[/green]")
            else:
                console.print("\n[bold]Full DAG vs Executed Path:[/bold]")
                print(md)

        if viz_run:
            try:
                from flowforge.viz.subtree import render_run_graphviz
                out = render_run_graphviz(engine.dag, run_trace, viz_output, fmt=viz_fmt)
                console.print(f"[green]✓ Subtree rendered to {out}[/green]")
            except ImportError:
                console.print("[yellow]graphviz unavailable — falling back to Mermaid:[/yellow]")
                from flowforge.viz.subtree import render_run_mermaid
                print(render_run_mermaid(engine.dag, run_trace))

    asyncio.run(_run())


@app.command(name="doc-generate")
def doc_generate(
    agent_file: Path = typer.Argument(..., help="Path to the agent Python file"),
    force: bool = typer.Option(False, "--force", help="Force regeneration (ignore cache)"),
) -> None:
    """Generate AI documentation for all nodes."""
    engine = _compile(agent_file)

    async def _gen():
        docs = await engine.generate_docs(force=force)
        console.print(f"[green]✓ Generated docs for {len(docs)} nodes.[/green]")
        for node_id, doc in docs.items():
            summary = getattr(doc, "summary", "")
            console.print(f"  [cyan]{node_id}[/cyan]: {summary[:80]}")

    asyncio.run(_gen())


def _bundled_mkdocs_yml() -> Path:
    """Return the path to the mkdocs.yml shipped inside the flowforge package."""
    import flowforge as _ff_pkg

    pkg_dir = Path(_ff_pkg.__file__).resolve().parent
    return pkg_dir / "mkdocs.yml"


def _require_mkdocs() -> None:
    """Fail with a helpful message if the `docs` extra is not installed."""
    try:
        import mkdocs  # noqa: F401
    except ImportError:
        console.print(
            "[red]mkdocs is not installed.[/red]\n"
            "Install the docs extra with:\n"
            "  [cyan]pip install 'flowforge[docs]'[/cyan]  "
            "[dim](or re-install with the [docs] extra)[/dim]"
        )
        raise typer.Exit(1)


@app.command()
def docs(
    port:   int  = typer.Option(8000, "--port", "-p", help="Port for the local dev server"),
    host:   str  = typer.Option("127.0.0.1", "--host", "-H", help="Host/interface to bind"),
    online: bool = typer.Option(False, "--online", help="Open the published GitHub Pages site instead of serving locally"),
    build:  bool = typer.Option(False, "--build", help="Build a static site to --build-dir and exit"),
    build_dir: Path = typer.Option(Path("site"), "--build-dir", help="Output directory for --build"),
    no_open:   bool = typer.Option(False, "--no-open", help="Don't automatically open the browser"),
) -> None:
    """Serve the FlowForge documentation locally (or open the published site).

    \b
    Examples
    --------
    flowforge docs                     # start local mkdocs server + open browser
    flowforge docs --online            # just open https://jy-1019.github.io/FlowForge/
    flowforge docs --port 9000         # serve on a different port
    flowforge docs --build             # build static site into ./site/
    """
    import subprocess
    import threading
    import time
    import webbrowser

    if online:
        url = "https://jy-1019.github.io/FlowForge/"
        console.print(f"[bold]Opening published docs:[/bold] [cyan]{url}[/cyan]")
        webbrowser.open(url)
        return

    _require_mkdocs()

    mkdocs_yml = _bundled_mkdocs_yml()
    if not mkdocs_yml.is_file():
        console.print(
            f"[red]Bundled mkdocs.yml not found at {mkdocs_yml}.[/red]\n"
            "Reinstall flowforge (the docs are shipped as package data):\n"
            "  [cyan]pip install --force-reinstall 'flowforge[docs]'[/cyan]\n"
            "Or open the published site instead:\n"
            "  [cyan]flowforge docs --online[/cyan]"
        )
        raise typer.Exit(1)

    if build:
        out = build_dir.resolve()
        console.print(f"[bold]Building static docs site → [cyan]{out}[/cyan][/bold]")
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "build",
             "-f", str(mkdocs_yml),
             "-d", str(out)],
            check=True,
        )
        console.print(f"[green]✓ Built to {out}[/green]")
        return

    url = f"http://{host}:{port}"
    console.print(
        f"[bold]Serving FlowForge docs at [cyan]{url}[/cyan][/bold]  "
        "[dim](Ctrl+C to stop)[/dim]"
    )

    if not no_open:
        def _open_later() -> None:
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open_later, daemon=True).start()

    try:
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "serve",
             "-f", str(mkdocs_yml),
             "-a", f"{host}:{port}"],
            check=False,
        )
    except KeyboardInterrupt:
        pass


skills_app = typer.Typer(
    name="skills",
    help="Manage vendored Anthropic Agent Skills (anthropics/skills).",
    no_args_is_help=True,
)
app.add_typer(skills_app)


@skills_app.command("list")
def skills_list() -> None:
    """Show vendored skills and the curated default bundle."""
    from flowforge.skills import bundled_skill_names
    from flowforge.skills.sync import DEFAULT_BUNDLED_SKILLS, UPSTREAM_REPO

    vendored = set(bundled_skill_names())
    default = set(DEFAULT_BUNDLED_SKILLS)
    table = Table(title=f"Anthropic Agent Skills  ({UPSTREAM_REPO})")
    table.add_column("name", style="cyan")
    table.add_column("vendored", justify="center", style="green")
    table.add_column("in default bundle", justify="center", style="magenta")

    for name in sorted(vendored | default):
        table.add_row(
            name,
            "✓" if name in vendored else "",
            "✓" if name in default else "",
        )
    console.print(table)
    if not vendored:
        console.print(
            "[dim]No skills vendored yet.  Run "
            "`flowforge skills sync` to install the default bundle.[/dim]"
        )


@skills_app.command("sync")
def skills_sync(
    name: Optional[str] = typer.Argument(
        None,
        help=(
            "Upstream skill name (folder under anthropics/skills/skills/). "
            "Omit to vendor the curated default bundle."
        ),
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-download even if already vendored."
    ),
    insecure: bool = typer.Option(
        False,
        "--insecure",
        help="Disable TLS verification (e.g. behind the LG CNS proxy).",
    ),
) -> None:
    """Download upstream Anthropic skills into the local bundle."""
    import urllib.error

    from flowforge.skills.sync import (
        DEFAULT_BUNDLED_SKILLS,
        sync_default_skills,
        sync_skill,
    )

    verify_ssl = not insecure
    try:
        if name is None:
            console.print(
                "[bold]Syncing default bundle:[/bold] "
                f"{', '.join(DEFAULT_BUNDLED_SKILLS)}"
            )
            for path in sync_default_skills(force=force, verify_ssl=verify_ssl):
                console.print(f"  [green]✓[/green] {path}")
        else:
            path = sync_skill(name, force=force, verify_ssl=verify_ssl)
            console.print(f"[green]✓[/green] {path}")
    except (FileNotFoundError, urllib.error.URLError, OSError) as exc:
        console.print(f"[red]✗ sync failed: {exc}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
