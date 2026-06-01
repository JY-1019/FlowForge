"""``ledger`` command-line interface.

    ledger show            recent trajectory summary
    ledger show --fail     only failed trajectories
    ledger apply           apply pending repair suggestions (with confirm)
    ledger rollback        undo the most recent apply
"""
from __future__ import annotations

import difflib
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from flowforge.ledger.store import LedgerStore

app = typer.Typer(
    name="ledger",
    help="Harness Ledger — inspect trajectories and apply repair suggestions.",
    no_args_is_help=True,
)
console = Console()

_DB_OPTION = typer.Option("./ledger.db", "--db", help="Path to the ledger SQLite file.")


def _open(db: str) -> LedgerStore:
    if not Path(db).exists():
        console.print(f"[red]No ledger database at {db}.[/red] Run a @ledger flow first.")
        raise typer.Exit(1)
    return LedgerStore(db)


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


@app.command()
def show(
    fail: bool = typer.Option(False, "--fail", help="Show only failed trajectories."),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show."),
    db: str = _DB_OPTION,
) -> None:
    """Print a summary of recent runs and their node trajectories."""
    store = _open(db)
    runs = store.recent_runs(limit=limit, failed_only=fail)
    if not runs:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        raise typer.Exit(0)

    for run in runs:
        status = run["status"] or "?"
        color = {"ok": "green", "failed": "red", "running": "yellow"}.get(status, "white")
        console.print(
            f"\n[bold]run [cyan]{run['run_id']}[/cyan][/bold]  "
            f"[{color}]{status}[/{color}]  {_fmt_ts(run['started_at'])}"
        )
        if run["error"]:
            console.print(f"  [red]error:[/red] {run['error']}")

        nodes = store.nodes_for_run(run["run_id"], failed_only=fail)
        if nodes:
            table = Table(show_header=True, header_style="bold")
            table.add_column("#", justify="right")
            table.add_column("node")
            table.add_column("type")
            table.add_column("status")
            table.add_column("ms", justify="right")
            for n in nodes:
                ns = n["status"] or "?"
                nc = {"ok": "green", "error": "red"}.get(ns, "white")
                table.add_row(
                    str(n["execution_order"]),
                    n["node_id"],
                    n["node_type"],
                    f"[{nc}]{ns}[/{nc}]",
                    f"{n['duration_ms']:.0f}" if n["duration_ms"] is not None else "-",
                )
            console.print(table)

        decisions = store.decisions_for_run(run["run_id"])
        for d in decisions:
            console.print(
                f"  [magenta]decision[/magenta] {d['node_id']} → "
                f"{d['selected']} ({d['kind']})"
            )
        violations = store.violations_for_run(run["run_id"])
        for v in violations:
            console.print(
                f"  [red]contract blocked[/red] {v['node_id']}: "
                f"{v['contract']} — {v['reason']}"
            )
    store.close()


@app.command()
def apply(
    db: str = _DB_OPTION,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Apply pending repair suggestions to the source after confirmation."""
    store = _open(db)
    pending = store.pending_suggestions()
    if not pending:
        console.print("[yellow]No pending suggestions.[/yellow]")
        store.close()
        raise typer.Exit(0)

    applied = 0
    for s in pending:
        console.print(
            f"\n[bold]suggestion #{s['id']}[/bold] for "
            f"[cyan]{s['step_name']}[/cyan]"
        )
        console.print(f"  cause:      {s['cause']}")
        console.print(f"  suggestion: {s['suggestion']}")

        if not s["new_source"] or not s["source_file"] or not s["old_source"]:
            console.print(
                "  [yellow]advisory only — no automated patch available.[/yellow]"
            )
            continue

        path = Path(s["source_file"])
        if not path.exists():
            console.print(f"  [red]source file missing: {path}[/red]")
            continue
        current = path.read_text()
        if s["old_source"] not in current:
            console.print(
                "  [red]source has changed since detection; skipping patch.[/red]"
            )
            continue

        diff = difflib.unified_diff(
            s["old_source"].splitlines(),
            s["new_source"].splitlines(),
            fromfile=f"{path.name} (current)",
            tofile=f"{path.name} (proposed)",
            lineterm="",
        )
        console.print("\n".join(diff) or "  (no textual diff)")

        if not yes and not typer.confirm(f"Apply this patch to {path}?"):
            console.print("  [yellow]skipped.[/yellow]")
            continue

        store.record_apply(s["id"], str(path), current)  # back up whole file
        path.write_text(current.replace(s["old_source"], s["new_source"], 1))
        store.mark_suggestion(s["id"], "applied")
        console.print(f"  [green]applied to {path}.[/green]")
        applied += 1

    console.print(f"\n[bold]{applied} suggestion(s) applied.[/bold]")
    store.close()


@app.command()
def rollback(db: str = _DB_OPTION) -> None:
    """Undo the most recent apply, restoring the backed-up source file."""
    store = _open(db)
    last = store.last_applied()
    if last is None:
        console.print("[yellow]Nothing to roll back.[/yellow]")
        store.close()
        raise typer.Exit(0)

    path = Path(last["target_file"])
    console.print(f"Rolling back last apply to [cyan]{path}[/cyan] …")
    path.write_text(last["backup_source"])
    store.mark_apply(last["id"], "rolled_back")
    store.mark_suggestion(last["suggestion_id"], "pending")
    console.print(f"[green]restored {path}.[/green]")
    store.close()


if __name__ == "__main__":
    app()
