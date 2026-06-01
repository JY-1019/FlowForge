"""Tests for the Harness Ledger (@ledger).

Covers:
- Trajectory Store: a @ledger flow records its run + node trajectory to SQLite.
- Contract Check: a must-not violation blocks execution and is recorded.
- Failure Detector: the same failure ≥ threshold surfaces a stored suggestion.
- CLI: ``ledger show`` prints recorded runs.
"""
import pytest
from typer.testing import CliRunner

from flowforge import global_config, flow, task, step, FlowForge, ledger
from flowforge.ledger import ContractViolation
from flowforge.ledger.cli import app as ledger_app
from flowforge.ledger.store import LedgerStore
from flowforge.ledger.detector import FailureDetector
from flowforge.ledger import core as ledger_core
from flowforge.ledger.contract import Violation


# ---------------------------------------------------------------------------
# Trajectory Store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ledger_records_trajectory(tmp_path):
    """A bare @ledger flow persists its run and node trajectory to SQLite."""
    db = str(tmp_path / "ledger.db")

    @ledger(db_path=db)
    @flow(name="traj_flow", prompt="trajectory flow")
    class TrajFlow:
        @task(name="traj_task", prompt="a task")
        class TrajTask:
            @step(order=1, prompt="do work")
            async def work(ctx):
                return {"ok": True}

    _traj_flow = TrajFlow

    @global_config(prompt="traj agent")
    class TrajAgent:
        f = _traj_flow

    engine = FlowForge.compile(TrajAgent)
    await engine.run("hello")

    store = LedgerStore(db)
    runs = store.recent_runs()
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    assert runs[0]["status"] == "ok"

    nodes = store.nodes_for_run(run_id)
    names = {n["name"] for n in nodes}
    assert "work" in names
    store.close()


# ---------------------------------------------------------------------------
# Contract Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_violation_blocks_and_records(tmp_path, monkeypatch):
    """A must-not contract violation blocks the step and is recorded."""
    db = str(tmp_path / "ledger.db")

    async def fake_check(*, node_id, node_contract, contracts, context_text, llm_config):
        return [Violation(contract="삭제 금지", reason="승인 없이 삭제 시도")]

    monkeypatch.setattr(ledger_core, "check_contracts", fake_check)

    ran = {"work": False}

    @ledger(contracts={"승인 없이 삭제": "차단"}, db_path=db)
    @flow(name="guard_flow", prompt="guarded flow")
    class GuardFlow:
        @task(name="guard_task", prompt="a task")
        class GuardTask:
            @step(order=1, prompt="delete things")
            async def work(ctx):
                ran["work"] = True
                return "done"

    _guard_flow = GuardFlow

    @global_config(prompt="guard agent")
    class GuardAgent:
        f = _guard_flow

    engine = FlowForge.compile(GuardAgent)
    with pytest.raises(ContractViolation):
        await engine.run("input")

    assert ran["work"] is False  # step never executed

    store = LedgerStore(db)
    run_id = store.recent_runs()[0]["run_id"]
    violations = store.violations_for_run(run_id)
    assert len(violations) == 1
    assert violations[0]["contract"] == "삭제 금지"
    store.close()


# ---------------------------------------------------------------------------
# Failure Detector
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failure_detector_surfaces_suggestion(tmp_path, capsys):
    """The same failure recurring ≥ threshold stores a suggestion + warns."""
    db = str(tmp_path / "ledger.db")
    store = LedgerStore(db)
    detector = FailureDetector(store, llm_config=None, threshold=3)

    exc = ValueError("boom 42")
    for _ in range(3):
        await detector.observe(
            run_id="r1", node_id="n1", step_name="flaky_step", exc=exc, func=None
        )

    pending = store.pending_suggestions()
    assert len(pending) == 1
    assert pending[0]["step_name"] == "flaky_step"

    out = capsys.readouterr().out
    assert "실패 패턴 감지: flaky_step" in out
    assert "ledger apply" in out
    store.close()


@pytest.mark.asyncio
async def test_failure_detector_below_threshold_is_silent(tmp_path, capsys):
    """Below threshold, no suggestion and no warning."""
    db = str(tmp_path / "ledger.db")
    store = LedgerStore(db)
    detector = FailureDetector(store, llm_config=None, threshold=3)

    exc = ValueError("nope")
    for _ in range(2):
        await detector.observe(
            run_id="r1", node_id="n1", step_name="step", exc=exc, func=None
        )

    assert store.pending_suggestions() == []
    assert "실패 패턴 감지" not in capsys.readouterr().out
    store.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_show_lists_runs(tmp_path):
    """``ledger show`` prints recorded runs."""
    db = str(tmp_path / "ledger.db")
    store = LedgerStore(db)
    store.start_run("abc123", "input")
    store.record_node(
        run_id="abc123", node_id="global.f.t.work[1]", node_type="step",
        name="work", execution_order=1, started_at=1.0, finished_at=2.0,
        duration_ms=1000.0, input_repr="in", output_repr="out", status="ok",
        error=None,
    )
    store.finish_run("abc123", "output", error=None)
    store.close()

    result = CliRunner().invoke(ledger_app, ["show", "--db", db])
    assert result.exit_code == 0
    assert "abc123" in result.stdout
    assert "work" in result.stdout
