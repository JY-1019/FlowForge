"""Tests for run-trace collection and subtree visualization.

Covers:
- RunTracer unit operations (start/finish/error)
- Full engine.run() / engine.run_traced() trace collection
- Branch selection recorded in NodeTrace.selected_branch
- Mermaid subtree rendering
- Terminal summary printing
"""
import pytest
from pydantic import BaseModel

from flowforge import global_config, flow, task, step, FlowForge, BranchCondition
from flowforge.viz.run_trace import RunTrace, NodeTrace, RunTracer, _safe_repr


# ---------------------------------------------------------------------------
# Shared agent definitions
# ---------------------------------------------------------------------------

call_log: list[str] = []


class RouteInput(BaseModel):
    route: str


async def handler_a(ctx):
    call_log.append("handler_a")
    return "result_a"


async def handler_b(ctx):
    call_log.append("handler_b")
    return "result_b"


@global_config(prompt="branch direct agent")
class BranchDirectAgent:
    """Agent whose only step is a branching @step — RouteInput flows straight in."""

    @flow(name="branch_flow", prompt="branch")
    class BranchFlow:

        @task(name="branch_task", prompt="branch")
        class BranchTask:
            # The old @branch decorator is replaced by @step with condition/branches.
            @step(
                order=1,
                prompt="route",
                condition=BranchCondition(field="route", enum=["a", "b"]),
                branches={"a": handler_a, "b": handler_b},
                fallback=handler_b,
            )
            async def router(ctx): ...


@global_config(prompt="trace test agent")
class TraceAgent:

    @flow(name="main_flow", prompt="main")
    class MainFlow:

        @task(name="step_task", prompt="steps")
        class StepTask:
            @step(order=1, prompt="s1")
            async def step_one(ctx):
                call_log.append("step_one")
                return "after_one"

            @step(order=2, prompt="s2")
            async def step_two(ctx):
                call_log.append("step_two")
                return "after_two"

        @task(name="branch_task", prompt="branch")
        class BranchTask:
            # Branching @step — receives the output of StepTask (a string),
            # so the condition value resolves to None → fallback is used.
            @step(
                order=1,
                prompt="route",
                condition=BranchCondition(field="route", enum=["a", "b"]),
                branches={"a": handler_a, "b": handler_b},
                fallback=handler_b,
            )
            async def router(ctx): ...


# ---------------------------------------------------------------------------
# _safe_repr
# ---------------------------------------------------------------------------

def test_safe_repr_none():
    assert _safe_repr(None) == "None"


def test_safe_repr_model():
    r = RouteInput(route="x")
    rep = _safe_repr(r)
    assert "route" in rep


def test_safe_repr_truncates():
    long = "x" * 300
    rep  = _safe_repr(long)
    assert len(rep) <= 165   # 160 chars + "…"
    assert rep.endswith("…")


# ---------------------------------------------------------------------------
# RunTracer unit tests
# ---------------------------------------------------------------------------

def test_run_tracer_records_start_finish():
    tracer = RunTracer(run_input="hello")
    tracer.start_node("global.flow", "flow", "flow", "hello")
    tracer.finish_node("global.flow", "world")
    trace = tracer.finish_run("world")

    assert len(trace.nodes) == 1
    nt = trace.nodes[0]
    assert nt.node_id == "global.flow"
    assert nt.execution_order == 1
    assert nt.succeeded
    assert "world" in nt.output_repr


def test_run_tracer_records_error():
    tracer = RunTracer()
    tracer.start_node("n1", "step", "my_step", None)
    tracer.error_node("n1", "something went wrong")
    trace = tracer.finish_run(error="something went wrong")

    nt = trace.nodes[0]
    assert not nt.succeeded
    assert "something went wrong" in nt.error
    assert "n1" not in trace.executed_node_ids
    assert "n1" in trace.all_visited_node_ids


def test_execution_order_increments():
    tracer = RunTracer()
    for name in ["a", "b", "c"]:
        tracer.start_node(name, "step", name, None)
        tracer.finish_node(name, None)
    trace = tracer.finish_run()

    orders = [n.execution_order for n in trace.nodes]
    assert orders == [1, 2, 3]


def test_get_node_trace():
    tracer = RunTracer()
    tracer.start_node("x", "step", "x", None)
    tracer.finish_node("x", "out")
    trace = tracer.finish_run()

    nt = trace.get_node_trace("x")
    assert nt is not None
    assert nt.node_id == "x"


# ---------------------------------------------------------------------------
# Integration: trace collected during engine.run()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_run_stores_trace():
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    await engine.run("input")
    trace = engine.last_trace

    assert trace is not None
    assert trace.succeeded
    assert trace.duration_ms is not None and trace.duration_ms >= 0


@pytest.mark.asyncio
async def test_run_traced_returns_trace():
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    result, trace = await engine.run_traced("input")
    assert trace is not None
    assert isinstance(trace, RunTrace)


@pytest.mark.asyncio
async def test_trace_contains_all_executed_nodes():
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    _, trace = await engine.run_traced(RouteInput(route="a"))

    exec_ids = trace.executed_node_ids
    # Flow and task nodes must appear.
    assert any("main_flow"   in nid for nid in exec_ids)
    assert any("step_task"   in nid for nid in exec_ids)
    assert any("branch_task" in nid for nid in exec_ids)
    # Steps inside step_task must appear.
    assert any("step_one" in nid for nid in exec_ids)
    assert any("step_two" in nid for nid in exec_ids)


@pytest.mark.asyncio
async def test_trace_node_ids_match_dag():
    """Every executed node_id must exist in the compiled DAG."""
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    _, trace = await engine.run_traced(RouteInput(route="b"))

    dag_ids = {n.id for n in engine.dag.get_all_nodes()}
    for node_id in trace.executed_node_ids:
        assert node_id in dag_ids, f"Trace node '{node_id}' not found in DAG"


@pytest.mark.asyncio
async def test_trace_branching_step_records_selected():
    """A branching @step stores selected_branch in its NodeTrace."""
    call_log.clear()
    engine = FlowForge.compile(BranchDirectAgent)

    _, trace = await engine.run_traced(RouteInput(route="a"))

    # Branching steps are recorded as node_type="step" with selected_branch set.
    branching_traces = [
        n for n in trace.nodes
        if n.node_type == "step" and n.selected_branch is not None
    ]
    assert len(branching_traces) == 1
    assert branching_traces[0].selected_branch == "a"


@pytest.mark.asyncio
async def test_trace_branching_step_fallback_records_fallback():
    """When no branch matches, selected_branch is set to '__fallback__'."""
    call_log.clear()
    engine = FlowForge.compile(BranchDirectAgent)

    _, trace = await engine.run_traced(RouteInput(route="unknown"))

    branching_traces = [
        n for n in trace.nodes
        if n.node_type == "step" and n.selected_branch is not None
    ]
    assert branching_traces[0].selected_branch == "__fallback__"


@pytest.mark.asyncio
async def test_execution_order_is_sequential():
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    _, trace = await engine.run_traced("input")

    orders = [n.execution_order for n in trace.nodes]
    assert orders == sorted(orders)
    assert orders[0] == 1


# ---------------------------------------------------------------------------
# Mermaid subtree rendering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_mermaid_contains_executed_nodes():
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    await engine.run(RouteInput(route="a"))
    mmd = engine.run_mermaid()

    assert "graph TD" in mmd
    assert "exec_flow" in mmd    # at least one executed flow node
    assert "skipped" in mmd      # at least one skipped node
    assert "classDef" in mmd


@pytest.mark.asyncio
async def test_run_mermaid_bold_edges_for_executed_path():
    """Executed A→B edges use ==> (bold); others use -->."""
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    await engine.run(RouteInput(route="b"))
    mmd = engine.run_mermaid()

    assert "==>" in mmd   # at least one bold executed edge


@pytest.mark.asyncio
async def test_visualize_run_raises_without_run():
    engine = FlowForge.compile(TraceAgent)
    # No run has been performed yet.
    with pytest.raises(RuntimeError, match="No run trace"):
        engine.visualize_run()


@pytest.mark.asyncio
async def test_print_run_summary_runs_without_error(capsys):
    call_log.clear()
    engine = FlowForge.compile(TraceAgent)

    await engine.run("anything")
    engine.print_run_summary()

    captured = capsys.readouterr()
    assert "Run" in captured.out
    assert "step" in captured.out or "flow" in captured.out
