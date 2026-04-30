"""Tests for route execution and task loop features.

Covers:
- Route: resolve_route on DAG, engine.run(route=...) filtering
- Loop: @task(max_loops=N, loop_condition=...) retry behavior
"""
import asyncio

import pytest
from pydantic import BaseModel

from flowforge import global_config, flow, task, step, FlowForge, OrderConflictError
from flowforge.types import LLMConfig


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

class Num(BaseModel):
    value: int


# ═══════════════════════════════════════════════════════════════════════
# Route tests
# ═══════════════════════════════════════════════════════════════════════

# Agent with multiple flows for route testing — defined at module level.

@global_config(prompt="route test agent")
class RouteAgent:
    @flow(name="alpha", prompt="alpha flow")
    class AlphaFlow:
        @task(name="a1", prompt="alpha task 1")
        class A1:
            @step(order=1, prompt="a1 step")
            async def s1(ctx):
                return {"result": "alpha_a1"}

        @task(name="a2", prompt="alpha task 2")
        class A2:
            @step(order=1, prompt="a2 step")
            async def s1(ctx):
                return {"result": "alpha_a2"}

    @flow(name="beta", prompt="beta flow")
    class BetaFlow:
        @task(name="b1", prompt="beta task")
        class B1:
            @step(order=1, prompt="b1 step")
            async def s1(ctx):
                return {"result": "beta_b1"}

    @flow(name="gamma", prompt="gamma flow")
    class GammaFlow:
        @task(name="g1", prompt="gamma task")
        class G1:
            @step(order=1, prompt="g1 step")
            async def s1(ctx):
                return {"result": "gamma_g1"}


class TestResolveRoute:
    def setup_method(self):
        self.engine = FlowForge.compile(RouteAgent)
        self.dag = self.engine.dag

    def test_resolve_flow(self):
        """Resolving a flow name includes the flow and all descendants."""
        ids = self.dag.resolve_route("alpha")
        assert "global" in ids
        assert "global.alpha" in ids
        assert "global.alpha.a1" in ids
        assert "global.alpha.a2" in ids
        # beta/gamma should NOT be included
        assert "global.beta" not in ids
        assert "global.gamma" not in ids

    def test_resolve_task(self):
        """Resolving a flow.task path includes only that task branch."""
        ids = self.dag.resolve_route("alpha.a1")
        assert "global.alpha" in ids  # ancestor included
        assert "global.alpha.a1" in ids
        # a2 should NOT be included
        assert "global.alpha.a2" not in ids

    def test_resolve_invalid_segment(self):
        """Invalid route segment raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            self.dag.resolve_route("nonexistent")

    def test_resolve_invalid_nested(self):
        """Invalid nested segment raises ValueError with available names."""
        with pytest.raises(ValueError, match="not found"):
            self.dag.resolve_route("alpha.nonexistent")


class TestRunWithRoute:
    @pytest.mark.asyncio
    async def test_route_single_flow(self):
        """route='beta' executes only the beta flow."""
        engine = FlowForge.compile(RouteAgent)
        result = await engine.run({"value": 1}, route="beta")
        assert result["result"] == "beta_b1"

    @pytest.mark.asyncio
    async def test_route_specific_task(self):
        """route='alpha.a1' executes only alpha flow → a1 task."""
        engine = FlowForge.compile(RouteAgent)
        result = await engine.run({"value": 1}, route="alpha.a1")
        assert result["result"] == "alpha_a1"

    @pytest.mark.asyncio
    async def test_route_multiple(self):
        """route=['alpha', 'gamma'] executes both but skips beta."""
        engine = FlowForge.compile(RouteAgent)
        # Last flow's output wins (gamma comes after alpha).
        result = await engine.run({"value": 1}, route=["alpha", "gamma"])
        assert result["result"] == "gamma_g1"

    @pytest.mark.asyncio
    async def test_route_multiple_preserves_input_order(self):
        """route list order controls root-flow execution order."""
        engine = FlowForge.compile(RouteAgent)
        result = await engine.run({"value": 1}, route=["gamma", "alpha"])
        assert result["result"] == "alpha_a2"

    @pytest.mark.asyncio
    async def test_route_with_trace(self):
        """Route filtering is visible in the run trace."""
        engine = FlowForge.compile(RouteAgent)
        result, trace = await engine.run_traced({"value": 1}, route="beta")
        executed_ids = trace.executed_node_ids
        assert "global.beta" in executed_ids
        assert "global.alpha" not in executed_ids
        assert "global.gamma" not in executed_ids


# ═══════════════════════════════════════════════════════════════════════
# Root-flow order-group parallelism
# ═══════════════════════════════════════════════════════════════════════

root_parallel_log: list[str] = []


@flow(name="root_a", prompt="root flow a", order=1)
class RootAFlow:
    @task(name="a", prompt="a")
    class ATask:
        @step(order=1, prompt="slow a")
        async def slow(ctx):
            root_parallel_log.append("a:start")
            await asyncio.sleep(0.05)
            root_parallel_log.append("a:end")
            return "a"


@flow(name="root_b", prompt="root flow b", order=1)
class RootBFlow:
    @task(name="b", prompt="b")
    class BTask:
        @step(order=1, prompt="fast b")
        async def fast(ctx):
            root_parallel_log.append("b:start")
            root_parallel_log.append("b:end")
            return "b"


@flow(name="root_c", prompt="root flow c", order=2)
class RootCFlow:
    @task(name="c", prompt="c")
    class CTask:
        @step(order=1, prompt="consume result from root order group")
        async def consume(ctx):
            root_parallel_log.append(f"c:{ctx.input}")
            return ctx.input


@global_config(prompt="root parallel agent")
class RootParallelAgent:
    RootAFlow = RootAFlow
    RootBFlow = RootBFlow
    RootCFlow = RootCFlow


@pytest.mark.asyncio
async def test_root_flows_with_same_order_run_in_parallel():
    """Root flows should follow the same order-group semantics as nested flows."""
    root_parallel_log.clear()
    engine = FlowForge.compile(RootParallelAgent)

    result = await engine.run("input")

    assert result == "b"
    assert root_parallel_log.index("b:start") < root_parallel_log.index("a:end")
    assert root_parallel_log[-1] == "c:b"


def test_root_flow_duplicate_unique_conflict_raises():
    """Only one root flow in a same-order group may declare unique=True."""
    with pytest.raises(OrderConflictError):
        @global_config(prompt="bad root unique agent")
        class BadRootUniqueAgent:
            @flow(name="first", prompt="first", order=1, unique=True)
            class First:
                pass

            @flow(name="second", prompt="second", order=1, unique=True)
            class Second:
                pass


# ═══════════════════════════════════════════════════════════════════════
# Loop tests
# ═══════════════════════════════════════════════════════════════════════

# Counter to track how many times the step runs.
_loop_call_count = 0


def _reset_counter():
    global _loop_call_count
    _loop_call_count = 0


# Agent with a looping task — result is "bad" first 2 times, "good" on 3rd.

@global_config(prompt="loop test agent")
class LoopAgent:
    @flow(name="loopy", prompt="loop flow")
    class LoopyFlow:
        @task(
            name="retry_task",
            prompt="task that retries until output is valid",
            max_loops=5,
            loop_condition=lambda output: output.get("valid", False),
        )
        class RetryTask:
            @step(order=1, prompt="produce result")
            async def produce(ctx):
                global _loop_call_count
                _loop_call_count += 1
                # Succeed on 3rd attempt
                if _loop_call_count >= 3:
                    return {"valid": True, "value": "success", "attempts": _loop_call_count}
                return {"valid": False, "value": "not_yet", "attempts": _loop_call_count}


class TestTaskLoop:
    @pytest.mark.asyncio
    async def test_loop_retries_until_condition_met(self):
        """Task re-runs step chain until loop_condition returns True."""
        _reset_counter()
        engine = FlowForge.compile(LoopAgent)
        result = await engine.run(None)
        assert result["valid"] is True
        assert result["value"] == "success"
        assert result["attempts"] == 3  # took 3 attempts

    @pytest.mark.asyncio
    async def test_loop_respects_max_loops(self):
        """When max_loops is exhausted, last result is returned."""
        # Agent where condition never passes.
        @global_config(prompt="never pass agent")
        class NeverPassAgent:
            @flow(name="f", prompt="f")
            class F:
                @task(
                    name="t",
                    prompt="always fails condition",
                    max_loops=3,
                    loop_condition=lambda output: False,  # never accepts
                )
                class T:
                    @step(order=1, prompt="do work")
                    async def s(ctx):
                        return {"attempt": True}

        engine = FlowForge.compile(NeverPassAgent)
        result = await engine.run(None)
        # Should return the last attempt's result even though condition failed.
        assert result["attempt"] is True

    @pytest.mark.asyncio
    async def test_no_loop_without_condition(self):
        """max_loops without loop_condition does nothing — runs once."""
        call_count = 0

        @global_config(prompt="no condition agent")
        class NoCondAgent:
            @flow(name="f", prompt="f")
            class F:
                @task(name="t", prompt="t", max_loops=5)  # no loop_condition
                class T:
                    @step(order=1, prompt="s")
                    async def s(ctx):
                        nonlocal call_count
                        call_count += 1
                        return {"done": True}

        engine = FlowForge.compile(NoCondAgent)
        result = await engine.run(None)
        assert call_count == 1  # only ran once

    @pytest.mark.asyncio
    async def test_loop_with_trace(self):
        """Loop iterations are visible in the trace."""
        _reset_counter()
        engine = FlowForge.compile(LoopAgent)
        result, trace = await engine.run_traced(None)
        assert result["valid"] is True
        assert trace.succeeded

    @pytest.mark.asyncio
    async def test_loop_first_try_success(self):
        """When condition is met on first try, no retry happens."""
        call_count = 0

        @global_config(prompt="first try agent")
        class FirstTryAgent:
            @flow(name="f", prompt="f")
            class F:
                @task(
                    name="t",
                    prompt="t",
                    max_loops=5,
                    loop_condition=lambda output: output.get("ok", False),
                )
                class T:
                    @step(order=1, prompt="s")
                    async def s(ctx):
                        nonlocal call_count
                        call_count += 1
                        return {"ok": True}

        engine = FlowForge.compile(FirstTryAgent)
        result = await engine.run(None)
        assert call_count == 1
        assert result["ok"] is True


# ═══════════════════════════════════════════════════════════════════════
# Route + Loop combined
# ═══════════════════════════════════════════════════════════════════════

_combined_count = 0


@global_config(prompt="combined agent")
class CombinedAgent:
    @flow(name="main", prompt="main flow")
    class MainFlow:
        @task(
            name="looper",
            prompt="loops until good",
            max_loops=3,
            loop_condition=lambda output: output.get("good", False),
        )
        class Looper:
            @step(order=1, prompt="work")
            async def work(ctx):
                global _combined_count
                _combined_count += 1
                return {"good": _combined_count >= 2, "count": _combined_count}

    @flow(name="other", prompt="other flow")
    class OtherFlow:
        @task(name="noop", prompt="noop")
        class Noop:
            @step(order=1, prompt="noop")
            async def noop(ctx):
                return {"noop": True}


class TestRouteAndLoop:
    @pytest.mark.asyncio
    async def test_route_with_loop(self):
        """Route to a specific flow that contains a looping task."""
        global _combined_count
        _combined_count = 0

        engine = FlowForge.compile(CombinedAgent)
        result = await engine.run(None, route="main")
        assert result["good"] is True
        assert result["count"] == 2  # looped once, succeeded on 2nd


# ═══════════════════════════════════════════════════════════════════════
# Shared data tests
# ═══════════════════════════════════════════════════════════════════════

@global_config(prompt="shared data agent")
class SharedDataAgent:
    @flow(name="producer", prompt="produces shared data")
    class ProducerFlow:
        @task(name="produce", prompt="write to shared_data")
        class ProduceTask:
            @step(order=1, prompt="store value in shared_data")
            async def store(ctx):
                ctx.shared_data["message"] = "hello from producer"
                ctx.shared_data["count"] = 42
                return {"produced": True}

    @flow(name="consumer", prompt="consumes shared data")
    class ConsumerFlow:
        @task(name="consume", prompt="read from shared_data")
        class ConsumeTask:
            @step(order=1, prompt="read value from shared_data")
            async def read(ctx):
                return {
                    "message": ctx.shared_data.get("message", ""),
                    "count": ctx.shared_data.get("count", 0),
                }


class TestSharedData:
    @pytest.mark.asyncio
    async def test_shared_data_across_flows(self):
        """shared_data written in one flow is readable in a subsequent flow."""
        engine = FlowForge.compile(SharedDataAgent)
        result = await engine.run(None)
        assert result["message"] == "hello from producer"
        assert result["count"] == 42

    @pytest.mark.asyncio
    async def test_shared_data_with_route(self):
        """shared_data is available but empty when only consumer runs."""
        engine = FlowForge.compile(SharedDataAgent)
        result = await engine.run(None, route="consumer")
        assert result["message"] == ""
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_shared_data_isolated_between_runs(self):
        """Each engine.run() call starts with a fresh shared_data dict."""
        engine = FlowForge.compile(SharedDataAgent)
        result1 = await engine.run(None)
        assert result1["message"] == "hello from producer"

        # Second run: shared_data should be fresh, not carry over.
        result2 = await engine.run(None, route="consumer")
        assert result2["message"] == ""

    @pytest.mark.asyncio
    async def test_shared_data_via_task_context(self):
        """shared_data is accessible via task_ctx.shared_data shortcut."""
        from flowforge.execution.context import GlobalContext, FlowContext, TaskContext
        from flowforge.tools.registry import ToolRegistry

        gctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        gctx.shared_data["key"] = "value"

        fctx = FlowContext(
            global_ctx=gctx,
            flow_name="f",
            flow_prompt="f",
        )
        tctx = TaskContext(
            flow_ctx=fctx,
            task_name="t",
            task_prompt="t",
        )
        assert tctx.shared_data["key"] == "value"
        tctx.shared_data["new_key"] = "new_value"
        assert gctx.shared_data["new_key"] == "new_value"
