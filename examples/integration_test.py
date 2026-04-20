"""FlowForge Integration Test — Personal Testing Script

Run this file directly to manually verify every major framework feature:

    python examples/integration_test.py

Each section is numbered and self-contained.  A [PASS] / [FAIL] is printed
for every scenario.  No external API keys are needed for most tests.
Scenarios that call a real LLM are clearly marked and skipped when the
ANTHROPIC_API_KEY environment variable is not set.

Sections
--------
1.  Basic step execution — data flows through steps in order
2.  LLMConfig accessible in ctx — ctx.llm_config gives the right config
3.  LLMConfig factory methods — for_claude / for_openai / for_gemini
4.  Provider switching — same agent, different LLMConfig
5.  Parallel steps — same order runs both steps
6.  unique=True step — only the marked step runs
7.  Branch dispatching (step level) — condition routes to correct handler
8.  Branch dispatching fallback (step level)
9.  Branch dispatching (task level)
10. Branch dispatching (flow level)
11. Container task — nested @task structure
12. Nested flows — @flow inside @flow
13. Run trace — trace captured after engine.run()
14. run_traced() — explicit (result, trace) return
15. Mermaid diagram — engine.mermaid() output
16. Run mermaid — executed path highlighting
17. LLM call inside a step — uses ctx.llm_config + call_with_tool  [LIVE LLM]
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pydantic import BaseModel

# Make sure we can import flowforge from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowforge import (
    global_config, flow, task, step,
    FlowForge, BranchCondition, LLMConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _ok(name: str) -> None:
    global _passed
    _passed += 1
    print(f"  [PASS] {name}")


def _fail(name: str, reason: str) -> None:
    global _failed
    _failed += 1
    print(f"  [FAIL] {name}: {reason}")


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def _run(agent_cls, input_data):
    """Compile and run an agent class; return the result."""
    engine = FlowForge.compile(agent_cls)
    return await engine.run(input_data)


# ============================================================
# 1. Basic step execution
# ============================================================
_section("1. Basic step execution")

call_log_1: list[str] = []


@task(name="basic_task", prompt="basic")
class BasicTask:
    @step(order=1, prompt="step 1")
    async def s1(ctx):
        call_log_1.append("s1")
        return "after_s1"

    @step(order=2, prompt="step 2")
    async def s2(ctx):
        call_log_1.append(f"s2_got:{ctx.input}")
        return "final"


@flow(name="basic_flow", prompt="basic")
class BasicFlow:
    BasicTask = BasicTask


@global_config(prompt="basic agent")
class BasicAgent:
    BasicFlow = BasicFlow


async def test_basic():
    result = await _run(BasicAgent, "hello")
    if call_log_1 == ["s1", "s2_got:after_s1"]:
        _ok("steps execute in order, output threads through")
    else:
        _fail("steps execute in order", f"call_log={call_log_1}")

    if result == "final":
        _ok("final step output is returned as agent result")
    else:
        _fail("final step output", f"got {result!r}")


# ============================================================
# 2. LLMConfig accessible inside ctx
# ============================================================
_section("2. LLMConfig accessible in ctx")

captured_llm_config = {}


@task(name="ctx_task", prompt="ctx")
class CtxTask:
    @step(order=1, prompt="capture llm config")
    async def capture(ctx):
        captured_llm_config["provider"] = ctx.llm_config.provider
        captured_llm_config["model"]    = ctx.llm_config.model
        return ctx.input


@flow(name="ctx_flow", prompt="ctx")
class CtxFlow:
    CtxTask = CtxTask


@global_config(
    prompt="ctx agent",
    llm_config=LLMConfig(provider="openai", model="gpt-4o", temperature=0.1),
)
class CtxAgent:
    CtxFlow = CtxFlow


async def test_ctx_llm_config():
    await _run(CtxAgent, "ping")
    if captured_llm_config.get("provider") == "openai":
        _ok("ctx.llm_config.provider == 'openai'")
    else:
        _fail("ctx.llm_config.provider", f"got {captured_llm_config.get('provider')!r}")

    if captured_llm_config.get("model") == "gpt-4o":
        _ok("ctx.llm_config.model == 'gpt-4o'")
    else:
        _fail("ctx.llm_config.model", f"got {captured_llm_config.get('model')!r}")


# ============================================================
# 3. LLMConfig factory methods
# ============================================================
_section("3. LLMConfig factory methods")


def test_llm_config_factories():
    c = LLMConfig.for_claude()
    assert c.provider == "anthropic" and c.model == "claude-sonnet-4-6"
    _ok("for_claude() defaults")

    c = LLMConfig.for_claude("claude-opus-4-6", temperature=0.7)
    assert c.model == "claude-opus-4-6" and c.temperature == 0.7
    _ok("for_claude() with overrides")

    c = LLMConfig.for_openai()
    assert c.provider == "openai" and c.model == "gpt-4o"
    _ok("for_openai() defaults")

    c = LLMConfig.for_openai("gpt-4-turbo")
    assert c.model == "gpt-4-turbo"
    _ok("for_openai() with model override")

    c = LLMConfig.for_gemini()
    assert c.provider == "google" and c.model == "gemini-2.0-flash"
    _ok("for_gemini() defaults")

    c = LLMConfig.for_gemini("gemini-1.5-pro")
    assert c.model == "gemini-1.5-pro"
    _ok("for_gemini() with model override")

    c = LLMConfig(provider="google", model="gemini-2.0-flash", api_key="abc")
    assert c.api_key == "abc"
    _ok("direct LLMConfig(provider='google', ...)")


# ============================================================
# 4. Provider switching — same agent, different LLMConfig
# ============================================================
_section("4. Provider switching — same agent, different LLMConfig")

provider_seen_4: list[str] = []


@task(name="provider_task", prompt="provider")
class ProviderTask:
    @step(order=1, prompt="record provider")
    async def record(ctx):
        provider_seen_4.append(ctx.llm_config.provider)
        return ctx.input


@flow(name="provider_flow", prompt="provider")
class ProviderFlow:
    ProviderTask = ProviderTask


# Same flow structure, different global_config LLMConfig per agent.
@global_config(prompt="claude agent", llm_config=LLMConfig.for_claude())
class ClaudeAgent:
    ProviderFlow = ProviderFlow


@global_config(prompt="openai agent", llm_config=LLMConfig.for_openai())
class OpenAIAgent:
    ProviderFlow = ProviderFlow


@global_config(prompt="gemini agent", llm_config=LLMConfig.for_gemini())
class GeminiAgent:
    ProviderFlow = ProviderFlow


async def test_provider_switching():
    provider_seen_4.clear()

    await _run(ClaudeAgent, "x")
    await _run(OpenAIAgent, "x")
    await _run(GeminiAgent, "x")

    if provider_seen_4 == ["anthropic", "openai", "google"]:
        _ok("each agent uses the correct provider in ctx.llm_config")
    else:
        _fail("provider switching", f"got {provider_seen_4}")


# ============================================================
# 5. Parallel steps (same order)
# ============================================================
_section("5. Parallel steps — same order runs both")

parallel_log_5: list[str] = []


@task(name="parallel_task_5", prompt="parallel")
class ParallelTask5:
    @step(order=1, prompt="branch A")
    async def a(ctx):
        parallel_log_5.append("a")
        return "a_result"

    @step(order=1, prompt="branch B — same order")
    async def b(ctx):
        parallel_log_5.append("b")
        return "b_result"

    @step(order=2, prompt="after parallel group")
    async def c(ctx):
        parallel_log_5.append(f"c_got:{ctx.input}")
        return ctx.input


@flow(name="parallel_flow_5", prompt="parallel")
class ParallelFlow5:
    ParallelTask5 = ParallelTask5


@global_config(prompt="parallel agent 5")
class ParallelAgent5:
    ParallelFlow5 = ParallelFlow5


async def test_parallel_steps():
    parallel_log_5.clear()
    await _run(ParallelAgent5, "start")

    if "a" in parallel_log_5 and "b" in parallel_log_5:
        _ok("both order=1 steps execute")
    else:
        _fail("both order=1 steps execute", f"log={parallel_log_5}")

    if any(e.startswith("c_got:") for e in parallel_log_5):
        _ok("order=2 step runs after the parallel group")
    else:
        _fail("order=2 step runs after", f"log={parallel_log_5}")


# ============================================================
# 6. unique=True step
# ============================================================
_section("6. unique=True step — only the marked step runs")

unique_log_6: list[str] = []


@task(name="unique_task_6", prompt="unique")
class UniqueTask6:
    @step(order=1, prompt="winner", unique=True)
    async def winner(ctx):
        unique_log_6.append("winner")
        return "winner"

    @step(order=1, prompt="skipped")
    async def skipped(ctx):
        unique_log_6.append("skipped")
        return "skipped"


@flow(name="unique_flow_6", prompt="unique")
class UniqueFlow6:
    UniqueTask6 = UniqueTask6


@global_config(prompt="unique agent 6")
class UniqueAgent6:
    UniqueFlow6 = UniqueFlow6


async def test_unique_step():
    unique_log_6.clear()
    await _run(UniqueAgent6, "x")

    if unique_log_6 == ["winner"]:
        _ok("unique=True step runs, sibling skipped")
    else:
        _fail("unique=True step", f"log={unique_log_6}")


# ============================================================
# 7. Branch dispatching — step level, correct path
# ============================================================
_section("7. Branch dispatching — step level")


class RouteInput7(BaseModel):
    route: str


branch_log_7: list[str] = []


async def handler_a7(ctx):
    branch_log_7.append("a")
    return "result_a"


async def handler_b7(ctx):
    branch_log_7.append("b")
    return "result_b"


@task(name="branch_task_7", prompt="branch")
class BranchTask7:
    @step(
        order=1,
        prompt="route",
        condition=BranchCondition(field="route", enum=["a", "b"]),
        branches={"a": handler_a7, "b": handler_b7},
        fallback=handler_b7,
    )
    async def router(ctx): ...


@flow(name="branch_flow_7", prompt="branch")
class BranchFlow7:
    BranchTask7 = BranchTask7


@global_config(prompt="branch agent 7")
class BranchAgent7:
    BranchFlow7 = BranchFlow7


async def test_step_branch():
    branch_log_7.clear()
    await _run(BranchAgent7, RouteInput7(route="a"))
    if branch_log_7 == ["a"]:
        _ok("route='a' dispatches to handler_a")
    else:
        _fail("route='a'", f"log={branch_log_7}")

    branch_log_7.clear()
    await _run(BranchAgent7, RouteInput7(route="b"))
    if branch_log_7 == ["b"]:
        _ok("route='b' dispatches to handler_b")
    else:
        _fail("route='b'", f"log={branch_log_7}")


# ============================================================
# 8. Branch dispatching — fallback
# ============================================================
_section("8. Branch dispatching — fallback")


async def test_step_branch_fallback():
    branch_log_7.clear()
    await _run(BranchAgent7, RouteInput7(route="unknown"))
    if branch_log_7 == ["b"]:
        _ok("unknown route falls back to handler_b")
    else:
        _fail("fallback", f"log={branch_log_7}")


# ============================================================
# 9. Branch dispatching — task level
# ============================================================
_section("9. Branch dispatching — task level")


class BackendInput9(BaseModel):
    backend: str


task_branch_log_9: list[str] = []


@task(name="fast_task_9", prompt="fast")
class FastTask9:
    @step(order=1, prompt="fast")
    async def run(ctx):
        task_branch_log_9.append("fast")
        return "fast"


@task(name="slow_task_9", prompt="slow")
class SlowTask9:
    @step(order=1, prompt="slow")
    async def run(ctx):
        task_branch_log_9.append("slow")
        return "slow"


@task(
    name="dispatch_task_9",
    prompt="dispatch",
    condition=BranchCondition(field="backend", enum=["fast", "slow"]),
    branches={"fast": FastTask9, "slow": SlowTask9},
    fallback=FastTask9,
)
class DispatchTask9: ...


@flow(name="task_branch_flow_9", prompt="task branch")
class TaskBranchFlow9:
    DispatchTask9 = DispatchTask9


@global_config(prompt="task branch agent 9")
class TaskBranchAgent9:
    TaskBranchFlow9 = TaskBranchFlow9


async def test_task_branch():
    task_branch_log_9.clear()
    await _run(TaskBranchAgent9, BackendInput9(backend="slow"))
    if task_branch_log_9 == ["slow"]:
        _ok("task-level branch selects SlowTask9")
    else:
        _fail("task-level branch", f"log={task_branch_log_9}")

    task_branch_log_9.clear()
    await _run(TaskBranchAgent9, BackendInput9(backend="unknown"))
    if task_branch_log_9 == ["fast"]:
        _ok("task-level branch fallback selects FastTask9")
    else:
        _fail("task-level branch fallback", f"log={task_branch_log_9}")


# ============================================================
# 10. Branch dispatching — flow level
# ============================================================
_section("10. Branch dispatching — flow level")


class SpeedInput10(BaseModel):
    speed: str


flow_branch_log_10: list[str] = []


@task(name="quick_task_10", prompt="quick")
class QuickTask10:
    @step(order=1, prompt="quick")
    async def run(ctx):
        flow_branch_log_10.append("quick")
        return "quick"


@task(name="thorough_task_10", prompt="thorough")
class ThoroughTask10:
    @step(order=1, prompt="thorough")
    async def run(ctx):
        flow_branch_log_10.append("thorough")
        return "thorough"


@flow(name="quick_flow_10", prompt="quick flow")
class QuickFlow10:
    QuickTask10 = QuickTask10


@flow(name="thorough_flow_10", prompt="thorough flow")
class ThoroughFlow10:
    ThoroughTask10 = ThoroughTask10


@flow(
    name="dispatch_flow_10",
    prompt="dispatch flow",
    condition=BranchCondition(field="speed", enum=["quick", "thorough"]),
    branches={"quick": QuickFlow10, "thorough": ThoroughFlow10},
    fallback=QuickFlow10,
)
class DispatchFlow10: ...


@global_config(prompt="flow branch agent 10")
class FlowBranchAgent10:
    DispatchFlow10 = DispatchFlow10


async def test_flow_branch():
    flow_branch_log_10.clear()
    await _run(FlowBranchAgent10, SpeedInput10(speed="thorough"))
    if flow_branch_log_10 == ["thorough"]:
        _ok("flow-level branch selects ThoroughFlow10")
    else:
        _fail("flow-level branch", f"log={flow_branch_log_10}")


# ============================================================
# 11. Container task — nested @task structure
# ============================================================
_section("11. Container task — nested @task structure")

container_log_11: list[str] = []


@task(name="step_one_task", prompt="step 1")
class StepOneTask:
    @step(order=1, prompt="do step 1")
    async def run(ctx):
        container_log_11.append("step_one")
        return "after_step_one"


@task(name="step_two_task", prompt="step 2")
class StepTwoTask:
    @step(order=1, prompt="do step 2")
    async def run(ctx):
        container_log_11.append(f"step_two_got:{ctx.input}")
        return "done"


@task(name="container_task_11", prompt="container")
class ContainerTask11:
    StepOneTask = StepOneTask
    StepTwoTask = StepTwoTask


@flow(name="container_flow_11", prompt="container")
class ContainerFlow11:
    ContainerTask11 = ContainerTask11


@global_config(prompt="container agent 11")
class ContainerAgent11:
    ContainerFlow11 = ContainerFlow11


async def test_container_task():
    container_log_11.clear()
    result = await _run(ContainerAgent11, "start")

    if container_log_11 == ["step_one", "step_two_got:after_step_one"]:
        _ok("child tasks run sequentially, output chains through")
    else:
        _fail("container task execution", f"log={container_log_11}")

    if result == "done":
        _ok("final child task output is agent result")
    else:
        _fail("container task result", f"got {result!r}")


# ============================================================
# 12. Nested flows — @flow inside @flow
# ============================================================
_section("12. Nested flows")

nested_log_12: list[str] = []


@task(name="inner_task_12", prompt="inner")
class InnerTask12:
    @step(order=1, prompt="inner step")
    async def run(ctx):
        nested_log_12.append("inner")
        return "inner_done"


@task(name="outer_task_12", prompt="outer")
class OuterTask12:
    @step(order=1, prompt="outer step")
    async def run(ctx):
        nested_log_12.append(f"outer_got:{ctx.input}")
        return "outer_done"


@flow(name="inner_flow_12", prompt="inner flow")
class InnerFlow12:
    InnerTask12 = InnerTask12


@flow(name="outer_flow_12", prompt="outer flow")
class OuterFlow12:
    InnerFlow12 = InnerFlow12
    OuterTask12 = OuterTask12


@global_config(prompt="nested agent 12")
class NestedAgent12:
    OuterFlow12 = OuterFlow12


async def test_nested_flows():
    nested_log_12.clear()
    result = await _run(NestedAgent12, "start")

    if "inner" in nested_log_12:
        _ok("inner flow executes")
    else:
        _fail("inner flow executes", f"log={nested_log_12}")

    if any(e.startswith("outer_got:") for e in nested_log_12):
        _ok("outer task runs after inner flow, receives inner flow output")
    else:
        _fail("outer task runs after inner flow", f"log={nested_log_12}")


# ============================================================
# 13. Run trace — stored in engine.last_trace
# ============================================================
_section("13. Run trace — engine.last_trace")


async def test_run_trace():
    engine = FlowForge.compile(BasicAgent)
    call_log_1.clear()
    await engine.run("hello")

    trace = engine.last_trace
    if trace is not None:
        _ok("engine.last_trace is populated after run()")
    else:
        _fail("engine.last_trace", "None")
        return

    if trace.succeeded:
        _ok("trace.succeeded is True")
    else:
        _fail("trace.succeeded", "False")

    if trace.duration_ms is not None and trace.duration_ms >= 0:
        _ok(f"trace.duration_ms = {trace.duration_ms:.1f}ms")
    else:
        _fail("trace.duration_ms", str(trace.duration_ms))

    exec_ids = trace.executed_node_ids
    if any("basic_flow" in nid for nid in exec_ids):
        _ok("trace contains basic_flow node")
    else:
        _fail("trace contains basic_flow", f"ids={exec_ids}")

    if any("s1" in nid for nid in exec_ids):
        _ok("trace contains step s1")
    else:
        _fail("trace contains step s1", f"ids={exec_ids}")


# ============================================================
# 14. run_traced() — explicit (result, trace) return
# ============================================================
_section("14. run_traced() — explicit (result, trace) return")


async def test_run_traced():
    engine = FlowForge.compile(BasicAgent)
    call_log_1.clear()
    result, trace = await engine.run_traced("hello")

    if result == "final":
        _ok("run_traced() returns the correct result")
    else:
        _fail("run_traced() result", f"got {result!r}")

    from flowforge.viz.run_trace import RunTrace
    if isinstance(trace, RunTrace):
        _ok("run_traced() returns a RunTrace instance")
    else:
        _fail("run_traced() trace type", type(trace).__name__)

    orders = [n.execution_order for n in trace.nodes]
    if orders == sorted(orders):
        _ok(f"trace nodes are in execution order: {orders}")
    else:
        _fail("trace node order", f"orders={orders}")


# ============================================================
# 15. Mermaid diagram
# ============================================================
_section("15. Mermaid diagram — engine.mermaid()")


def test_mermaid():
    engine = FlowForge.compile(BasicAgent)
    mmd = engine.mermaid()

    if "graph TD" in mmd:
        _ok("mermaid() contains 'graph TD'")
    else:
        _fail("mermaid() format", "missing 'graph TD'")

    if "basic_flow" in mmd:
        _ok("mermaid() includes flow node 'basic_flow'")
    else:
        _fail("mermaid() flow node", "missing 'basic_flow'")

    print("\n  --- Mermaid output (copy to https://mermaid.live) ---")
    for line in mmd.strip().splitlines()[:20]:
        print(f"  {line}")
    if mmd.count("\n") > 20:
        print("  ... (truncated)")
    print()


# ============================================================
# 16. Run mermaid — executed path
# ============================================================
_section("16. Run mermaid — executed path highlighting")


async def test_run_mermaid():
    engine = FlowForge.compile(BasicAgent)
    call_log_1.clear()
    await engine.run("hello")
    mmd = engine.run_mermaid()

    if "==>" in mmd:
        _ok("run_mermaid() contains bold (==>) edges for executed path")
    else:
        _fail("run_mermaid() bold edges", "no ==> found")

    if "exec_flow" in mmd or "exec_task" in mmd or "exec_step" in mmd:
        _ok("run_mermaid() marks executed nodes")
    else:
        _fail("run_mermaid() executed nodes", "no exec_* class found")


# ============================================================
# 17. LLM call inside a step  [LIVE LLM — skipped if no API key]
# ============================================================
_section("17. LLM call inside a step  [LIVE — requires ANTHROPIC_API_KEY]")

llm_result_17 = {}


async def handler_live_llm(ctx):
    """Branch handler that makes a real Anthropic API call."""
    from flowforge.llm.caller import call_with_tool

    _GREET_TOOL = {
        "name": "make_greeting",
        "description": "Return a short greeting",
        "input_schema": {
            "type": "object",
            "properties": {
                "greeting": {"type": "string", "description": "A one-sentence greeting"},
            },
            "required": ["greeting"],
        },
    }
    data = await call_with_tool(
        prompt=f"Say hello to '{ctx.input}' in one short sentence.",
        tool_schema=_GREET_TOOL,
        llm_config=ctx.llm_config,   # uses whatever LLMConfig was set on the agent
        max_tokens=64,
    )
    llm_result_17["greeting"] = data.get("greeting", "")
    return llm_result_17["greeting"]


@task(name="live_llm_task", prompt="live llm test")
class LiveLLMTask:
    @step(order=1, prompt="greet via LLM")
    async def greet(ctx):
        return await handler_live_llm(ctx)


@flow(name="live_llm_flow", prompt="live llm flow")
class LiveLLMFlow:
    LiveLLMTask = LiveLLMTask


@global_config(
    prompt="live llm agent",
    llm_config=LLMConfig.for_claude(),
)
class LiveLLMAgent:
    LiveLLMFlow = LiveLLMFlow


async def test_live_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [SKIP] ANTHROPIC_API_KEY not set — skipping live LLM test")
        print("         Set ANTHROPIC_API_KEY=<your key> and re-run to test.")
        return

    try:
        result = await _run(LiveLLMAgent, "FlowForge")
        if isinstance(result, str) and len(result) > 5:
            _ok(f"LLM returned a greeting: {result!r}")
        else:
            _fail("LLM greeting", f"got {result!r}")
    except Exception as e:
        _fail("live LLM call", str(e))


# ============================================================
# Main runner
# ============================================================

async def main():
    print("\n" + "="*60)
    print("  FlowForge Integration Test")
    print("="*60)

    await test_basic()
    await test_ctx_llm_config()
    test_llm_config_factories()
    await test_provider_switching()
    await test_parallel_steps()
    await test_unique_step()
    await test_step_branch()
    await test_step_branch_fallback()
    await test_task_branch()
    await test_flow_branch()
    await test_container_task()
    await test_nested_flows()
    await test_run_trace()
    await test_run_traced()
    test_mermaid()
    await test_run_mermaid()
    await test_live_llm()

    print("\n" + "="*60)
    total = _passed + _failed
    print(f"  Result: {_passed}/{total} passed", end="")
    if _failed:
        print(f"  ({_failed} FAILED)")
    else:
        print("  — all passed!")
    print("="*60 + "\n")

    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
