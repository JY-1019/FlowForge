"""Tests for execution runners.

Covers:
- StepRunner: basic invocation, result storage
- Full agent compile-and-run via FlowForge.compile()
- Step-level branch dispatching (condition / branches / fallback)
- Task-level branch dispatching
- Flow-level branch dispatching
"""
import pytest
from pydantic import BaseModel

from flowforge import global_config, flow, task, step, FlowForge, BranchCondition
from flowforge.execution.context import GlobalContext, FlowContext, TaskContext
from flowforge.execution.runner import StepRunner, TaskRunner, FlowRunner, _try_parse_json
from flowforge.annotations.metadata import StepMeta, TaskMeta, FlowMeta
from flowforge.tools.registry import ToolRegistry
from flowforge.types import LLMConfig


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Num(BaseModel):
    value: int

class NumResult(BaseModel):
    result: int


# ---------------------------------------------------------------------------
# Context factory helpers
# ---------------------------------------------------------------------------

def make_global_ctx():
    return GlobalContext(
        llm_config=LLMConfig(),
        global_prompt="test",
        tool_registry=ToolRegistry(),
    )


def make_flow_ctx(global_ctx=None):
    gctx = global_ctx or make_global_ctx()
    return FlowContext(
        global_ctx=gctx,
        flow_name="test_flow",
        flow_prompt="test",
        flow_input=None,
    )


def make_task_ctx(flow_ctx=None):
    fctx = flow_ctx or make_flow_ctx()
    return TaskContext(
        flow_ctx=fctx,
        task_name="test_task",
        task_prompt="test",
    )


# ---------------------------------------------------------------------------
# StepRunner — basic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_step_runner_calls_function():
    """StepRunner invokes the step function with the correct StepContext."""
    called_with = {}

    async def my_step(ctx):
        called_with["ctx"] = ctx
        return {"value": 42}

    meta     = StepMeta(order=1, prompt="test", func=my_step)
    runner   = StepRunner()
    task_ctx = make_task_ctx()

    await runner.run(meta, task_ctx, None)
    assert called_with["ctx"].order == 1


@pytest.mark.asyncio
async def test_step_runner_stores_result():
    """StepRunner stores the step result in task_ctx.step_results."""
    async def my_step(ctx):
        return "output_value"

    meta     = StepMeta(order=1, prompt="test", func=my_step)
    runner   = StepRunner()
    task_ctx = make_task_ctx()

    await runner.run(meta, task_ctx, None)
    assert 1 in task_ctx.step_results
    assert task_ctx.step_results[1] == "output_value"


# ---------------------------------------------------------------------------
# Full-agent execution — step ordering
# ---------------------------------------------------------------------------

call_log: list[str] = []


@task(name="counter_task", prompt="increment counter")
class CounterTask:
    @step(order=1, prompt="add one")
    async def add_one(ctx):
        call_log.append("add_one")
        return ctx.input

    @step(order=2, prompt="add two")
    async def add_two(ctx):
        call_log.append("add_two")
        return ctx.input


@flow(name="counter_flow", prompt="counting flow")
class CounterFlow:
    CounterTask = CounterTask


@global_config(prompt="counter agent")
class CounterAgent:
    CounterFlow = CounterFlow


@pytest.mark.asyncio
async def test_task_steps_execute_in_order():
    """Steps within a task are executed in ascending order."""
    call_log.clear()
    engine = FlowForge.compile(CounterAgent)
    await engine.run("hello")
    assert call_log == ["add_one", "add_two"]


# ---------------------------------------------------------------------------
# Step-level branch dispatching
# ---------------------------------------------------------------------------

branch_log: list[str] = []


async def branch_a(ctx):
    branch_log.append("a")
    return "branch_a_result"


async def branch_b(ctx):
    branch_log.append("b")
    return "branch_b_result"


class RouteInput(BaseModel):
    route: str


@task(name="routing_task", prompt="route test")
class RoutingTask:
    # @step with condition/branches replaces the old standalone @branch.
    @step(
        order=1,
        prompt="pick a or b",
        condition=BranchCondition(field="route", enum=["a", "b"]),
        branches={"a": branch_a, "b": branch_b},
        fallback=branch_b,
    )
    async def router(ctx):
        # This function body is called when no branch matches and no fallback
        # is configured.  With a fallback set, it is never reached in normal
        # operation — but it must be async.
        ...


@flow(name="routing_flow", prompt="routing")
class RoutingFlow:
    RoutingTask = RoutingTask


@global_config(prompt="routing agent")
class RoutingAgent:
    RoutingFlow = RoutingFlow


@pytest.mark.asyncio
async def test_step_branch_takes_correct_path():
    """A branching @step dispatches to the handler matching condition.field."""
    branch_log.clear()
    engine = FlowForge.compile(RoutingAgent)
    await engine.run(RouteInput(route="a"))
    assert branch_log == ["a"]


@pytest.mark.asyncio
async def test_step_branch_fallback():
    """When condition value is not in branches, the fallback handler is called."""
    branch_log.clear()
    engine = FlowForge.compile(RoutingAgent)
    await engine.run(RouteInput(route="unknown"))
    assert branch_log == ["b"]


@pytest.mark.asyncio
async def test_step_branch_selected_branch_in_context():
    """The StepContext.selected_branch field is set by the runner."""
    # Use ContextBranchAgent defined at module level (class bodies cannot
    # reference local variables from an enclosing function scope in Python).
    ctx_branch_log.clear()
    engine = FlowForge.compile(ContextBranchAgent)
    await engine.run(ContextInput(kind="x"))
    assert ctx_branch_log == ["x"]


# ---------------------------------------------------------------------------
# StepContext.selected_branch — module-level agent for scoping
# ---------------------------------------------------------------------------

# Python does not allow a class body to reference a local variable from an
# enclosing function scope (e.g. `CtxTask = CtxTask` would be ambiguous).
# Agents used in parametrised / closure-heavy tests are therefore defined at
# module level.

ctx_branch_log: list[str] = []


class ContextInput(BaseModel):
    kind: str


async def ctx_recording_handler(ctx):
    """Records selected_branch so the test can assert on it."""
    ctx_branch_log.append(ctx.selected_branch)
    return "ok"


@task(name="ctx_task", prompt="test")
class CtxTask:
    @step(
        order=1,
        prompt="record branch",
        condition=BranchCondition(field="kind", enum=["x"]),
        branches={"x": ctx_recording_handler},
    )
    async def dispatch(ctx): ...


@flow(name="ctx_flow", prompt="test")
class CtxFlow:
    CtxTask = CtxTask


@global_config(prompt="ctx agent")
class ContextBranchAgent:
    CtxFlow = CtxFlow


# ---------------------------------------------------------------------------
# Task-level branch dispatching
# ---------------------------------------------------------------------------

task_branch_log: list[str] = []


@task(name="task_branch_a", prompt="task branch a")
class TaskBranchA:
    @step(order=1, prompt="run a")
    async def run_a(ctx):
        task_branch_log.append("task_a")
        return "result_a"


@task(name="task_branch_b", prompt="task branch b")
class TaskBranchB:
    @step(order=1, prompt="run b")
    async def run_b(ctx):
        task_branch_log.append("task_b")
        return "result_b"


class TaskRouteInput(BaseModel):
    backend: str


@task(
    name="dispatch_task",
    prompt="dispatch to a or b",
    condition=BranchCondition(field="backend", enum=["a", "b"]),
    branches={"a": TaskBranchA, "b": TaskBranchB},
    fallback=TaskBranchA,
)
class DispatchTask: ...


@flow(name="task_branch_flow", prompt="task branch flow")
class TaskBranchFlow:
    DispatchTask = DispatchTask


@global_config(prompt="task branch agent")
class TaskBranchAgent:
    TaskBranchFlow = TaskBranchFlow


@pytest.mark.asyncio
async def test_task_branch_dispatches_correctly():
    """A @task with condition/branches delegates to the matching task."""
    task_branch_log.clear()
    engine = FlowForge.compile(TaskBranchAgent)
    await engine.run(TaskRouteInput(backend="b"))
    assert task_branch_log == ["task_b"]


@pytest.mark.asyncio
async def test_task_branch_fallback():
    """When no task branch matches, the fallback task is executed."""
    task_branch_log.clear()
    engine = FlowForge.compile(TaskBranchAgent)
    await engine.run(TaskRouteInput(backend="unknown"))
    assert task_branch_log == ["task_a"]


# ---------------------------------------------------------------------------
# Parallel step execution (same order → run concurrently)
# ---------------------------------------------------------------------------

parallel_step_log: list[str] = []


@task(name="parallel_steps_task", prompt="parallel steps test")
class ParallelStepsTask:
    @step(order=1, prompt="step a")
    async def step_a(ctx):
        parallel_step_log.append("a")
        return "a_result"

    @step(order=1, prompt="step b — same order as a, runs in parallel")
    async def step_b(ctx):
        parallel_step_log.append("b")
        return "b_result"

    @step(order=2, prompt="step c — after the parallel group")
    async def step_c(ctx):
        parallel_step_log.append("c")
        return ctx.input


@flow(name="parallel_steps_flow", prompt="parallel steps")
class ParallelStepsFlow:
    ParallelStepsTask = ParallelStepsTask


@global_config(prompt="parallel steps agent")
class ParallelStepsAgent:
    ParallelStepsFlow = ParallelStepsFlow


@pytest.mark.asyncio
async def test_parallel_steps_both_run():
    """Two @step nodes sharing the same order both execute (in parallel)."""
    parallel_step_log.clear()
    engine = FlowForge.compile(ParallelStepsAgent)
    await engine.run("input")
    assert "a" in parallel_step_log
    assert "b" in parallel_step_log
    assert "c" in parallel_step_log


@pytest.mark.asyncio
async def test_parallel_steps_followed_by_sequential():
    """The step after a parallel group runs once, after both parallel steps finish."""
    parallel_step_log.clear()
    engine = FlowForge.compile(ParallelStepsAgent)
    await engine.run("input")
    # c must appear after both a and b (order=2 runs after order=1 group)
    assert parallel_step_log.index("c") > parallel_step_log.index("a")
    assert parallel_step_log.index("c") > parallel_step_log.index("b")


# ---------------------------------------------------------------------------
# unique=True step — exclusive representative of its order group
# ---------------------------------------------------------------------------

unique_step_log: list[str] = []


@task(name="unique_step_task", prompt="unique step test")
class UniqueStepTask:
    @step(order=1, prompt="winner — unique", unique=True)
    async def winner(ctx):
        unique_step_log.append("winner")
        return "winner_result"

    @step(order=1, prompt="skipped — not unique")
    async def skipped(ctx):
        unique_step_log.append("skipped")
        return "skipped_result"


@flow(name="unique_step_flow", prompt="unique step")
class UniqueStepFlow:
    UniqueStepTask = UniqueStepTask


@global_config(prompt="unique step agent")
class UniqueStepAgent:
    UniqueStepFlow = UniqueStepFlow


@pytest.mark.asyncio
async def test_unique_step_skips_siblings():
    """A step with unique=True is the sole runner in its order group."""
    unique_step_log.clear()
    engine = FlowForge.compile(UniqueStepAgent)
    await engine.run("input")
    assert unique_step_log == ["winner"]


# ---------------------------------------------------------------------------
# Parallel task execution (same order → run concurrently)
# ---------------------------------------------------------------------------

parallel_task_log: list[str] = []


@task(name="ptask_a", prompt="parallel task a", order=1)
class PTaskA:
    @step(order=1, prompt="run a")
    async def run_a(ctx):
        parallel_task_log.append("ptask_a")
        return "a"


@task(name="ptask_b", prompt="parallel task b", order=1)
class PTaskB:
    @step(order=1, prompt="run b")
    async def run_b(ctx):
        parallel_task_log.append("ptask_b")
        return "b"


@task(name="ptask_c", prompt="sequential task c — after the parallel group", order=2)
class PTaskC:
    @step(order=1, prompt="run c")
    async def run_c(ctx):
        parallel_task_log.append("ptask_c")
        return ctx.input


@flow(name="parallel_tasks_flow", prompt="parallel tasks")
class ParallelTasksFlow:
    PTaskA = PTaskA
    PTaskB = PTaskB
    PTaskC = PTaskC


@global_config(prompt="parallel tasks agent")
class ParallelTasksAgent:
    ParallelTasksFlow = ParallelTasksFlow


@pytest.mark.asyncio
async def test_parallel_tasks_both_run():
    """Two @task nodes sharing the same order both execute."""
    parallel_task_log.clear()
    engine = FlowForge.compile(ParallelTasksAgent)
    await engine.run("input")
    assert "ptask_a" in parallel_task_log
    assert "ptask_b" in parallel_task_log
    assert "ptask_c" in parallel_task_log


@pytest.mark.asyncio
async def test_parallel_tasks_sequential_follows():
    """The task in order=2 runs after both order=1 tasks complete."""
    parallel_task_log.clear()
    engine = FlowForge.compile(ParallelTasksAgent)
    await engine.run("input")
    assert parallel_task_log.index("ptask_c") > parallel_task_log.index("ptask_a")
    assert parallel_task_log.index("ptask_c") > parallel_task_log.index("ptask_b")


# ---------------------------------------------------------------------------
# unique=True task — exclusive representative of its order group
# ---------------------------------------------------------------------------

unique_task_log: list[str] = []


@task(name="utask_winner", prompt="unique task winner", order=1, unique=True)
class UTaskWinner:
    @step(order=1, prompt="run winner")
    async def run(ctx):
        unique_task_log.append("winner")
        return "winner"


@task(name="utask_skipped", prompt="unique task skipped", order=1)
class UTaskSkipped:
    @step(order=1, prompt="run skipped")
    async def run(ctx):
        unique_task_log.append("skipped")
        return "skipped"


@flow(name="unique_task_flow", prompt="unique task")
class UniqueTaskFlow:
    UTaskWinner = UTaskWinner
    UTaskSkipped = UTaskSkipped


@global_config(prompt="unique task agent")
class UniqueTaskAgent:
    UniqueTaskFlow = UniqueTaskFlow


@pytest.mark.asyncio
async def test_unique_task_skips_siblings():
    """A @task with unique=True is the sole runner in its order group."""
    unique_task_log.clear()
    engine = FlowForge.compile(UniqueTaskAgent)
    await engine.run("input")
    assert unique_task_log == ["winner"]


# ---------------------------------------------------------------------------
# Flow-level branch dispatching
# ---------------------------------------------------------------------------

flow_branch_log: list[str] = []


@task(name="fast_task", prompt="fast task")
class FastTask:
    @step(order=1, prompt="run fast")
    async def run(ctx):
        flow_branch_log.append("fast")
        return "fast_result"


@task(name="slow_task", prompt="slow task")
class SlowTask:
    @step(order=1, prompt="run slow")
    async def run(ctx):
        flow_branch_log.append("slow")
        return "slow_result"


@flow(name="fast_flow", prompt="fast processing flow")
class FastFlow:
    FastTask = FastTask


@flow(name="slow_flow", prompt="slow processing flow")
class SlowFlow:
    SlowTask = SlowTask


class FlowRouteInput(BaseModel):
    speed: str


@flow(
    name="dispatch_flow",
    prompt="route to fast or slow flow",
    condition=BranchCondition(field="speed", enum=["fast", "slow"]),
    branches={"fast": FastFlow, "slow": SlowFlow},
    fallback=FastFlow,
)
class DispatchFlow: ...


@global_config(prompt="flow branch agent")
class FlowBranchAgent:
    DispatchFlow = DispatchFlow


@pytest.mark.asyncio
async def test_flow_branch_dispatches_correctly():
    """A @flow with condition/branches delegates to the matching flow."""
    flow_branch_log.clear()
    engine = FlowForge.compile(FlowBranchAgent)
    await engine.run(FlowRouteInput(speed="slow"))
    assert flow_branch_log == ["slow"]


@pytest.mark.asyncio
async def test_flow_branch_fallback():
    """When no flow branch matches, the fallback flow is executed."""
    flow_branch_log.clear()
    engine = FlowForge.compile(FlowBranchAgent)
    await engine.run(FlowRouteInput(speed="unknown"))
    assert flow_branch_log == ["fast"]


# ═══════════════════════════════════════════════════════════════════════
# Output schema coercion — LLM string → Pydantic model
# ═══════════════════════════════════════════════════════════════════════

class PageContent(BaseModel):
    url: str
    title: str = ""
    content: str = ""


# --- Pure JSON string ---

@global_config(prompt="json coerce agent")
class JsonCoerceAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="s", output_schema=PageContent)
            async def s(ctx):
                return '{"url": "https://example.com", "title": "Test", "content": "hello"}'


@pytest.mark.asyncio
async def test_output_schema_coerces_json_string():
    """A step returning a JSON string is parsed into output_schema."""
    engine = FlowForge.compile(JsonCoerceAgent)
    result = await engine.run(None)
    assert result.url == "https://example.com"
    assert result.title == "Test"


# --- JSON in markdown fence ---

@global_config(prompt="fence agent")
class FenceAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="s", output_schema=PageContent)
            async def s(ctx):
                return (
                    'Here is the result:\n\n```json\n'
                    '{"url": "https://x.com", "title": "Fenced"}\n'
                    '```\n\nHope that helps!'
                )


@pytest.mark.asyncio
async def test_output_schema_coerces_fenced_json():
    """A step returning markdown-fenced JSON is parsed into output_schema."""
    engine = FlowForge.compile(FenceAgent)
    result = await engine.run(None)
    assert result.url == "https://x.com"
    assert result.title == "Fenced"


# --- JSON embedded in prose ---

@global_config(prompt="prose agent")
class ProseAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="s", output_schema=PageContent)
            async def s(ctx):
                return (
                    '네! 페이지를 분석했습니다.\n\n'
                    '{"url": "https://hn.com", "title": "HN", "content": "top stories"}\n\n'
                    '위 결과를 확인하세요.'
                )


@pytest.mark.asyncio
async def test_output_schema_coerces_embedded_json():
    """A step returning JSON embedded in prose is parsed into output_schema."""
    engine = FlowForge.compile(ProseAgent)
    result = await engine.run(None)
    assert result.url == "https://hn.com"
    assert result.content == "top stories"


# --- Unit tests for _try_parse_json ---

class TestTryParseJson:
    def test_pure_json(self):
        assert _try_parse_json('{"a": 1}') == {"a": 1}

    def test_json_with_whitespace(self):
        assert _try_parse_json('  \n {"a": 1} \n  ') == {"a": 1}

    def test_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        assert _try_parse_json(text) == {"a": 1}

    def test_fenced_no_lang(self):
        text = '```\n{"a": 1}\n```'
        assert _try_parse_json(text) == {"a": 1}

    def test_embedded_in_prose(self):
        text = 'Here you go:\n{"a": 1}\nDone!'
        assert _try_parse_json(text) == {"a": 1}

    def test_no_json(self):
        assert _try_parse_json("just plain text") is None

    def test_invalid_json(self):
        assert _try_parse_json("{not valid json}") is None

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = _try_parse_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}


# --- Doc generator string coercion ---

class TestDocGeneratorLooseTypes:
    """Doc model fields are Any — LLM can return string, dict, or list."""

    def test_string_value_accepted(self):
        from flowforge.doc.generator import DocGenerator
        from flowforge.doc.cache import DocCache
        from flowforge.schema.dag import NodeType
        from flowforge.types import LLMConfig

        gen = DocGenerator(llm_config=LLMConfig(model="test"), cache=DocCache())

        data = {
            "summary": "Test step",
            "input_schema_desc": '{"channel_type": "알림 채널 타입"}',
            "capabilities": '["send notifications"]',
        }
        doc = gen._dict_to_doc(NodeType.STEP, data)
        assert doc.summary == "Test step"
        # String is accepted as-is (no validation error)
        assert doc.input_schema_desc == '{"channel_type": "알림 채널 타입"}'

    def test_dict_value_accepted(self):
        from flowforge.doc.generator import DocGenerator
        from flowforge.doc.cache import DocCache
        from flowforge.schema.dag import NodeType
        from flowforge.types import LLMConfig

        gen = DocGenerator(llm_config=LLMConfig(model="test"), cache=DocCache())

        data = {
            "summary": "OK",
            "input_schema_desc": {"field": "desc"},
            "capabilities": ["cap1", "cap2"],
        }
        doc = gen._dict_to_doc(NodeType.STEP, data)
        assert doc.input_schema_desc == {"field": "desc"}
        assert doc.capabilities == ["cap1", "cap2"]


# ═══════════════════════════════════════════════════════════════════════
# on_error="skip_remaining": skip later steps on error
# ═══════════════════════════════════════════════════════════════════════

@global_config(prompt="on_error agent")
class SkipRemainingAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t", on_error="skip_remaining")
        class T:
            @step(order=1, prompt="succeed")
            async def ok_step(ctx):
                return {"value": 42}

            @step(order=2, prompt="fail here")
            async def bad_step(ctx):
                raise RuntimeError("boom")

            @step(order=3, prompt="never reached")
            async def unreachable(ctx):
                return {"value": 999}


@global_config(prompt="on_error first fail agent")
class SkipRemainingFirstFailAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t", on_error="skip_remaining")
        class T:
            @step(order=1, prompt="fail immediately")
            async def fail_first(ctx):
                raise RuntimeError("first step fails")

            @step(order=2, prompt="never reached")
            async def never(ctx):
                return {"value": 1}


class TestOnErrorSkipRemaining:
    @pytest.mark.asyncio
    async def test_skip_remaining_returns_last_good(self):
        """When a middle step fails, return the last successful output."""
        engine = FlowForge.compile(SkipRemainingAgent)
        result = await engine.run("start")
        # Step 1 succeeded with {"value": 42}, step 2 failed, step 3 skipped
        assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_skip_remaining_first_step_still_raises(self):
        """When the FIRST step fails, there's no prior output — error propagates."""
        engine = FlowForge.compile(SkipRemainingFirstFailAgent)
        from flowforge.errors import ExecutionError
        with pytest.raises(ExecutionError):
            await engine.run("start")

    @pytest.mark.asyncio
    async def test_default_on_error_raises(self):
        """Default on_error='raise' propagates the error."""
        @global_config(prompt="default agent")
        class DefaultAgent:
            @flow(name="f", prompt="f")
            class F:
                @task(name="t", prompt="t")
                class T:
                    @step(order=1, prompt="ok")
                    async def ok(ctx):
                        return {"v": 1}

                    @step(order=2, prompt="fail")
                    async def fail(ctx):
                        raise RuntimeError("fail")

        engine = FlowForge.compile(DefaultAgent)
        from flowforge.errors import ExecutionError
        with pytest.raises(ExecutionError):
            await engine.run("x")
