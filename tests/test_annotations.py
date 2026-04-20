"""Tests for annotation decorators and metadata.

Covers:
- @step: metadata attachment, schema fields, branching parameters
- @task: step collection, child task collection, order conflict detection
- @flow: task collection, nested flow collection
- @global_config: flow collection
- @task / @flow with branch dispatching
"""
import pytest
from pydantic import BaseModel

from flowforge import step, task, flow, global_config, LLMConfig, BranchCondition
from flowforge.errors import OrderConflictError, CompileError
from flowforge.annotations.decorators import _STEP_ATTR, _TASK_ATTR, _FLOW_ATTR, _GLOBAL_ATTR


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

class InputA(BaseModel):
    value: str

class OutputA(BaseModel):
    result: str


# ---------------------------------------------------------------------------
# @step — basic
# ---------------------------------------------------------------------------

def test_step_attaches_meta():
    """@step attaches StepMeta to the decorated function."""
    @step(order=1, prompt="do something")
    async def my_step(ctx):
        ...

    meta = getattr(my_step, _STEP_ATTR)
    assert meta.order == 1
    assert meta.prompt == "do something"
    assert meta.func.__name__ == "my_step"


def test_step_with_schemas():
    """@step correctly stores explicit input/output schemas."""
    @step(order=2, prompt="test", input_schema=InputA, output_schema=OutputA)
    async def typed_step(ctx):
        ...

    meta = getattr(typed_step, _STEP_ATTR)
    assert meta.input_schema is InputA
    assert meta.output_schema is OutputA


def test_step_is_not_branch_by_default():
    """A plain @step is not a branch dispatcher."""
    @step(order=1, prompt="normal step")
    async def s(ctx): ...

    meta = getattr(s, _STEP_ATTR)
    assert meta.is_branch is False


# ---------------------------------------------------------------------------
# @step — branch dispatching
# ---------------------------------------------------------------------------

async def handler_a(ctx): ...
async def handler_b(ctx): ...


def test_step_branch_attaches_condition():
    """@step with condition/branches/fallback sets is_branch=True."""
    @step(
        order=2,
        prompt="route",
        condition=BranchCondition(field="type", enum=["a", "b"]),
        branches={"a": handler_a, "b": handler_b},
        fallback=handler_a,
    )
    async def router(ctx):
        ...

    meta = getattr(router, _STEP_ATTR)
    assert meta.is_branch is True
    assert "a" in meta.branches
    assert "b" in meta.branches
    assert meta.fallback is handler_a


def test_step_branch_stored_as_step_meta():
    """Branching @step is stored under _STEP_ATTR — NOT a separate BranchMeta."""
    @step(
        order=1,
        prompt="dispatch",
        condition=BranchCondition(field="x", enum=["a"]),
        branches={"a": handler_a},
    )
    async def dispatch(ctx): ...

    # Must carry _STEP_ATTR; there is no _BRANCH_ATTR anymore.
    assert hasattr(dispatch, _STEP_ATTR)


# ---------------------------------------------------------------------------
# @task — step / child-task collection
# ---------------------------------------------------------------------------

def test_task_collects_steps():
    """@task collects all @step-decorated functions from its class body."""
    @task(name="my_task", prompt="do task")
    class MyTask:
        @step(order=1, prompt="step one")
        async def step_one(ctx): ...

        @step(order=2, prompt="step two")
        async def step_two(ctx): ...

    meta = getattr(MyTask, _TASK_ATTR)
    assert meta.name == "my_task"
    assert len(meta.steps) == 2
    assert meta.is_leaf is True


def test_task_collects_branching_steps():
    """@task also collects @step nodes that are branch dispatchers."""
    @task(name="branch_task", prompt="branch task")
    class BranchTask:
        @step(order=1, prompt="normal")
        async def normal(ctx): ...

        @step(
            order=2,
            prompt="dispatch",
            condition=BranchCondition(field="kind", enum=["a", "b"]),
            branches={"a": handler_a, "b": handler_b},
        )
        async def dispatch(ctx): ...

    meta = getattr(BranchTask, _TASK_ATTR)
    assert len(meta.steps) == 2
    # The branching step is still a StepMeta
    branching = next(s for s in meta.steps if s.is_branch)
    assert branching.order == 2


def test_task_same_order_steps_are_parallel_group():
    """Two @step nodes with the same order form a parallel group — no error raised."""
    @task(name="parallel_task", prompt="test")
    class ParallelTask:
        @step(order=1, prompt="first")
        async def step_a(ctx): ...

        @step(order=1, prompt="parallel sibling — same order is allowed")
        async def step_b(ctx): ...

    meta = getattr(ParallelTask, _TASK_ATTR)
    assert len(meta.steps) == 2
    assert all(s.order == 1 for s in meta.steps)


def test_task_unique_conflict_raises():
    """Two @step nodes with the same order AND both unique=True raise OrderConflictError."""
    with pytest.raises(OrderConflictError):
        @task(name="conflict_task", prompt="test")
        class ConflictTask:
            @step(order=1, prompt="first", unique=True)
            async def step_x(ctx): ...

            @step(order=1, prompt="second", unique=True)
            async def step_y(ctx): ...


def test_task_branching_step_and_regular_step_same_order_allowed():
    """A branching @step and a regular @step can share the same order (parallel)."""
    async def h(ctx): ...

    @task(name="mix_task", prompt="mix")
    class MixTask:
        @step(order=1, prompt="regular step")
        async def s(ctx): ...

        # order=1 alongside another step is now allowed — they run in parallel.
        @step(
            order=1,
            prompt="branching step — parallel with s",
            condition=BranchCondition(field="x", enum=["a"]),
            branches={"a": h},
        )
        async def b(ctx): ...

    meta = getattr(MixTask, _TASK_ATTR)
    assert len(meta.steps) == 2


def test_task_with_child_tasks():
    """@task recognises nested @task classes (container task pattern)."""
    @task(name="parent_task", prompt="parent")
    class ParentTask:
        @task(name="child_one", prompt="child 1")
        class ChildOne:
            @step(order=1, prompt="s1")
            async def s1(ctx): ...

        @task(name="child_two", prompt="child 2")
        class ChildTwo:
            @step(order=1, prompt="s2")
            async def s2(ctx): ...

    meta = getattr(ParentTask, _TASK_ATTR)
    assert meta.is_leaf is False
    assert len(meta.child_tasks) == 2


# ---------------------------------------------------------------------------
# @task — branch dispatching
# ---------------------------------------------------------------------------

def test_task_branch_dispatching():
    """@task with condition/branches becomes a branch dispatcher."""
    @task(name="web_search", prompt="web")
    class WebSearch:
        @step(order=1, prompt="search")
        async def search(ctx): ...

    @task(name="db_search", prompt="db")
    class DBSearch:
        @step(order=1, prompt="query")
        async def query(ctx): ...

    @task(
        name="dispatch_search",
        prompt="route to correct search backend",
        condition=BranchCondition(field="source", enum=["web", "db"]),
        branches={"web": WebSearch, "db": DBSearch},
        fallback=WebSearch,
    )
    class DispatchSearch: ...

    meta = getattr(DispatchSearch, _TASK_ATTR)
    assert meta.is_branch is True
    assert meta.is_leaf is False
    assert "web" in meta.branches
    assert "db" in meta.branches
    assert meta.fallback is not None
    assert meta.fallback.name == "web_search"


def test_task_branch_requires_task_decorated_classes():
    """Passing a non-@task class as a branch target raises CompileError."""
    class NotATask: ...  # missing @task

    with pytest.raises(CompileError):
        @task(
            name="bad_dispatch",
            prompt="bad",
            condition=BranchCondition(field="x", enum=["a"]),
            branches={"a": NotATask},
        )
        class BadDispatch: ...


# ---------------------------------------------------------------------------
# @flow — task / nested-flow collection
# ---------------------------------------------------------------------------

def test_flow_collects_tasks():
    """@flow collects @task-decorated inner classes."""
    @flow(name="my_flow", prompt="flow")
    class MyFlow:
        @task(name="t1", prompt="task one")
        class T1:
            @step(order=1, prompt="s")
            async def s(ctx): ...

    meta = getattr(MyFlow, _FLOW_ATTR)
    assert meta.name == "my_flow"
    assert len(meta.tasks) == 1


def test_flow_nested_flows():
    """@flow recognises nested @flow classes (child-flow pattern)."""
    @flow(name="outer_flow", prompt="outer")
    class OuterFlow:
        @flow(name="inner_flow", prompt="inner")
        class InnerFlow:
            pass

    meta = getattr(OuterFlow, _FLOW_ATTR)
    assert len(meta.child_flows) == 1
    assert meta.child_flows[0].name == "inner_flow"


# ---------------------------------------------------------------------------
# @flow — branch dispatching
# ---------------------------------------------------------------------------

def test_flow_branch_dispatching():
    """@flow with condition/branches becomes a branch dispatcher."""
    @flow(name="fast_process", prompt="fast")
    class FastProcess: ...

    @flow(name="thorough_process", prompt="thorough")
    class ThoroughProcess: ...

    @flow(
        name="process_dispatch",
        prompt="route to correct processing mode",
        condition=BranchCondition(field="mode", enum=["fast", "thorough"]),
        branches={"fast": FastProcess, "thorough": ThoroughProcess},
        fallback=FastProcess,
    )
    class ProcessDispatch: ...

    meta = getattr(ProcessDispatch, _FLOW_ATTR)
    assert meta.is_branch is True
    assert "fast" in meta.branches
    assert "thorough" in meta.branches
    assert meta.fallback.name == "fast_process"


def test_flow_branch_requires_flow_decorated_classes():
    """Passing a non-@flow class as a branch target raises CompileError."""
    class NotAFlow: ...

    with pytest.raises(CompileError):
        @flow(
            name="bad_dispatch",
            prompt="bad",
            condition=BranchCondition(field="x", enum=["a"]),
            branches={"a": NotAFlow},
        )
        class BadDispatch: ...


# ---------------------------------------------------------------------------
# @global_config
# ---------------------------------------------------------------------------

def test_global_config_collects_flows():
    """@global_config collects @flow-decorated inner classes."""
    @global_config(prompt="agent prompt")
    class MyAgent:
        @flow(name="root_flow", prompt="root")
        class RootFlow:
            pass

    meta = getattr(MyAgent, _GLOBAL_ATTR)
    assert meta.prompt == "agent prompt"
    assert len(meta.flows) == 1
    assert meta.flows[0].name == "root_flow"
