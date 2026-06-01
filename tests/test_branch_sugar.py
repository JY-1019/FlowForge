"""Tests for the single-object convenience forms:

- ``branch=Branch(on=..., cases=..., default=...)`` on @step / @task / @flow
- ``pass_criteria=PassCriteria(criteria, max_retries=...)`` on @step

These fold down to the same ``condition`` / ``branches`` / ``fallback`` and
``pass_criteria`` / ``pass_criteria_max_retries`` fields the metadata
dataclasses already understand, so they must stay equivalent to the legacy
parameter forms.
"""
import pytest

from flowforge import (
    Branch,
    BranchCondition,
    PassCriteria,
    flow,
    step,
    task,
)
from flowforge.errors import CompileError
from flowforge.annotations.decorators import _STEP_ATTR, _TASK_ATTR, _FLOW_ATTR


async def handler_a(ctx): ...
async def handler_b(ctx): ...


# ---------------------------------------------------------------------------
# Branch on @step
# ---------------------------------------------------------------------------

def test_step_branch_object_equivalent_to_legacy():
    @step(
        order=1,
        prompt="route",
        branch=Branch(
            on="kind",
            cases={"a": handler_a, "b": handler_b},
            default=handler_a,
        ),
    )
    async def router(ctx): ...

    meta = getattr(router, _STEP_ATTR)
    assert meta.is_branch is True
    assert meta.condition.field == "kind"
    # enum is derived from cases keys
    assert meta.condition.enum == ["a", "b"]
    assert set(meta.branches) == {"a", "b"}
    assert meta.fallback is handler_a


def test_step_branch_default_optional():
    @step(
        order=1,
        prompt="route",
        branch=Branch(on="kind", cases={"a": handler_a}),
    )
    async def router(ctx): ...

    meta = getattr(router, _STEP_ATTR)
    assert meta.fallback is None


def test_step_branch_conflict_with_legacy_raises():
    with pytest.raises(CompileError):
        @step(
            order=1,
            prompt="route",
            branch=Branch(on="kind", cases={"a": handler_a}),
            condition=BranchCondition(field="kind", enum=["a"]),
        )
        async def router(ctx): ...


# ---------------------------------------------------------------------------
# Branch on @task
# ---------------------------------------------------------------------------

def test_task_branch_object():
    @task(name="ta", prompt="task a")
    class TaskA:
        @step(order=1, prompt="s")
        async def s(ctx): ...

    @task(name="tb", prompt="task b")
    class TaskB:
        @step(order=1, prompt="s")
        async def s(ctx): ...

    @task(
        name="dispatch",
        prompt="dispatch",
        branch=Branch(on="kind", cases={"a": TaskA, "b": TaskB}, default=TaskA),
    )
    class Dispatch: ...

    meta = getattr(Dispatch, _TASK_ATTR)
    assert meta.is_branch is True
    assert meta.condition.field == "kind"
    assert set(meta.branches) == {"a", "b"}
    assert meta.fallback is getattr(TaskA, _TASK_ATTR)


def test_task_branch_rejects_undecorated_target():
    class NotATask: ...

    with pytest.raises(CompileError):
        @task(
            name="dispatch",
            prompt="dispatch",
            branch=Branch(on="kind", cases={"a": NotATask}),
        )
        class Dispatch: ...


# ---------------------------------------------------------------------------
# Branch on @flow
# ---------------------------------------------------------------------------

def test_flow_branch_object():
    @flow(name="fa", prompt="flow a")
    class FlowA:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="s")
            async def s(ctx): ...

    @flow(name="fb", prompt="flow b")
    class FlowB:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="s")
            async def s(ctx): ...

    @flow(
        name="dispatch",
        prompt="dispatch",
        branch=Branch(on="kind", cases={"a": FlowA, "b": FlowB}),
    )
    class Dispatch: ...

    meta = getattr(Dispatch, _FLOW_ATTR)
    assert meta.is_branch is True
    assert set(meta.branches) == {"a", "b"}
    assert meta.fallback is None


# ---------------------------------------------------------------------------
# PassCriteria on @step
# ---------------------------------------------------------------------------

def test_pass_criteria_object_equivalent_to_legacy():
    @step(
        order=1,
        prompt="answer",
        pass_criteria=PassCriteria("must be korean", max_retries=1),
    )
    async def answer(ctx): ...

    meta = getattr(answer, _STEP_ATTR)
    assert meta.pass_criteria == "must be korean"
    assert meta.pass_criteria_max_retries == 1


def test_pass_criteria_string_still_supported():
    @step(
        order=1,
        prompt="answer",
        pass_criteria="must be korean",
        pass_criteria_max_retries=2,
    )
    async def answer(ctx): ...

    meta = getattr(answer, _STEP_ATTR)
    assert meta.pass_criteria == "must be korean"
    assert meta.pass_criteria_max_retries == 2
