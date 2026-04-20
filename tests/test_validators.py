"""Tests for annotation compile-time validators.

Covers:
- validate_order_uniqueness: same-order steps are now allowed (parallel);
  only two unique=True steps at the same order raise OrderConflictError
- validate_io_chain: None schemas skip, matching pass, mismatching raise;
  within-group (same-order) pairs are skipped
- validate_step_branch_handlers: consistent return types pass,
  inconsistent raise BranchOutputMismatchError
"""
import pytest
from pydantic import BaseModel

from flowforge import step, task, BranchCondition
from flowforge.errors import OrderConflictError, IOBindingError, BranchOutputMismatchError
from flowforge.annotations.validators import (
    validate_order_uniqueness,
    validate_io_chain,
    validate_step_branch_handlers,
)
from flowforge.annotations.metadata import StepMeta, TaskMeta


# ---------------------------------------------------------------------------
# Order uniqueness
# ---------------------------------------------------------------------------

def test_order_uniqueness_pass():
    """Two steps with different orders in the same task should not raise."""
    @task(name="ok_task", prompt="ok")
    class OkTask:
        @step(order=1, prompt="s1")
        async def s1(ctx): ...

        @step(order=2, prompt="s2")
        async def s2(ctx): ...

    # No exception expected — test passes if we reach this line.


def test_same_order_steps_are_allowed():
    """Two steps with the same order in the same task do NOT raise — they run in parallel."""
    @task(name="parallel_task", prompt="parallel")
    class ParallelTask:
        @step(order=1, prompt="a")
        async def a(ctx): ...

        @step(order=1, prompt="b")
        async def b(ctx): ...

    # No exception expected — same-order steps are a parallel group.


def test_branching_step_and_regular_step_same_order_allowed():
    """A branching @step and a regular @step can share the same order (parallel group)."""
    async def h(ctx): ...

    @task(name="mix_task", prompt="mix")
    class MixTask:
        @step(order=1, prompt="regular step")
        async def s(ctx): ...

        # order=1 is now allowed alongside another step — they run in parallel.
        @step(
            order=1,
            prompt="branching step — parallel with s",
            condition=BranchCondition(field="x", enum=["a"]),
            branches={"a": h},
        )
        async def b(ctx): ...

    # No exception expected.


def test_unique_conflict_raises():
    """Two steps at the same order both with unique=True raise OrderConflictError."""
    with pytest.raises(OrderConflictError) as exc_info:
        @task(name="conflict_task", prompt="conflict")
        class ConflictTask:
            @step(order=1, prompt="a", unique=True)
            async def a(ctx): ...

            @step(order=1, prompt="b", unique=True)
            async def b(ctx): ...

    assert "conflict_task" in str(exc_info.value)
    assert exc_info.value.order == 1


def test_unique_single_passes():
    """One unique=True step at a given order (alongside non-unique siblings) is allowed."""
    @task(name="unique_ok_task", prompt="ok")
    class UniqueOkTask:
        @step(order=1, prompt="winner", unique=True)
        async def winner(ctx): ...

        @step(order=1, prompt="skipped")
        async def skipped(ctx): ...


# ---------------------------------------------------------------------------
# I/O chain validation (non-strict — None schemas skip the check)
# ---------------------------------------------------------------------------

class TypeA(BaseModel):
    a: int

class TypeB(BaseModel):
    b: int


def test_io_chain_none_schemas_pass():
    """When either schema is None auto-binding is assumed — no error raised."""
    async def s1(ctx): ...
    async def s2(ctx): ...

    m = TaskMeta(
        name="t",
        prompt="t",
        cls=object,
        steps=[
            StepMeta(order=1, prompt="s1", func=s1, output_schema=None),
            StepMeta(order=2, prompt="s2", func=s2, input_schema=None),
        ],
    )
    validate_io_chain(m)  # Must not raise.


def test_io_chain_matching_schemas_pass():
    """Consecutive steps whose schemas match should not raise."""
    async def s1(ctx): ...
    async def s2(ctx): ...

    m = TaskMeta(
        name="t",
        prompt="t",
        cls=object,
        steps=[
            StepMeta(order=1, prompt="s1", func=s1, output_schema=TypeA),
            StepMeta(order=2, prompt="s2", func=s2, input_schema=TypeA),
        ],
    )
    validate_io_chain(m)  # Must not raise.


def test_io_chain_mismatching_schemas_fail():
    """Consecutive steps whose schemas differ (both declared) raise IOBindingError."""
    async def s1(ctx): ...
    async def s2(ctx): ...

    m = TaskMeta(
        name="t",
        prompt="t",
        cls=object,
        steps=[
            StepMeta(order=1, prompt="s1", func=s1, output_schema=TypeA),
            StepMeta(order=2, prompt="s2", func=s2, input_schema=TypeB),  # mismatch
        ],
    )
    with pytest.raises(IOBindingError):
        validate_io_chain(m)


def test_io_chain_skips_within_parallel_group():
    """Steps in the same parallel order group are not validated against each other."""
    async def s1(ctx): ...
    async def s2(ctx): ...  # same order as s1 — would be flagged if checked

    # Both have order=1 (parallel group): s1 outputs TypeA, s2 inputs TypeB.
    # Since they are in the same order group they don't chain — no error.
    m = TaskMeta(
        name="t",
        prompt="t",
        cls=object,
        steps=[
            StepMeta(order=1, prompt="s1", func=s1, output_schema=TypeA),
            StepMeta(order=1, prompt="s2", func=s2, input_schema=TypeB),
        ],
    )
    validate_io_chain(m)  # Must not raise.


# ---------------------------------------------------------------------------
# Step branch handler consistency
# ---------------------------------------------------------------------------

def test_step_branch_handlers_same_type_pass():
    """Handlers that all return the same type should not raise."""
    class Result(BaseModel):
        value: str

    async def h_a(ctx) -> Result: ...
    async def h_b(ctx) -> Result: ...

    async def my_step(ctx): ...
    meta = StepMeta(
        order=1,
        prompt="dispatch",
        func=my_step,
        condition=BranchCondition(field="x", enum=["a", "b"]),
        branches={"a": h_a, "b": h_b},
    )
    validate_step_branch_handlers(meta)  # Must not raise.


def test_step_branch_handlers_no_annotations_pass():
    """Handlers without return annotations are assumed compatible — no error."""
    async def h_a(ctx): ...   # no annotation
    async def h_b(ctx): ...   # no annotation

    async def my_step(ctx): ...
    meta = StepMeta(
        order=1,
        prompt="dispatch",
        func=my_step,
        condition=BranchCondition(field="x", enum=["a", "b"]),
        branches={"a": h_a, "b": h_b},
    )
    validate_step_branch_handlers(meta)  # Must not raise.


def test_step_branch_handlers_different_types_fail():
    """Handlers with different return-type annotations raise BranchOutputMismatchError."""
    class TypeX(BaseModel):
        x: int

    class TypeY(BaseModel):
        y: int

    async def h_a(ctx) -> TypeX: ...
    async def h_b(ctx) -> TypeY: ...

    async def my_step(ctx): ...
    meta = StepMeta(
        order=1,
        prompt="dispatch",
        func=my_step,
        condition=BranchCondition(field="x", enum=["a", "b"]),
        branches={"a": h_a, "b": h_b},
    )
    with pytest.raises(BranchOutputMismatchError):
        validate_step_branch_handlers(meta)


def test_step_branch_decorator_validates_handlers():
    """The @step decorator runs validate_step_branch_handlers at decoration time."""
    class T1(BaseModel):
        v: int

    class T2(BaseModel):
        v: str

    async def h1(ctx) -> T1: ...
    async def h2(ctx) -> T2: ...

    with pytest.raises(BranchOutputMismatchError):
        # The error should be raised at decoration time, not at run time.
        @step(
            order=1,
            prompt="bad dispatch",
            condition=BranchCondition(field="x", enum=["a", "b"]),
            branches={"a": h1, "b": h2},
        )
        async def bad_dispatch(ctx): ...
