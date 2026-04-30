# Error Reference

All errors are importable from `flowforge` or `flowforge.errors`.

## Compile-Time Errors

### `CompileError`

General compile failure. Common causes:

- The class passed to `FlowForge.compile()` is not decorated with
  `@global_config`.
- A branch task target is not decorated with `@task`.
- A branch flow target is not decorated with `@flow`.
- A dynamically added flow is invalid or duplicates an existing name.

### `CycleDetectedError`

Raised when `depends_on` edges create a cycle.

```python
@flow(name="a", prompt="A", depends_on=["b"])
class A: ...

@flow(name="b", prompt="B", depends_on=["a"])
class B: ...
```

### `OrderConflictError`

Raised when more than one sibling in the same explicit order group sets
`unique=True`.

```python
@task(name="bad", prompt="bad")
class BadTask:
    @step(order=1, prompt="a", unique=True)
    async def a(ctx): ...

    @step(order=1, prompt="b", unique=True)
    async def b(ctx): ...
```

Same `order` without duplicate `unique=True` is valid and means parallel
execution.

### `IOBindingError`

Raised when consecutive step groups have incompatible schemas.

```python
class A(BaseModel): x: int
class B(BaseModel): y: str

@step(order=1, prompt="one", output_schema=A)
async def one(ctx): ...

@step(order=2, prompt="two", input_schema=B)
async def two(ctx): ...
```

### `BranchOutputMismatchError`

Raised when handlers for a branching `@step` advertise incompatible return
type annotations.

```python
async def handle_a(ctx) -> TypeA: ...
async def handle_b(ctx) -> TypeB: ...

@step(
    order=1,
    prompt="route",
    condition=BranchCondition(field="kind", enum=["a", "b"]),
    branches={"a": handle_a, "b": handle_b},
)
async def route(ctx): ...
```

Make all branch handlers return the same type or omit incompatible annotations.

## Runtime Errors

### `ExecutionError`

Raised when a flow/task/step fails during execution and the error is not
handled by retries or `on_error="skip_remaining"`.

### `ApprovalRequired`

Raised before a step marked `approval=True` executes.

```python
try:
    await engine.run(data)
except ApprovalRequired as exc:
    checkpoint = exc.checkpoint
    # after external approval:
    result = await engine.run(data, resume_from=checkpoint)
```

### `PlannerError`

Raised when autonomous or hybrid planning fails, returns an invalid selection,
or requests dynamic generation without enough valid generation input.
