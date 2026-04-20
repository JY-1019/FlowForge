# Error Reference

All errors are importable from `flowforge` or `flowforge.errors`.

---

## Compile-Time Errors

These are raised during `FlowForge.compile()` or at decoration time (import).

### OrderConflictError

Two `@step` or `@branch` decorators share the same `order` value within a single leaf task.

```python
@task(name="process")
class ProcessTask:
    @step(order=1, prompt="...")
    async def a(ctx): ...

    @step(order=1, prompt="...")  # ❌ OrderConflictError!
    async def b(ctx): ...
```

**Fix:** Assign unique `order` values.

---

### IOBindingError

The `output_schema` of step N is not compatible with the `input_schema` of step N+1.

```python
class A(BaseModel): x: int
class B(BaseModel): y: str

@step(order=1, output_schema=A)
async def step1(ctx): ...

@step(order=2, input_schema=B)  # ❌ A ≠ B → IOBindingError
async def step2(ctx): ...
```

**Fix:** Align schemas, or omit them to skip validation.

---

### BranchOutputMismatchError

Branch handlers return different types.

```python
async def handler_a(ctx) -> TypeA: ...
async def handler_b(ctx) -> TypeB: ...  # ❌ TypeA ≠ TypeB

@branch(..., branches={"a": handler_a, "b": handler_b})
async def route(ctx): ...
```

**Fix:** Make all handlers return the same type.

---

### CycleDetectedError

`depends_on` references create a cycle in the DAG.

```python
@flow(name="a", depends_on=["b"])
class FlowA: ...

@flow(name="b", depends_on=["a"])  # ❌ a→b→a cycle
class FlowB: ...
```

---

### CompileError

General compile failure — the class is not decorated with `@global_config`, or the DAG is structurally invalid.

```python
engine = FlowForge.compile(NotAnAgent)  # ❌ CompileError
```

---

## Runtime Errors

### ExecutionError

Raised when a step, branch, task, or flow fails during execution.

```python
try:
    result = await engine.run(my_input)
except ExecutionError as e:
    print(f"Node '{e.node_name}' failed: {e.message}")
```

---

## Base Class

All errors inherit from `FlowForgeError`:

```python
from flowforge import FlowForgeError

try:
    engine = FlowForge.compile(MyAgent)
    result = await engine.run(data)
except FlowForgeError as e:
    print(f"FlowForge error: {e}")
```
