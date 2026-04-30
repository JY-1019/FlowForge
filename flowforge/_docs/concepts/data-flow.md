# Data Flow & I/O

The most important rule in FlowForge is simple:

> A node's return value becomes the next node's input.

This applies to flows, tasks, step groups, branch dispatchers, routes, and
dynamic generated flows.

## Runtime Chain

```text
engine.run(input_data)
  -> root flow group
     -> child flow groups
     -> task groups
        -> child task groups or ordered step groups
           -> step function / selected branch handler
```

Each runner keeps a `current_output`. After a node or order group finishes,
that value is forwarded to the next group.

## Context Objects

| Context | Available in | Key fields |
|---------|--------------|------------|
| `GlobalContext` | internal runner | global prompt, LLM config, tools, docs, memory |
| `FlowContext` | flow execution | global context, flow metadata, shared data |
| `TaskContext` | task execution | flow context, step results, task tools |
| `StepContext` | step functions and branch handlers | `input`, `previous_results`, tools, LLM helpers |

Inside a step:

```python
@step(order=1, prompt="Normalize input")
async def normalize(ctx):
    raw = ctx.input
    prior = ctx.previous_results
    result = await ctx.call_llm(f"Normalize this: {raw}")
    return {"normalized": result}
```

Branching steps additionally expose:

```python
ctx.condition_value   # value read from condition.field
ctx.selected_branch   # selected branch key, if any
```

`pass_criteria` retries expose:

```python
ctx.pass_criteria_feedback  # feedback from previous failed judge attempts
```

## Schemas

Schemas are optional. When present, FlowForge validates and coerces values at
the boundary:

```python
from pydantic import BaseModel

class Raw(BaseModel):
    text: str

class Clean(BaseModel):
    text: str
    lang: str

@step(order=1, prompt="Clean text", input_schema=Raw, output_schema=Clean)
async def clean(ctx):
    return {"text": ctx.input.text.strip(), "lang": "en"}
```

If a value cannot be validated, execution fails instead of silently forwarding
bad data.

## Sequential And Parallel Groups

```python
@step(order=1, prompt="A")
async def a(ctx): ...

@step(order=1, prompt="B")
async def b(ctx): ...

@step(order=2, prompt="C")
async def c(ctx): ...
```

`a` and `b` receive the same input and run in parallel. `c` runs once after
the order-1 group completes.

The same grouping model applies to sibling flows and sibling tasks with the
same explicit `order`.

Root flows under `@global_config` follow this same model: `order=None` root
flows run sequentially in declaration order, while root flows with the same
explicit order run in parallel and forward the last result in that order group.

## Route Filtering

Use `route` to execute only a subset:

```python
await engine.run(data, route="research")
await engine.run(data, route="research.answer")
await engine.run(data, route=["research", "notify"])
```

FlowForge resolves the route into DAG node IDs, includes necessary ancestors,
and skips unrelated nodes. Invalid route segments raise `ValueError`.

## Task Loops

```python
@task(
    name="refine",
    prompt="Refine until acceptable",
    max_loops=5,
    loop_condition=lambda output: output.get("valid", False),
)
class RefineTask:
    @step(order=1, prompt="Produce candidate")
    async def produce(ctx):
        return {"valid": check(ctx.input), "data": ctx.input}
```

`loop_condition(output) == True` accepts the result. `False` re-runs the task
step chain until `max_loops` is exhausted. The last result is returned even if
the condition never passes.

## Error Handling

By default, step errors propagate:

```python
@task(name="strict", prompt="Stop on error", on_error="raise")
class StrictTask: ...
```

For best-effort pipelines:

```python
@task(name="best_effort", prompt="Return the last successful result", on_error="skip_remaining")
class BestEffortTask: ...
```

`skip_remaining` stops later steps in the task and returns the last successful
step output. If the first step fails, the error still propagates.

## Approval And Resume

```python
@step(order=2, prompt="Needs human approval", approval=True)
async def approve(ctx):
    return ctx.input
```

An approval step raises `ApprovalRequired` before running and stores a
checkpoint on the trace. Resume with:

```python
result = await engine.run(data, resume_from=trace.checkpoint)
```
