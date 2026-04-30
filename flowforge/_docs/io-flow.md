# Input / Output Flow

This document focuses on how values move at runtime.

## Flow-Level Chain

```text
engine.run(input_data)
  -> first root flow receives input_data
  -> next root flow receives previous root output
  -> final root output is returned
```

When root flows share the same explicit `order`, they run in parallel and each
receives the same `input_data`.

## Flow Internals

Inside a flow, child flows run before tasks according to their order groups:

```text
flow_input
  -> child flow group(s)
  -> task group(s)
  -> flow_output
```

Nested flows behave like regular flows. They receive the current value and
return a new value.

## Task Internals

A task is either a container, leaf, or branch dispatcher.

```text
container task:
  task_input -> child task groups -> task_output

leaf task:
  task_input -> step order groups -> task_output

branch task:
  task_input -> selected branch task -> task_output
```

## Step Groups

```python
@step(order=1, prompt="First")
async def first(ctx): ...

@step(order=2, prompt="Second")
async def second(ctx): ...
```

`second` receives the return value from `first` as `ctx.input`.

Same-order steps run together:

```python
@step(order=1, prompt="Source A")
async def a(ctx): ...

@step(order=1, prompt="Source B")
async def b(ctx): ...
```

Both receive the same input. The group completes before the next order group.

## Branch Dispatching

```python
@step(
    order=1,
    prompt="Route by file type",
    condition=BranchCondition(field="kind", enum=["csv", "json"]),
    branches={"csv": parse_csv, "json": parse_json},
    fallback=parse_json,
)
async def route(ctx):
    ...
```

FlowForge reads `kind` from the current input and calls the selected handler.
The selected handler's return value is forwarded to the next node.

## Schemas

Schemas validate values at boundaries:

```python
@task(name="parse", prompt="Parse input", input_schema=RawDoc, output_schema=ParsedDoc)
class ParseTask:
    @step(order=1, prompt="Parse", input_schema=RawDoc, output_schema=ParsedDoc)
    async def parse(ctx):
        return ParsedDoc(...)
```

If no schema is provided, FlowForge passes values through as normal Python
objects.

## Routes

```python
await engine.run(data, route="alpha")
await engine.run(data, route="alpha.extract")
await engine.run(data, route=["alpha", "beta"])
```

Routes select explicit DAG subtrees and override planner modes.
