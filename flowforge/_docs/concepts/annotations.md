# Annotations In Depth

FlowForge has four public decorators:

| Decorator | Applies to | Purpose |
|-----------|------------|---------|
| `@global_config` | class | Agent root: global prompt, model, tools, dynamic settings |
| `@flow` | class | Pipeline stage; can contain flows and tasks |
| `@task` | class | Work unit; can contain child tasks or steps |
| `@step` | async function | Atomic executable action |

!!! warning "No `@branch` Decorator"
    Branching is implemented by adding `condition`, `branches`, and optional
    `fallback` to `@step`, `@task`, or `@flow`. Do not import or use
    `@branch`.

## Hierarchy

```text
@global_config
  └─ @flow
       ├─ @flow
       └─ @task
            ├─ @task
            └─ @step
```

| Parent | Allowed direct children |
|--------|-------------------------|
| `@global_config` | `@flow` |
| `@flow` | `@flow`, `@task` |
| container `@task` | `@task` |
| leaf `@task` | `@step` |

A task should be either a container task with child tasks or a leaf task with
steps. Branch tasks and branch flows are dispatchers; their class bodies are
ignored at runtime.

## `@global_config`

```python
from flowforge import LLMConfig, global_config
from flowforge.types import FunctionTool

@global_config(
    prompt="You are a data-processing assistant.",
    llm_config=LLMConfig.for_claude(temperature=0.2),
    tools=[FunctionTool(func=my_search, name="search")],
    dynamic_flow=False,
    include_builtin_tools=False,
)
class MyAgent:
    ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | Global system prompt for LLM calls |
| `llm_config` | `LLMConfig()` | Default provider, model, temperature, tokens |
| `tools` | `[]` | Global MCP, HTTP, function, Claude Skill, or Agent Skill tools |
| `dynamic_flow` | `False` | Allows runtime generation and injection of missing flows |
| `include_builtin_tools` | `False` | Adds FlowForge builtin tools to the global tool list |

## `@flow`

```python
@flow(
    name="research",
    prompt="Research a topic and produce source-backed notes",
    input_schema=Query,
    output_schema=Notes,
    depends_on=["auth"],
    order=1,
    unique=False,
    tools=["search"],
)
class ResearchFlow:
    ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | required | Unique name within the parent scope |
| `prompt` | required | Natural-language role description |
| `input_schema` / `output_schema` | `None` | Optional Pydantic boundary validation |
| `depends_on` | `[]` | Flow names or IDs that must run first |
| `parallel` | `False` | Legacy flag to run immediate child flows concurrently |
| `max_retries` | `3` | Retries on execution errors |
| `order` | `None` | Execution slot within the parent; same slot runs in parallel |
| `unique` | `False` | Exclusive runner for its order group |
| `tools` | `[]` | Tool configs or names available to descendants |
| `condition` / `branches` / `fallback` | `None` | Turn the flow into a branch dispatcher |

## `@task`

```python
@task(
    name="draft",
    prompt="Draft an answer",
    order=2,
    on_error="raise",
    max_loops=3,
    loop_condition=lambda out: out.get("quality", 0) >= 0.8,
)
class DraftTask:
    ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | required | Unique name within the parent flow/task |
| `prompt` | required | Task role description |
| `input_schema` / `output_schema` | `None` | Optional Pydantic boundary validation |
| `order` | `None` | Execution slot within the parent |
| `unique` | `False` | Exclusive runner for same-order siblings |
| `tools` | `[]` | Tool configs or names available to child tasks/steps |
| `on_error` | `"raise"` | `"raise"` or `"skip_remaining"` |
| `max_loops` | `1` | Maximum attempts for the task step chain |
| `loop_condition` | `None` | `(output) -> bool`; `True` accepts the result |
| `condition` / `branches` / `fallback` | `None` | Turn the task into a branch dispatcher |

## `@step`

```python
@step(
    order=1,
    prompt="Fetch search results",
    timeout_seconds=30,
    approval=False,
    pass_criteria="The output must include at least one citation.",
    pass_criteria_max_retries=2,
    tools=["search"],
)
async def fetch(ctx):
    return await ctx.call_tool("search", query=ctx.input["query"])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `order` | required | Execution slot inside a leaf task |
| `prompt` | required | Step instruction |
| `input_schema` / `output_schema` | `None` | Optional Pydantic validation |
| `tool_mode` | `False` | Metadata flag for tool-oriented steps; ordered execution still applies |
| `timeout_seconds` | `60` | Per-step timeout |
| `unique` | `False` | Exclusive runner for same-order step group |
| `approval` | `False` | Pause before running and raise `ApprovalRequired` |
| `tools` | `[]` | Tool configs or names available to this step |
| `condition` / `branches` / `fallback` | `None` | Turn the step into a branch dispatcher |
| `pass_criteria` | `None` | LLM-judged acceptance criteria |
| `pass_criteria_max_retries` | `3` | Retry attempts when criteria fail |

## Order, Parallelism, And `unique`

`order=None` preserves insertion-order sequential execution. Explicit order
values create execution groups:

```python
@step(order=1, prompt="Search web")
async def web(ctx): ...

@step(order=1, prompt="Search docs")
async def docs(ctx): ...

@step(order=2, prompt="Merge results")
async def merge(ctx): ...
```

`web` and `docs` run in parallel and receive the same input. The next group
receives the final result forwarded by the group.

Use `unique=True` when exactly one same-order node should run:

```python
@step(order=1, prompt="Canonical implementation", unique=True)
async def canonical(ctx): ...
```

Only one node per same-order group may set `unique=True`; duplicates raise
`OrderConflictError`.

## Branch Dispatching

```python
from flowforge import BranchCondition

async def web_handler(ctx):
    return {"source": "web", "text": ctx.input["query"]}

async def db_handler(ctx):
    return {"source": "db", "text": ctx.input["query"]}

@step(
    order=1,
    prompt="Route by source",
    condition=BranchCondition(field="source", enum=["web", "db"]),
    branches={"web": web_handler, "db": db_handler},
    fallback=web_handler,
)
async def route(ctx):
    ...
```

At runtime FlowForge reads `condition.field` from `ctx.input`, selects a
handler, sets `ctx.condition_value` and `ctx.selected_branch`, and returns the
handler output.
