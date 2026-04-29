# Annotations In Depth

---

## Hierarchy Rules

```
@global_config
  └─ @flow                 ← 1+ flows required (supports branch dispatching)
       ├─ @flow            ← flows nest recursively
       └─ @task            ← leaf tasks hold steps (supports branch dispatching)
            ├─ @task       ← container tasks hold child tasks
            └─ @step       ← leaf only (supports branch dispatching)
```

| Parent | Allowed Children |
|--------|-----------------|
| `@global_config` | `@flow` |
| `@flow` | `@flow`, `@task` |
| `@task` (container) | `@task` |
| `@task` (leaf) | `@step` |
| `@step` | — (leaf) |

!!! note "No `@branch` Decorator"
    Branch dispatching is built into `@step`, `@task`, and `@flow` via the optional `condition`, `branches`, and `fallback` parameters. There is no separate `@branch` decorator.

---

## @global_config

Top-level agent configuration. Every agent has exactly one.

```python
from flowforge import global_config
from flowforge.types import LLMConfig, MCPServer

@global_config(
    prompt="You are a data processing specialist. Always respond in English.",
    llm_config=LLMConfig(
        model="claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=4096,
    ),
    tools=[MCPServer("https://api.example.com/mcp")],
)
class MyAgent: ...
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | `str` | ✅ | Global system prompt passed to every LLM call |
| `llm_config` | `LLMConfig` | | Default LLM settings for all nodes |
| `tools` | `list[ToolConfig]` | | MCP servers, Python functions, HTTP APIs, Claude Skills, local Agent Skills |

---

## @flow

High-level pipeline stage. Flows can nest and depend on each other.

```python
from flowforge import flow
from pydantic import BaseModel

class UserQuery(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    summary: str
    keywords: list[str]

@flow(
    name="analyze",
    prompt="Analyze the user query and extract structured information",
    input_schema=UserQuery,
    output_schema=AnalysisResult,
    depends_on=["auth"],        # run after 'auth' flow
    parallel=False,
    max_retries=2,
)
class AnalyzeFlow:
    ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | ✅ | Unique identifier within the agent |
| `prompt` | `str` | ✅ | Role description (used at runtime by LLM) |
| `input_schema` | `Type[BaseModel]` | `None` | Pydantic model for input validation |
| `output_schema` | `Type[BaseModel]` | `None` | Pydantic model for output validation |
| `depends_on` | `list[str]` | `[]` | Flow names that must complete first |
| `parallel` | `bool` | `False` | Run child nodes in parallel |
| `max_retries` | `int` | `3` | Retry count on failure |

### Nested Flows

```python
@flow(name="pipeline", prompt="Full data pipeline")
class PipelineFlow:

    @flow(name="ingest", prompt="Ingest raw data")   # child flow
    class IngestFlow:
        @task(name="fetch", prompt="Fetch from source")
        class FetchTask: ...

    @flow(name="transform", prompt="Transform data")  # runs after ingest
    class TransformFlow:
        @task(name="clean", prompt="Clean data")
        class CleanTask: ...
```

---

## @task

Execution unit within a flow. Can be a **container** (holds child tasks) or a **leaf** (holds steps/branches directly).

```python
from flowforge import task

@task(
    name="process_document",
    prompt="Parse and analyze a document",
    input_schema=RawDoc,
    output_schema=ProcessedDoc,
)
class ProcessDocumentTask:
    # Leaf task — contains steps/branches directly
    @step(order=1, prompt="Detect document format")
    async def detect_format(ctx): ...

    @branch(order=2, ...)
    async def route_parser(ctx): ...

    @step(order=3, prompt="Normalize output")
    async def normalize(ctx): ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | ✅ | Unique identifier within parent |
| `prompt` | `str` | ✅ | Task role description |
| `input_schema` | `Type[BaseModel]` | `None` | Input Pydantic model |
| `output_schema` | `Type[BaseModel]` | `None` | Output Pydantic model |

### Container Task Pattern

```python
@task(name="analyze_and_format", prompt="Analyze and format")
class AnalyzeAndFormatTask:

    @task(name="analyze", prompt="Analyze query")   # child task 1
    class AnalyzeTask:
        @step(order=1, prompt="Classify intent")
        async def classify(ctx): ...

    @task(name="format", prompt="Format result")    # child task 2
    class FormatTask:
        @step(order=1, prompt="Draft answer")
        async def draft(ctx): ...
        @step(order=2, prompt="Add citations")
        async def cite(ctx): ...
```

---

## @step

A single action within a leaf task. Steps execute in `order` sequence.

```python
from flowforge import step
from pydantic import BaseModel

class RawDoc(BaseModel):
    content: str

class ValidatedDoc(BaseModel):
    content: str
    format: str

@step(
    order=1,
    prompt="Validate the document schema and detect its format",
    input_schema=RawDoc,
    output_schema=ValidatedDoc,
    timeout_seconds=30,
)
async def validate_doc(ctx):
    # ctx.input  → RawDoc instance
    # ctx.tools  → ToolAccessor
    # ctx.previous_results → {order: result} dict
    doc = ctx.input
    fmt = "json" if doc.content.startswith("{") else "text"
    return ValidatedDoc(content=doc.content, format=fmt)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | `int` | ✅ | Execution order within task. **Must be unique per task.** |
| `prompt` | `str` | ✅ | What this step does |
| `input_schema` | `Type[BaseModel]` | `None` | Validates input before calling function |
| `output_schema` | `Type[BaseModel]` | `None` | Validates return value |
| `tool_mode` | `bool` | `False` | Register as LLM tool instead of sequential step |
| `timeout_seconds` | `int` | `60` | Execution timeout |

!!! warning "Order Must Be Unique"
    Within a single leaf task, every `@step` and `@branch` must have a **distinct** `order`. Steps and branches share the same order space.

    ```python
    @task(name="process")
    class ProcessTask:
        @step(order=1, ...)  # ✅
        async def a(ctx): ...

        @branch(order=2, ...)  # ✅ (branch also uses order)
        async def b(ctx): ...

        @step(order=3, ...)  # ✅
        async def c(ctx): ...

        @step(order=3, ...)  # ❌ OrderConflictError at import time!
        async def d(ctx): ...
    ```

---

## Branch Dispatching

Conditional routing is built into `@step`, `@task`, and `@flow` — there is no separate `@branch` decorator. Add `condition`, `branches`, and optionally `fallback` to any decorator.

### Step-level branching

```python
from flowforge import step
from flowforge.types import BranchCondition

async def handle_web(ctx): ...
async def handle_db(ctx): ...

@step(
    order=2,
    prompt="Route to the appropriate data source",
    condition=BranchCondition(field="source", enum=["web", "db"]),
    branches={"web": handle_web, "db": handle_db},
    fallback=handle_web,
)
async def route_source(ctx): ...
```

### Task-level branching

```python
@task(
    name="dispatch",
    prompt="Route to fast or slow processing",
    condition=BranchCondition(field="mode", enum=["fast", "slow"]),
    branches={"fast": FastTask, "slow": SlowTask},
)
class DispatchTask: ...
```

### Flow-level branching

```python
@flow(
    name="dispatch",
    prompt="Route by request type",
    condition=BranchCondition(field="type", enum=["a", "b"]),
    branches={"a": FlowA, "b": FlowB},
)
class DispatchFlow: ...
```

### Branch parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | `BranchCondition` | `None` | `field` to inspect + valid `enum` values |
| `branches` | `dict[str, Callable]` | `None` | Value → handler/class mapping |
| `fallback` | `Callable` | `None` | Handler when no value matches |

!!! tip "Handler Context"
    Branch handlers receive a `StepContext` with `selected_branch` and `condition_value` populated:
    ```python
    async def handle_web(ctx):
        # ctx.input           → same as step input
        # ctx.condition_value  → the resolved field value
        # ctx.selected_branch  → "web"
        return SearchResult(source="web", results=[...])
    ```

!!! warning "Output Type Consistency"
    All branch handlers **must return the same type** (or `None`).
    The output of the selected handler becomes the input to the next `order` node.
