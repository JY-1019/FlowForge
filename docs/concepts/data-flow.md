# Data Flow & I/O

Understanding how data moves between annotations is essential to building correct FlowForge agents.

---

## The Golden Rule

> **Every node's return value becomes the next node's input.**

This applies at every level: flow → flow, task → task, step → step, branch → step.

---

## Context Hierarchy

Each annotation level gets its own context object, which holds a reference to its parent.

```
GlobalContext
  llm_config, global_prompt, tool_registry, tracer
  └─ FlowContext
       flow_name, flow_prompt, flow_input, parent_flow_output
       └─ TaskContext
            task_name, task_prompt, task_input
            step_results: {order: result}
            └─ StepContext / BranchContext
                 input         ← previous node's output
                 tools         ← ToolAccessor
                 previous_results ← TaskContext.step_results
```

In your step/branch functions, `ctx` is always a `StepContext` or `BranchContext`:

```python
@step(order=2, prompt="Enrich the document")
async def enrich(ctx):
    doc = ctx.input                         # output of order=1
    prev = ctx.previous_results             # {1: <order-1 output>}
    result1 = ctx.previous_results.get(1)   # order=1 result directly
    return EnrichedDoc(...)
```

---

## Flow-Level I/O

Flows at the same level run **sequentially** by default, passing output forward.

```
engine.run(input_data)
  │
  ├─ FlowA.run(input_data)  → output_A
  └─ FlowB.run(output_A)    → output_B   ← final result
```

Child flows within a parent flow:

```python
@flow(name="pipeline", prompt="Full pipeline")
class PipelineFlow:

    @flow(name="fetch", prompt="Fetch data")       # runs first
    class FetchFlow: ...

    @flow(name="process", prompt="Process data")    # receives FetchFlow output
    class ProcessFlow: ...

    @task(name="finalize", prompt="Finalize")       # receives ProcessFlow output
    class FinalizeTask: ...
```

### Parallel Flows

```python
@flow(name="parallel_search", prompt="Search multiple sources", parallel=True)
class ParallelSearchFlow:

    @flow(name="web_search", prompt="Search the web")
    class WebSearchFlow: ...

    @flow(name="db_search", prompt="Search the database")
    class DbSearchFlow: ...
    # Both receive the same input; last result is the output
```

### `depends_on` — Cross-Flow Ordering

```python
@global_config(prompt="agent")
class MyAgent:

    @flow(name="auth", prompt="Authenticate")
    class AuthFlow: ...

    @flow(name="fetch", prompt="Fetch data", depends_on=["auth"])
    class FetchFlow: ...   # guaranteed to run AFTER AuthFlow
```

---

## Task-Level I/O

Within a flow, tasks run sequentially. Each task's output is the next task's input.

```
FlowContext input
  │
  ├─ TaskA.run(input)    → output_A
  ├─ TaskB.run(output_A) → output_B
  └─ TaskC.run(output_B) → output_C   ← flow output
```

**Container tasks** pass through to their children:

```python
@task(name="analyze_format", prompt="...")
class AnalyzeFormatTask:

    @task(name="analyze", prompt="Analyze")   # receives flow input
    class AnalyzeTask: ...

    @task(name="format", prompt="Format")     # receives AnalyzeTask output
    class FormatTask: ...
```

---

## Step-Level I/O

Within a leaf task, steps form a **chain**: each step's return value is the next step's `ctx.input`.

```
task_input
  │
  ├─ step(order=1) → out1    validate_input(ctx.input = task_input)
  ├─ step(order=2) → out2    enrich(ctx.input = out1)
  ├─ branch(order=3) → out3  route(ctx.input = out2) → selected handler
  └─ step(order=4) → out4    save(ctx.input = out3)  ← task output
```

```python
@task(name="process", prompt="Process a document")
class ProcessTask:

    @step(order=1, prompt="Validate input")
    async def validate(ctx):
        raw: RawDoc = ctx.input      # ← task_input flows in here
        return ValidatedDoc(...)

    @step(order=2, prompt="Enrich metadata")
    async def enrich(ctx):
        doc: ValidatedDoc = ctx.input   # ← output of order=1
        return EnrichedDoc(...)

    @branch(order=3, name="router", prompt="Route by format",
            condition=BranchCondition(field="format", enum=["json","csv"]),
            branches={"json": handle_json, "csv": handle_csv},
            fallback=handle_text)
    async def route(ctx): ...

    @step(order=4, prompt="Save result")
    async def save(ctx):
        result = ctx.input   # ← output of whichever branch handler ran
        return SaveResult(...)
```

---

## Schema Validation

Input and output validation happens **at execution time**, not just at compile time.

```python
@step(
    order=1,
    prompt="...",
    input_schema=RawDoc,      # validated BEFORE calling the function
    output_schema=ValidDoc,   # validated AFTER the function returns
)
async def validate(ctx):
    doc: RawDoc = ctx.input   # guaranteed to be a valid RawDoc
    return ValidDoc(...)      # coerced through output_schema.model_validate()
```

If no `input_schema` is specified, the raw value from the previous step flows through unchanged.

---

## Branch I/O

The condition field is looked up on the branch's input object:

```python
class AnalyzedQuery(BaseModel):
    intent: str
    source_preference: str   # "web" | "db" | "api"

@branch(
    order=2,
    name="source_router",
    condition=BranchCondition(field="source_preference", enum=["web","db","api"]),
    branches={"web": web_fn, "db": db_fn, "api": api_fn},
    fallback=web_fn,
)
async def route(ctx):
    # ctx.input.source_preference == "web"
    # → web_fn(ctx) is called
    # → its return value becomes input for order=3
    ...
```

When the condition value does not match any branch key:
- If `fallback` is set → fallback handler runs, `selected_branch = "__fallback__"`
- If no `fallback` → the branch's own body runs

---

## Full Data Flow Diagram

```
engine.run(UserQuery)
│
└─ FlowRunner(research_flow)
     │  flow_input = UserQuery
     │
     ├─ [child FlowRunner(search_flow)]
     │    │  flow_input = UserQuery
     │    │
     │    └─ TaskRunner(execute_search)
     │         │  task_input = UserQuery
     │         │
     │         ├─ StepRunner(optimize_query[1])
     │         │    ctx.input = UserQuery → returns OptimizedQuery
     │         │
     │         ├─ BranchRunner(source_select[2])
     │         │    ctx.input = OptimizedQuery
     │         │    value = ctx.input.source_preference → "web"
     │         │    → web_handler(ctx) returns SearchResult
     │         │
     │         └─ StepRunner(deduplicate[3])
     │              ctx.input = SearchResult → returns CleanResult
     │
     └─ TaskRunner(analyze_format)
          task_input = CleanResult (search_flow output)
          │
          ├─ TaskRunner(analyze)   task_input = CleanResult
          │    └─ StepRunner(classify[1]) → AnalyzedQuery
          │
          └─ TaskRunner(format)    task_input = AnalyzedQuery
               ├─ StepRunner(draft[1])    → DraftAnswer
               └─ StepRunner(cite[2])     → FormattedAnswer
```
