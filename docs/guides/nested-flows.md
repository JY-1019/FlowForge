# Nested Flows & Tasks

FlowForge supports recursive nesting: flows can contain flows, tasks can contain tasks.

---

## Nested Flows

Use nested flows to split a complex pipeline into logical sub-pipelines:

```python
@global_config(prompt="Data pipeline agent")
class PipelineAgent:

    @flow(name="pipeline", prompt="Full ETL pipeline")
    class PipelineFlow:

        # Sub-flow 1: data ingestion
        @flow(name="ingest", prompt="Ingest raw data from external sources")
        class IngestFlow:

            @task(name="fetch", prompt="Fetch raw data")
            class FetchTask:
                @step(order=1, prompt="Validate source URL")
                async def validate_source(ctx): ...

                @step(order=2, prompt="Download raw data")
                async def download(ctx): ...

        # Sub-flow 2: transformation (runs AFTER ingest output)
        @flow(name="transform", prompt="Clean and transform ingested data")
        class TransformFlow:

            @task(name="clean", prompt="Remove nulls and duplicates")
            class CleanTask:
                @step(order=1, prompt="Drop empty rows")
                async def drop_nulls(ctx): ...

                @step(order=2, prompt="Deduplicate records")
                async def deduplicate(ctx): ...

        # Final task (runs after both sub-flows)
        @task(name="export", prompt="Export to destination")
        class ExportTask:
            @step(order=1, prompt="Write output")
            async def write_output(ctx): ...
```

**Execution order:**
1. `IngestFlow` runs (receives original input)
2. `TransformFlow` runs (receives `IngestFlow` output)
3. `ExportTask` runs (receives `TransformFlow` output)

---

## `depends_on` — Explicit Ordering

For flows at the same level under `@global_config`, use `depends_on` to enforce order:

```python
@global_config(prompt="agent")
class MyAgent:

    @flow(name="auth", prompt="Authenticate the user")
    class AuthFlow: ...

    @flow(name="fetch", prompt="Fetch data", depends_on=["auth"])
    class FetchFlow: ...

    @flow(name="process", prompt="Process data", depends_on=["fetch"])
    class ProcessFlow: ...
```

DAG edges: `auth → fetch → process`

---

## Parallel Flows

```python
@flow(name="multi_search", prompt="Search multiple sources in parallel",
      parallel=True)
class MultiSearchFlow:

    @flow(name="web", prompt="Web search")
    class WebSearchFlow: ...

    @flow(name="db", prompt="Database search")
    class DbSearchFlow: ...

    @flow(name="api", prompt="API search")
    class ApiSearchFlow: ...
    # All three receive the same input; last result is the output
```

---

## Nested Tasks (Container Tasks)

A task that contains other tasks is called a **container task**.
It runs child tasks sequentially, passing each output to the next.

```python
@task(name="analyze_and_format", prompt="Analyze and format the document")
class AnalyzeAndFormatTask:

    @task(name="analyze", prompt="Analyze content and extract structure")
    class AnalyzeTask:          # leaf task — has steps directly
        @step(order=1, prompt="Classify document type")
        async def classify(ctx): ...

        @step(order=2, prompt="Extract key entities")
        async def extract(ctx): ...

    @task(name="format", prompt="Format the analysis into a report")
    class FormatTask:           # leaf task — has steps directly
        @step(order=1, prompt="Draft summary paragraph")
        async def draft(ctx): ...

        @step(order=2, prompt="Add citations and references")
        async def cite(ctx): ...
```

**Data flow:** `AnalyzeTask output → FormatTask input`

---

## Deep Nesting Example

```
PipelineAgent (global)
└─ pipeline (flow)
     ├─ ingest (flow)
     │    └─ fetch (task, leaf)
     │         ├─ validate_source (step, order=1)
     │         └─ download (step, order=2)
     ├─ transform (flow)
     │    └─ clean (task, leaf)
     │         ├─ drop_nulls (step, order=1)
     │         └─ deduplicate (step, order=2)
     └─ export (task, leaf)
          └─ write_output (step, order=1)
```

Resulting node IDs:
```
global
global.pipeline
global.pipeline.ingest
global.pipeline.ingest.fetch
global.pipeline.ingest.fetch.validate_source[1]
global.pipeline.ingest.fetch.download[2]
global.pipeline.transform
global.pipeline.transform.clean
global.pipeline.transform.clean.drop_nulls[1]
global.pipeline.transform.clean.deduplicate[2]
global.pipeline.export
global.pipeline.export.write_output[1]
```

---

## Best Practices

!!! tip "When to use nested flows vs nested tasks"
    - Use **nested flows** when the sub-pipeline represents a distinct, reusable capability (e.g., "search", "auth", "ingest").
    - Use **nested tasks** when the sub-steps are tightly coupled and share the same data context.

!!! tip "Keep leaf tasks focused"
    A leaf task should do one thing (validate, parse, summarize, etc.) with 2–5 steps.
    If you find yourself with 8+ steps in a task, consider splitting into child tasks or flows.

!!! warning "Parallel flows share the same input"
    All parallel child flows receive the **same input** (the parent's current output).
    They do not receive each other's output. Only the last result is forwarded.
