# Nested Flows & Tasks

Flows can contain flows and tasks. Tasks can contain tasks or steps. Use
nesting to keep large agents readable without manual graph wiring.

## Nested Flows

```python
@global_config(prompt="ETL agent")
class PipelineAgent:

    @flow(name="pipeline", prompt="Run the full ETL pipeline")
    class PipelineFlow:

        @flow(name="ingest", prompt="Load raw data")
        class IngestFlow:
            @task(name="fetch", prompt="Fetch data")
            class FetchTask:
                @step(order=1, prompt="Fetch")
                async def fetch(ctx):
                    return {"raw": ctx.input}

        @flow(name="transform", prompt="Clean and shape data")
        class TransformFlow:
            @task(name="clean", prompt="Clean records")
            class CleanTask:
                @step(order=1, prompt="Clean")
                async def clean(ctx):
                    return {"clean": ctx.input}

        @task(name="export", prompt="Export final result")
        class ExportTask:
            @step(order=1, prompt="Export")
            async def export(ctx):
                return {"exported": ctx.input}
```

Execution is insertion-order sequential by default:

```text
ingest -> transform -> export
```

## Container Tasks

```python
@task(name="analyze", prompt="Analyze in stages")
class AnalyzeTask:

    @task(name="extract", prompt="Extract facts")
    class ExtractTask:
        @step(order=1, prompt="Extract")
        async def extract(ctx): ...

    @task(name="score", prompt="Score facts")
    class ScoreTask:
        @step(order=1, prompt="Score")
        async def score(ctx): ...
```

`AnalyzeTask` is a container task. `ExtractTask` and `ScoreTask` are leaf
tasks.

## Explicit Order And Parallel Groups

```python
@task(name="web", prompt="Search web", order=1)
class WebTask: ...

@task(name="docs", prompt="Search docs", order=1)
class DocsTask: ...

@task(name="merge", prompt="Merge results", order=2)
class MergeTask: ...
```

`web` and `docs` run in parallel with the same input. `merge` runs after both.

## `unique=True`

```python
@task(name="canonical_search", prompt="Use this implementation", order=1, unique=True)
class CanonicalSearch: ...
```

If a same-order group includes a `unique=True` node, only that node runs and
its siblings are skipped. Two `unique=True` nodes in the same group raise
`OrderConflictError`.

## `depends_on`

Use `depends_on` for flow dependency edges:

```python
@flow(name="auth", prompt="Authenticate")
class AuthFlow: ...

@flow(name="fetch", prompt="Fetch data", depends_on=["auth"])
class FetchFlow: ...
```

`depends_on` participates in DAG cycle detection and topological ordering.
