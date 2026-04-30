# Branch Dispatching

FlowForge supports branching on `@step`, `@task`, and `@flow` using:

```python
condition=BranchCondition(field="...", enum=[...])
branches={...}
fallback=...
```

There is no separate `@branch` decorator.

## Step Branching

Use step branching when a single task needs to choose one handler.

```python
from flowforge import BranchCondition, FlowForge, global_config, flow, task, step

async def handle_web(ctx):
    return {"source": "web", "query": ctx.input["query"]}

async def handle_db(ctx):
    return {"source": "db", "query": ctx.input["query"]}

@global_config(prompt="Search agent")
class SearchAgent:
    @flow(name="search", prompt="Search for information")
    class SearchFlow:
        @task(name="dispatch", prompt="Choose backend")
        class DispatchTask:
            @step(
                order=1,
                prompt="Route by source",
                condition=BranchCondition(field="source", enum=["web", "db"]),
                branches={"web": handle_web, "db": handle_db},
                fallback=handle_web,
            )
            async def route(ctx):
                ...
```

At runtime:

```python
engine = FlowForge.compile(SearchAgent)
result = await engine.run({"query": "FlowForge", "source": "db"})
```

FlowForge reads `ctx.input["source"]`, selects `handle_db`, and forwards that
handler's return value.

## Task Branching

Use task branching when each branch needs its own step chain.

```python
@task(name="web_task", prompt="Search the web")
class WebTask:
    @step(order=1, prompt="Fetch web results")
    async def fetch(ctx):
        return {"backend": "web", "query": ctx.input["query"]}

@task(name="db_task", prompt="Search the database")
class DBTask:
    @step(order=1, prompt="Fetch DB rows")
    async def fetch(ctx):
        return {"backend": "db", "query": ctx.input["query"]}

@task(
    name="dispatch",
    prompt="Dispatch to a backend task",
    condition=BranchCondition(field="source", enum=["web", "db"]),
    branches={"web": WebTask, "db": DBTask},
    fallback=WebTask,
)
class DispatchTask:
    pass
```

Branch task classes must be decorated with `@task`.

## Flow Branching

Use flow branching when each branch is a whole pipeline.

```python
@flow(name="quick", prompt="Quick answer")
class QuickFlow:
    @task(name="answer", prompt="Short answer")
    class Answer:
        @step(order=1, prompt="Draft")
        async def draft(ctx):
            return {"mode": "quick"}

@flow(name="deep", prompt="Deep research")
class DeepFlow:
    @task(name="research", prompt="Long research")
    class Research:
        @step(order=1, prompt="Research")
        async def research(ctx):
            return {"mode": "deep"}

@flow(
    name="router",
    prompt="Route by mode",
    condition=BranchCondition(field="mode", enum=["quick", "deep"]),
    branches={"quick": QuickFlow, "deep": DeepFlow},
    fallback=QuickFlow,
)
class RouterFlow:
    pass
```

Branch flow classes must be decorated with `@flow`.

## Fallback Behavior

If no branch key matches, FlowForge uses `fallback` when provided. For a
branching step, if no fallback is provided, the decorated function itself is
the last-resort handler.

## Trace Fields

Branch selections appear in the run trace:

```python
result, trace = await engine.run_traced(data)
for node in trace.nodes:
    if node.selected_branch:
        print(node.name, node.condition_value, node.selected_branch)
```
