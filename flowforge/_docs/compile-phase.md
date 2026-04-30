# Compile Phase

This page follows one agent from Python decoration to a compiled
`CompiledAgent`.

## Overview

```text
Python imports module
  -> @step attaches StepMeta
  -> @task collects StepMeta and child TaskMeta
  -> @flow collects FlowMeta and TaskMeta
  -> @global_config collects root FlowMeta
  -> FlowForge.compile() builds and validates the DAG
```

## Decorator Processing Order

```python
@global_config(prompt="agent")
class Agent:
    @flow(name="research", prompt="Research")
    class Research:
        @task(name="answer", prompt="Answer")
        class Answer:
            @step(order=1, prompt="Draft")
            async def draft(ctx):
                return ctx.input
```

Python applies the decorators in this order:

1. `@step` attaches `StepMeta` to `draft`.
2. `@task` scans `Answer.__dict__` and collects `draft`.
3. `@flow` scans `Research.__dict__` and collects `Answer`.
4. `@global_config` scans `Agent.__dict__` and collects `Research`.

## Branching During Decoration

There is no standalone `@branch` decorator and no `BranchMeta`.

```python
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

This is stored as a normal `StepMeta` with `condition`, `branches`, and
`fallback`. The same model applies to branching `@task` and `@flow`.

## Metadata Tree

After decoration, the agent holds an in-memory tree:

```text
GlobalMeta(prompt, llm_config, tools, flows)
  -> FlowMeta(name, child_flows, tasks, order, condition?)
     -> TaskMeta(name, child_tasks, steps, order, loop_condition?)
        -> StepMeta(order, func, tools, condition?, pass_criteria?)
```

## DAG Build

`FlowForge.compile()` reads `GlobalMeta`, emits `DAGNode` objects, creates
containment edges, adds `depends_on` edges, and resolves topological order.

```python
engine = FlowForge.compile(Agent)
print(engine.mermaid())
```

The compiler does not execute user steps. It only reads metadata.

## Dynamic Compile Additions

Dynamic generation can extend the DAG after initial compile:

```python
engine.add_flow(GeneratedFlow)
engine.replace_flow(RegeneratedFlow)
engine.recompile()
```

`add_flow()` and `replace_flow()` update both the live DAG and global metadata
so later runs use the expanded structure.
