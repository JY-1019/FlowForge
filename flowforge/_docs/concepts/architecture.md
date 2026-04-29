# Architecture Overview

FlowForge has four phases: **Decorate → Compile → Plan → Execute**.
When `dynamic_flow=True`, a fifth phase — **Dynamic Generation** — can insert itself between Plan and Execute.

---

## Phase 1 — Decorate

Python processes inner class decorators before outer ones.
This is the foundational mechanism that makes FlowForge work.

```python
@global_config(...)         # 4. runs last — finds FlowMeta
class MyAgent:

    @flow(...)              # 3. runs third — finds TaskMeta
    class MyFlow:

        @task(...)          # 2. runs second — finds StepMeta
        class MyTask:

            @step(order=1)  # 1. runs first
            async def do_something(ctx): ...
```

Each decorator attaches a metadata object to the class/function:

| Decorator | Metadata attached | Attribute name |
|-----------|------------------|---------------|
| `@step` | `StepMeta` | `__flowforge_step_meta__` |
| `@task` | `TaskMeta` | `__flowforge_task_meta__` |
| `@flow` | `FlowMeta` | `__flowforge_flow_meta__` |
| `@global_config` | `GlobalMeta` | `__flowforge_global_meta__` |

Branch dispatching is not a separate decorator. It is a parameter of `@step`, `@task`, and `@flow` via `condition`, `branches`, and `fallback`.

Validation also runs here — `OrderConflictError` is raised at import time if two steps share the same `order`.

---

## Phase 2 — Compile

`FlowForge.compile(MyAgent)` traverses the metadata tree depth-first and emits `DAGNode` + `DAGEdge` objects into a `networkx.DiGraph`.

```
GlobalMeta
  └─ FlowMeta(research)
       ├─ FlowMeta(search)           ← child flow
       │    └─ TaskMeta(execute)
       │         ├─ StepMeta(order=1)
       │         ├─ BranchMeta(order=2)
       │         └─ StepMeta(order=3)
       └─ TaskMeta(analyze_format)
            ├─ TaskMeta(analyze)     ← container task → leaf task
            └─ TaskMeta(format)
```

**Node ID scheme** (dotted path):

```
global
global.research
global.research.search
global.research.search.execute
global.research.search.execute.optimize_query[1]
global.research.search.execute.source_select[2]
global.research.search.execute.deduplicate[3]
```

Two edge types exist:
- `parent_child` — containment (flow contains task, task contains step)
- `depends_on` — cross-flow sequencing (`@flow(depends_on=["other_flow"])`)

---

## Phase 3 — Plan *(optional)*

When AI-based path selection is enabled, the `LLMPlanner` receives a structured prompt built from the DAG + `doc` metadata for each node. It returns an `ExecutionPlan` — the subset of nodes to execute for the current input.

This phase is **optional**. By default the engine executes the full DAG.

---

## Phase 4 — Execute

The `ExecutionEngine` drives a `FlowRunner`, which drives `TaskRunner`, which drives `StepRunner`. Branch dispatching is handled within each runner when `condition`/`branches` parameters are present.

```
ExecutionEngine.run(input)
  └─ FlowRunner.run(FlowMeta, global_ctx, input)
       ├─ [child flows run first, sequentially or in parallel]
       └─ TaskRunner.run(TaskMeta, flow_ctx, task_input)
            ├─ [container: recurse into child tasks]
            └─ [leaf: iterate steps in order, thread output→input]
                 └─ StepRunner.run(StepMeta, task_ctx, step_input)
                      └─ [if step has condition/branches: dispatch to handler]
```

After every run, a `RunTrace` is stored in `engine.last_trace`.

---

## Phase 4.5 — Dynamic Generation *(optional)*

When `@global_config(dynamic_flow=True)` is set and the planner reports a gap (missing capability), the engine triggers the built-in `_dynamic_generator` meta-flow:

```
Planner reports gap_detected / uncovered requirements
  → Engine triggers _dynamic_generator (per requirement)
  → Meta-flow 3-step pipeline:
      [1] analyse_gap      — verify the gap is real
      [2] prepare_codegen  — build implementation brief
      [3] generate_and_inject — LLM codegen → AST safety → compile → persist → inject
  → Engine replans with the new flow(s) in the DAG
  → Normal execution continues
```

Generated code passes through AST safety validation (blocks `os.system`,
`subprocess`, `eval`, etc.) before execution. When `persist_generated=True`,
flows are saved to disk and registered in `manifest.json` with file-lock
protection. On later compiles, `auto_load_generated=True` loads those records
back into the DAG; before new generation, FlowForge checks both the current DAG
and the manifest so existing generated flows are not recreated.

See [Dynamic Flow Generation Guide](../guides/dynamic-flow.md) for details.

---

## Component Map

```
flowforge/
├── annotations/      ← decorators + metadata dataclasses + validators
├── schema/           ← DAG node/edge models, compiler (metadata→DAG), resolver
├── dynamic/          ← dynamic flow generation (generator, meta-flow, manifest)
├── execution/        ← context hierarchy, runners, engine, LLM caller
├── doc/              ← AI doc generation, cache
├── planner/          ← path selection (Deterministic/Autonomous/Hybrid)
├── tools/            ← MCP, function, HTTP adapters + builtin tool pack
├── viz/              ← graphviz/mermaid renderer, run_trace, subtree renderer
└── cli/              ← typer CLI entry point
```

---

## Design Patterns Used

| Component | Pattern |
|-----------|---------|
| Flow/Task hierarchy | Composite |
| Step chain | Chain of Responsibility |
| Branch routing | Strategy + Discriminated Union |
| Doc generation | Template Method |
| Tool integration | Adapter + Plugin |
| Context passing | Context Object |
| DAG building | Builder |
| Execution engine | Visitor |
