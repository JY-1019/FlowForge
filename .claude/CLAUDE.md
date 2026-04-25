# FlowForge — Design Specification v1.1

> Annotation-Based AI Agent Framework  
> Version 1.1 | April 2026

---

## Claude Code Working Notes

Read this section first. It captures project-specific details that are easy to
miss when editing the codebase.

### Project State

- Python package in this repository.
- Current verification target: `python -m pytest tests/ -x -q`.
- Public API: `@global_config`, `@flow`, `@task`, `@step`,
  `FlowForge.compile()`.
- There is no public `@branch` decorator. Branch dispatch is configured on
  `@step`, `@task`, or `@flow` through `condition`, `branches`, and
  `fallback`.
- Dynamic flow generation is enabled with
  `@global_config(dynamic_flow=True)` and controlled by `DynamicRunOptions`.

### Key Files

| File | Role |
|------|------|
| `flowforge/annotations/decorators.py` | Decorator entry points |
| `flowforge/annotations/metadata.py` | `StepMeta`, `TaskMeta`, `FlowMeta`, `GlobalMeta` |
| `flowforge/annotations/validators.py` | Compile-time validators |
| `flowforge/execution/runner.py` | Step, task, and flow runners |
| `flowforge/execution/context.py` | Context hierarchy and `ctx.call_llm()` / `ctx.call_tool()` access |
| `flowforge/execution/llm.py` | Provider calls, tool-use loop, Claude Skills, Agent Skills |
| `flowforge/schema/compiler.py` | Annotation metadata to DAG compilation |
| `flowforge/schema/dag.py` | `FlowForgeDAG`, `DAGNode`, `NodeType` |
| `flowforge/types.py` | `LLMConfig`, tool types, `DynamicRunOptions`, `DependencyPolicy` |
| `flowforge/dynamic/generator.py` | Dynamic code generation and AST safety |
| `flowforge/dynamic/meta_flow.py` | Built-in `_dynamic_generator` meta-flow |
| `flowforge/dynamic/manifest.py` | Generated flow/tool manifest persistence |
| `flowforge/tools/builtin.py` | Built-in utility, file, shell, and artifact tools |
| `flowforge/planner/llm_planner.py` | Requirement decomposition and gap detection |
| `tests/test_tools_and_llm.py` | Tool inheritance, LLM calls, Skills tests |
| `tests/test_dynamic_flow.py` | Dynamic flow safety, manifest, contract, tools |

### Test Scoping Rule

Agent classes used by async tests should be defined at module scope. Python
class bodies cannot reference variables from an enclosing function scope.

```python
# Bad: MyFlow cannot see the local MyTask binding inside its class body.
async def test_foo():
    @task(name="t", prompt="t")
    class MyTask:
        ...

    @flow(name="f", prompt="f")
    class MyFlow:
        MyTask = MyTask


# Good: both classes are module-level names.
@task(name="t", prompt="t")
class MyTask:
    ...


@flow(name="f", prompt="f")
class MyFlow:
    MyTask = MyTask
```

### Decorator Processing Order

Python applies inner decorators before outer decorators:

```text
@step          -> attaches StepMeta to the function
@task          -> scans class attributes for steps/tasks
@flow          -> scans class attributes for flows/tasks
@global_config -> scans class attributes for root flows
FlowForge.compile() -> builds and validates the DAG
```

### Order, Parallelism, And Uniqueness

| Scenario | Behavior |
|----------|----------|
| `order=None` | Auto-sequential insertion order |
| Same explicit `order` among siblings | Parallel execution group |
| One `unique=True` in a same-order group | Only that node runs |
| Multiple `unique=True` at the same order | `OrderConflictError` |

### Branch Dispatch

```python
@step(
    order=2,
    prompt="Route by source",
    condition=BranchCondition(field="source", enum=["web", "db"]),
    branches={"web": web_handler, "db": db_handler},
    fallback=web_handler,
)
async def route(ctx):
    ...
```

Task-level branches point to task classes. Flow-level branches point to flow
classes. `StepContext.selected_branch` and `StepContext.condition_value` are
populated before a branch handler is called.

### Route Execution

```python
engine = FlowForge.compile(MyAgent)

await engine.run(data, route="beta")
await engine.run(data, route="alpha.a1")
await engine.run(data, route=["alpha", "gamma"])
```

- `dag.resolve_route("flow_name")` returns ancestors and descendants.
- `dag.resolve_route("flow.task")` returns that task branch and ancestors.
- Invalid route segments raise `ValueError`.
- Explicit routes override `planning_mode`.

### Task Loop

```python
@task(
    name="retry_task",
    prompt="Keep trying until valid",
    max_loops=5,
    loop_condition=lambda output: output.get("valid", False),
)
class RetryTask:
    ...
```

`loop_condition(output) == True` accepts the output and stops looping. `False`
re-runs the task chain until `max_loops` is exhausted.

### What To Avoid

- Do not add a public `@branch` decorator back.
- Do not import `BranchContext` or `BranchRunner`; they are not public runtime
  concepts in the current implementation.
- Do not reference `NodeType.BRANCH`; valid node types are global, flow, task,
  and step.
- Do not use `T = T` in a class body for a locally scoped class.
- Do not skip `python -m pytest tests/ -x -q` after changes.
- Do not remove dynamically generated modules from `sys.modules`; runtime
  class references depend on stable `__module__` values.
- Do not bypass `_validate_generated_ast()` for generated code.
- Do not write `manifest.json` without the manifest lock.

---

## 1. Overview

FlowForge is a Python framework for building agent pipelines with decorators.
Users describe structure with `@global_config`, `@flow`, `@task`, and `@step`.
FlowForge compiles those annotations into a DAG, validates the graph, and runs
it with async execution runners.

The planner can use generated docs to choose execution routes. When
`dynamic_flow=True`, missing capabilities can be generated as FlowForge code,
validated, injected, persisted, and reused.

### Design Principles

- **Annotation-first**: no manual graph wiring.
- **Type-aware**: Pydantic schemas define I/O contracts where needed.
- **DAG-native**: compilation produces a graph with route and cycle checks.
- **Dual-prompt**: `prompt` is runtime instruction; `doc` is planning metadata.
- **Recursive**: flows can contain flows; tasks can contain tasks.
- **Tool-agnostic**: MCP, HTTP, Python functions, Claude Skills, and local
  Agent Skills share `tools=[...]`.
- **Extensible**: dynamic generation can add missing flows at runtime.

---

## 2. Annotation Model

```text
@global_config
└─ @flow
   ├─ @flow
   │  └─ @task
   │     └─ @step
   └─ @task
      ├─ @task
      │  └─ @step
      └─ @step
```

| Parent | Allowed children |
|--------|------------------|
| `@global_config` | root flows |
| `@flow` | flows and tasks |
| `@task` | tasks and steps |
| `@step` | none |

Branching is configured on existing annotations with `condition`, `branches`,
and `fallback`.

---

## 3. Prompt And Doc

Each node has a user-authored `prompt`. Docs can be generated from prompts and
schemas to give the planner summaries, capabilities, preconditions, and
children overviews.

```python
@step(order=1, prompt="Detect document format and validate the input")
async def validate(ctx):
    ...
```

Generated docs are cached by prompt/schema content and reused unless forced.

---

## 4. Data Flow

```text
engine.run(input)
  -> root flow input
  -> child flow/task input
  -> step ctx.input
  -> step return value
  -> next sibling input
  -> final result
```

Runtime Pydantic validation happens at step boundaries when `input_schema` or
`output_schema` is declared.

---

## 5. Tools And Skills

FlowForge supports five tool families:

| Type | Behavior |
|------|----------|
| `FunctionTool` | Local Python function executed in the tool-use loop |
| `HTTPTool` | HTTP endpoint called by FlowForge |
| `MCPServer` | Remote MCP server with schema discovery and `tools/call` |
| `ClaudeSkill` | Anthropic-native Skills API through `container.skills` |
| `AgentSkill` | Local standard `SKILL.md` loaded into model context |

```python
@global_config(
    prompt="Engineering assistant",
    tools=[
        FunctionTool(func=calculate, name="calculator"),
        ClaudeSkill(name="pptx"),
        AgentSkill(path=".agents/skills/code-review"),
    ],
)
class Agent:
    ...
```

Use tools in `ctx.call_llm()` with angle-bracket references:

```python
await ctx.call_llm("Solve this with <calculator> and review with <code-review>")
```

`ClaudeSkill` is Anthropic-only. `AgentSkill` is provider-neutral because it
uses prompt activation.

---

## 6. Dynamic Flow Generation

When enabled, the dynamic generator:

1. receives uncovered planner requirements;
2. checks whether a matching flow already exists in the DAG or manifest;
3. builds a generation brief;
4. asks the LLM for FlowForge decorator code;
5. validates the AST for unsafe imports/calls/builtins;
6. compiles the generated class;
7. persists it to `manifest.json` when configured;
8. injects it into the DAG and replans.

Generated flows and tools are stored under `DynamicRunOptions.generated_dir`
and indexed by `manifest.json`. `auto_load_generated=True` reloads them during
the next compile.

---

## 7. Execution Lifecycle

```text
Decorate
  -> Compile
  -> Generate docs, optionally
  -> Plan, optionally
  -> Generate missing flows, optionally
  -> Execute
  -> Trace
```

After a run, `engine.last_trace` records executed nodes, statuses, timings,
branch selections, and route information.

---

## 8. Module Map

```text
flowforge/
├── annotations/      # decorators, metadata, validators
├── schema/           # DAG compiler, registry, resolver
├── execution/        # engine, runners, contexts, LLM integration
├── dynamic/          # dynamic generator, meta-flow, manifest
├── tools/            # MCP, HTTP, function, built-in tools
├── doc/              # generated docs and cache
├── planner/          # deterministic/autonomous/hybrid planning
├── viz/              # DAG and run trace rendering
└── cli/              # Typer CLI
```

---

## 9. CLI Commands

```bash
flowforge validate ./agent.py
flowforge viz ./agent.py --mermaid
flowforge run ./agent.py --query "hello" --trace
flowforge doc-generate ./agent.py --force
```

---

## 10. Verification

After code or docs changes, run:

```bash
python -m pytest tests/ -x -q
python -m mkdocs build -f flowforge/mkdocs.yml --strict --site-dir /tmp/flowforge-docs-build
```
