# FlowForge Design Notes

> Annotation-Based AI Agent Framework  
> Current implementation notes for the v1.x documentation set.

This page is a compact design companion to the guides and API reference. It
summarizes what FlowForge is, how the core pieces fit together, and which
features are available today.

## 1. Overview

FlowForge is a Python package for building AI-agent systems with decorators:

```text
@global_config -> @flow -> @task -> @step
```

Those annotations are compiled into a validated DAG, then executed by async
runners. The same DAG powers route selection, LLM planning, visualization,
dynamic flow generation, and trace reporting.

## 2. Design Principles

- **Annotation-first**: agent structure lives in normal Python classes and
  functions.
- **DAG-native**: metadata compiles to explicit graph nodes and edges.
- **Typed where useful**: Pydantic schemas validate boundaries when supplied.
- **Prompt/document split**: `prompt` guides runtime behavior; generated docs
  guide planning.
- **Composable**: flows can contain flows; tasks can contain tasks.
- **Tool-agnostic**: Python functions, HTTP tools, MCP tools, Claude Skills,
  and local Agent Skills use one tool surface.
- **Extensible at runtime**: dynamic generation can add missing flows.

## 3. Public Decorators

| Decorator | Role |
|-----------|------|
| `@global_config` | Agent root: global prompt, model config, tools, dynamic settings |
| `@flow` | Pipeline stage; may contain flows/tasks or dispatch to branch flows |
| `@task` | Work unit; may contain child tasks/steps or dispatch to branch tasks |
| `@step` | Async function that performs one action or dispatches to branch handlers |

There is no public `@branch` decorator. Branch behavior is built into
`@step`, `@task`, and `@flow` through `condition`, `branches`, and `fallback`.

## 4. Ordering Model

| Configuration | Behavior |
|---------------|----------|
| `order=None` | Auto-sequential, preserving declaration order |
| same explicit `order` | Parallel group; each sibling receives the same input |
| `unique=True` | Exclusive runner for its same-order group |

Only one sibling in a same-order group may set `unique=True`.

## 5. Data Flow

```text
engine.run(input)
  -> root flow group
  -> child flow groups
  -> task groups
  -> step groups
  -> final output
```

Each return value becomes the next node's input. Schemas validate values at
the boundaries where they are declared.

## 6. Branch Dispatch

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

At runtime FlowForge reads `source` from the current input, records
`ctx.condition_value` and `ctx.selected_branch`, then forwards the selected
handler's output.

## 7. Tools And LLM Calls

Tool configurations:

| Type | Behavior |
|------|----------|
| `FunctionTool` | Local Python callable |
| `HTTPTool` | HTTP request through `httpx` |
| `MCPServer` | MCP Streamable HTTP tool integration |
| `ClaudeSkill` | Anthropic-native Skill attached to Messages API calls |
| `AgentSkill` | Local `SKILL.md` instructions injected into model context |

Steps can call tools directly:

```python
result = await ctx.call_tool("search", query="FlowForge")
```

Or expose tools to the LLM:

```python
answer = await ctx.call_llm("Search for {query} with <search>")
```

## 8. Planning And Docs

`engine.generate_docs()` creates structured node docs. In autonomous and
hybrid modes, the planner uses those docs to select which root flows to run:

```python
engine = FlowForge.compile(MyAgent)
await engine.generate_docs(planning_only=True)
result = await engine.run(data, planning_mode="autonomous")
```

Passing `route=...` skips planner selection and runs the explicit route.

## 9. Dynamic Flow Generation

With `@global_config(dynamic_flow=True)`, FlowForge can generate missing
capabilities when autonomous or hybrid planning finds a gap.

`DynamicRunOptions` controls persistence, generated directories, builtin
tools, MCP server declarations, shell access, and dependency installation
policy.

```python
options = DynamicRunOptions(project_root=".", persist_generated=True)
engine = FlowForge.compile(MyAgent, dynamic_options=options)
```

Generated flows can be persisted to `manifest.json`, auto-loaded on later
compiles, and repaired or replaced when runtime self-repair succeeds.

## 10. Execution Features

| Feature | API |
|---------|-----|
| deterministic run | `await engine.run(data)` |
| explicit route | `await engine.run(data, route="flow.task")` |
| traced run | `result, trace = await engine.run_traced(data)` |
| task loop | `@task(max_loops=N, loop_condition=...)` |
| best-effort task | `@task(on_error="skip_remaining")` |
| human approval gate | `@step(approval=True)` |
| pass/fail judge retries | `@step(pass_criteria="...", pass_criteria_max_retries=N)` |
| session isolation | `session = engine.create_session()` |

## 11. Visualization

```python
print(engine.mermaid())
engine.visualize("dag.svg")

result, trace = await engine.run_traced(data)
engine.print_run_summary(trace)
print(engine.run_mermaid(trace))
engine.visualize_run("run.svg", trace)
```

The run trace includes executed nodes, duration, success/failure, selected
branches, and checkpoint data for approval/resume flows.

## 12. Module Map

```text
flowforge/
├── annotations/      decorators, metadata, validators
├── schema/           DAG compiler and resolver
├── execution/        engine, contexts, runners, memory, LLM calls
├── dynamic/          dynamic generation, manifest, safety, MCP provisioning
├── tools/            MCP, HTTP, function, builtin tool adapters
├── doc/              generated docs and cache
├── planner/          planning prompt and path selection
├── viz/              full-DAG and run-trace rendering
└── cli/              Typer command line interface
```
