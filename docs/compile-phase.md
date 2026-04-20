# Phase 1: Compile — How FlowForge Builds the DAG

FlowForge's compile phase converts your Python decorator annotations into an immutable, validated DAG (Directed Acyclic Graph). This document explains exactly what happens, step by step, from the moment you write `@step(...)` to the moment `FlowForge.compile()` returns a `CompiledAgent`.

---

## Overview

```
Your Python source code
        │
        ▼  (Python imports the module)
┌─────────────────────────┐
│  Step 1: Decoration     │  @step / @branch mark functions
│                         │  @task / @flow / @global_config
│                         │  scan their class bodies
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Step 2: Metadata Tree  │  In-memory tree of dataclasses:
│                         │  GlobalMeta → FlowMeta → TaskMeta
│                         │            → StepMeta / BranchMeta
└─────────────────────────┘
        │
        ▼  (FlowForge.compile(MyAgent))
┌─────────────────────────┐
│  Step 3: DAG Build      │  DFS traversal of the metadata tree
│                         │  → DAGNode + DAGEdge per element
│                         │  → depends_on edges added
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Step 4: Validation     │  Topological sort (cycle detection)
│                         │  Already ran at decoration time:
│                         │  order uniqueness, I/O chain, branch
│                         │  output consistency
└─────────────────────────┘
        │
        ▼
  CompiledAgent(.dag, .docs, .run(), .mermaid(), ...)
```

---

## Step 1: Decoration — How Python Processes the Decorators

### The key insight: Python processes inner class decorators first

Because Python evaluates a class body from top to bottom and applies decorators *before* returning control to the outer scope, the decoration order is always **innermost first, outermost last**.

Given this code:

```python
@global_config(prompt="...")
class MyAgent:

    @flow(name="my_flow", prompt="...")
    class MyFlow:

        @task(name="my_task", prompt="...")
        class MyTask:

            @step(order=1, prompt="...")
            async def step_one(ctx): ...

            @step(order=2, prompt="...")
            async def step_two(ctx): ...
```

Python applies the decorators in this exact order:

```
1. @step(order=1)   applied to step_one   → StepMeta attached to function
2. @step(order=2)   applied to step_two   → StepMeta attached to function
3. @task            applied to MyTask     → scans MyTask.__dict__, finds step_one and step_two
4. @flow            applied to MyFlow     → scans MyFlow.__dict__, finds MyTask
5. @global_config   applied to MyAgent   → scans MyAgent.__dict__, finds MyFlow
```

By the time `@task` runs, the step functions already carry their metadata. By the time `@flow` runs, the task class already carries its metadata. This "bottom-up" property is what makes the annotation approach work without any separate registration step.

---

### @step — What it does

**File:** `flowforge/annotations/decorators.py:35`

```python
@step(order=1, prompt="validate input", input_schema=RawDoc, output_schema=ValidDoc)
async def validate(ctx): ...
```

1. Creates a `StepMeta` dataclass with all parameters plus a reference to the function itself (`func=validate`).
2. Attaches it as `validate.__flowforge_step_meta__ = StepMeta(...)`.
3. Returns the original function unchanged — the function is still callable normally.

**What `StepMeta` holds:**

| Field | Type | Notes |
|-------|------|-------|
| `order` | `int` | Execution position within the parent task |
| `prompt` | `str` | Natural language description (used by LLM at runtime) |
| `func` | `Callable` | Reference to the actual async function |
| `input_schema` | `type \| None` | Pydantic model for input validation |
| `output_schema` | `type \| None` | Pydantic model for output validation |
| `tool_mode` | `bool` | If True, the step is a dynamic LLM tool instead of a fixed chain node |
| `timeout_seconds` | `int` | Execution time limit |

---

### @branch — What it does

**File:** `flowforge/annotations/decorators.py:66`

```python
@branch(
    order=2,
    name="format_router",
    prompt="route by document format",
    condition=BranchCondition(field="doc_type", enum=["csv", "json", "xml"]),
    branches={"csv": csv_handler, "json": json_handler, "xml": xml_handler},
    fallback=default_handler,
)
async def route_by_format(ctx): ...
```

1. Creates a `BranchMeta` dataclass.
2. **Immediately runs `validate_branch_output_consistency(meta)`** — if the handlers have type annotations that return different types, a `BranchOutputMismatchError` is raised right here, at class-definition time.
3. Attaches it as `route_by_format.__flowforge_branch_meta__ = BranchMeta(...)`.

The `condition.field` names the field on the input object whose value determines which handler to call. The `branches` dict maps each possible field value to a handler function.

---

### @task — What it does

**File:** `flowforge/annotations/decorators.py:102`

```python
@task(name="process_doc", prompt="parse and enrich a document")
class ProcessDocTask:
    @step(order=1, prompt="detect format") async def detect(ctx): ...
    @step(order=2, prompt="parse")         async def parse(ctx):  ...
    @branch(order=3, name="enrich_router", ...) async def route(ctx): ...
```

When `@task` runs, `ProcessDocTask` is passed to the decorator. It does:

1. Iterates `ProcessDocTask.__dict__` (direct attributes only, not inherited).
2. For each attribute value:
   - If it has `__flowforge_step_meta__` → append to `steps` list
   - If it has `__flowforge_branch_meta__` → append to `steps` list (steps and branches share the same list and order namespace)
   - If it's a class with `__flowforge_task_meta__` → append to `child_tasks` list
3. Constructs `TaskMeta(name, prompt, cls, steps=[...], child_tasks=[...])`.
4. **Runs two validators immediately:**
   - `validate_order_uniqueness(meta)` — checks for duplicate `order` values within the `steps` list
   - `validate_io_chain(meta)` — checks that consecutive nodes' output/input schemas are compatible
5. Attaches it as `ProcessDocTask.__flowforge_task_meta__ = TaskMeta(...)`.

A task with no `child_tasks` is a **leaf task** (`meta.is_leaf == True`). Only leaf tasks directly contain steps/branches. A task with `child_tasks` is a **container task** — it orchestrates its children sequentially.

---

### @flow — What it does

**File:** `flowforge/annotations/decorators.py:150`

```python
@flow(name="pipeline", prompt="full pipeline", depends_on=["auth_flow"])
class PipelineFlow:
    @flow(name="ingest", prompt="...") class IngestSubFlow: ...
    @task(name="transform", prompt="...") class TransformTask: ...
```

1. Iterates the class `__dict__`.
2. Collects classes with `__flowforge_flow_meta__` → `child_flows` list.
3. Collects classes with `__flowforge_task_meta__` → `tasks` list.
4. Builds `FlowMeta(name, prompt, child_flows, tasks, depends_on, parallel, max_retries, ...)`.
5. Attaches it as `PipelineFlow.__flowforge_flow_meta__ = FlowMeta(...)`.

No validation runs at this level yet (cycle detection requires the full graph, which isn't available until compile time).

---

### @global_config — What it does

**File:** `flowforge/annotations/decorators.py:197`

```python
@global_config(
    prompt="You are a research assistant.",
    llm_config=LLMConfig(model="claude-sonnet-4-6", temperature=0.3),
    tools=[MCPServer("https://search.example.com/mcp")],
)
class ResearchAgent:
    @flow(name="research", prompt="...") class ResearchFlow: ...
```

1. Iterates the class `__dict__`.
2. Collects all classes with `__flowforge_flow_meta__` → `flows` list. These are the root-level flows.
3. Builds `GlobalMeta(prompt, cls, llm_config, tools, flows=[...])`.
4. Attaches it as `ResearchAgent.__flowforge_global_meta__ = GlobalMeta(...)`.

After this, the entire agent structure lives as a tree of in-memory dataclasses hanging off the agent class.

---

## Step 2: The Metadata Tree

After all decorators have run, the in-memory structure looks like this:

```
GlobalMeta
├── prompt: "You are a research assistant."
├── llm_config: LLMConfig(model="claude-sonnet-4-6", ...)
├── tools: [MCPServer(...)]
└── flows: [
      FlowMeta(name="research")
      ├── child_flows: [
      │     FlowMeta(name="search")
      │     └── tasks: [
      │           TaskMeta(name="execute_search", is_leaf=True)
      │           └── steps: [
      │                 StepMeta(order=1, func=optimize_query),
      │                 BranchMeta(order=2, name="source_select", ...),
      │                 StepMeta(order=3, func=deduplicate)
      │               ]
      │         ]
      │   ]
      └── tasks: [
            TaskMeta(name="analyze_and_format", is_leaf=False)
            └── child_tasks: [
                  TaskMeta(name="analyze", is_leaf=True)
                  └── steps: [StepMeta(order=1, func=classify_intent)]
                  TaskMeta(name="format", is_leaf=True)
                  └── steps: [StepMeta(order=1, func=draft_answer),
                               StepMeta(order=2, func=add_citations)]
                ]
          ]
    ]
```

This tree is pure data. No execution has happened. No networkx graph exists yet.

---

## Step 3: `FlowForge.compile()` — Building the DAG

**File:** `flowforge/__init__.py:104`, `flowforge/schema/compiler.py`

```python
engine = FlowForge.compile(ResearchAgent)
```

`compile()` does three things:

### 3a. Read the GlobalMeta

```python
# flowforge/__init__.py
global_meta = getattr(agent_cls, "__flowforge_global_meta__")
```

This retrieves the `GlobalMeta` attached by `@global_config`.

### 3b. Build the DAG via DFS traversal

**File:** `flowforge/schema/compiler.py:14`

`compile_dag(global_meta)` performs a depth-first traversal of the metadata tree and adds a `DAGNode` + `DAGEdge` for every element it visits.

**Node ID scheme** — dotted path from root:

| Element | Node ID |
|---------|---------|
| Global root | `"global"` |
| Flow `research` | `"global.research"` |
| Child flow `search` | `"global.research.search"` |
| Task `execute_search` | `"global.research.search.execute_search"` |
| Step `optimize_query` at order 1 | `"global.research.search.execute_search.optimize_query[1]"` |
| Branch `source_select` at order 2 | `"global.research.search.execute_search.source_select[2]"` |

The traversal algorithm:

```
compile_dag(global_meta):
  1. Add node "global"  (NodeType.GLOBAL)
  2. For each FlowMeta in global_meta.flows:
       _add_flow(dag, flow_meta, parent_id="global")

_add_flow(dag, flow_meta, parent_id):
  1. node_id = f"{parent_id}.{flow_meta.name}"
  2. Add DAGNode(id=node_id, type=FLOW, meta=flow_meta)
  3. Add DAGEdge(source=parent_id, target=node_id, type="parent_child")
  4. Recurse: for each child_flow → _add_flow(dag, child_flow, node_id)
  5. For each task_meta → _add_task(dag, task_meta, node_id)

_add_task(dag, task_meta, parent_id):
  1. node_id = f"{parent_id}.{task_meta.name}"
  2. Add DAGNode(id=node_id, type=TASK, meta=task_meta)
  3. Add DAGEdge(parent_id → node_id, "parent_child")
  4. If leaf task: for each step/branch sorted by order → _add_step_or_branch(...)
     If container:  for each child_task → _add_task(dag, child_task, node_id)

_add_step_or_branch(dag, meta, parent_id):
  1. name = func.__name__ (for StepMeta) or meta.name (for BranchMeta)
  2. node_id = f"{parent_id}.{name}[{meta.order}]"
  3. Add DAGNode + DAGEdge
```

### 3c. Add `depends_on` edges

After the tree is fully added, a second pass adds cross-flow dependency edges:

```python
# flowforge/schema/compiler.py:108
def _add_depends_on_edges(dag):
    flow_nodes = {n.name: n for n in dag.nodes_by_type(NodeType.FLOW)}
    for node in dag.nodes_by_type(NodeType.FLOW):
        for dep_name in node.meta.depends_on:
            dep_node = flow_nodes.get(dep_name)
            if dep_node:
                dag.add_edge(DAGEdge(dep_node.id, node.id, "depends_on"))
```

`parent_child` edges express containment (flow owns task, task owns step). `depends_on` edges express sequencing between sibling flows (flow B must run after flow A completes).

---

## Step 4: Validation

### Validation at decoration time (already ran before compile())

These checks fire the moment a decorator is applied — not during `compile()`. If they fail, the Python module fails to import.

| Check | Where | Error raised |
|-------|-------|-------------|
| Order uniqueness within a leaf task | `@task` decorator | `OrderConflictError` |
| Consecutive step/branch I/O schema compatibility | `@task` decorator | `IOBindingError` |
| All branch handlers return the same type | `@branch` decorator | `BranchOutputMismatchError` |

**Example — `OrderConflictError` fires immediately:**
```python
@task(name="bad_task", prompt="...")
class BadTask:
    @step(order=1, prompt="a") async def step_a(ctx): ...
    @step(order=1, prompt="b") async def step_b(ctx): ...  # ← ERROR HERE
# OrderConflictError: Task 'bad_task': duplicate order=1
```

**`validate_order_uniqueness` logic** (`flowforge/annotations/validators.py:12`):
```
seen = {}
for each node in task_meta.steps:
    if node.order in seen → raise OrderConflictError
    seen[node.order] = node_name
```

**`validate_io_chain` logic** (`flowforge/annotations/validators.py:26`):
```
sort steps by order
for consecutive pair (A, B):
    if A.output_schema is None or B.input_schema is None → skip (auto-binding)
    if A.output_schema is not B.input_schema → raise IOBindingError
```

### Validation at `FlowForge.compile()` time

**File:** `flowforge/__init__.py:122`, `flowforge/schema/resolver.py`

After the DAG is built, `compile()` calls:

```python
resolve_execution_order(dag)
```

This runs `nx.topological_sort(dag._graph)`. If the graph has a cycle (e.g., flow A `depends_on` flow B, and flow B `depends_on` flow A), networkx raises and FlowForge converts it to:

```
CycleDetectedError: Cycle detected in DAG: ['global.flow_a', 'global.flow_b', 'global.flow_a']
```

---

## What `FlowForge.compile()` Returns

`compile()` returns a `CompiledAgent` object with the following interface:

```python
engine = FlowForge.compile(ResearchAgent)

engine.dag          # FlowForgeDAG — the full graph
engine.docs         # dict[node_id, AnyDoc] — empty until generate_docs() is called

# Inspect the DAG
engine.dag.get_node("global.research.search")        # DAGNode
engine.dag.get_children("global.research")           # list[DAGNode]
engine.dag.nodes_by_type(NodeType.STEP)              # all step nodes
engine.dag.topological_order()                       # nodes sorted by dependency
engine.dag.detect_cycles()                           # [] if clean

# Visualize
engine.mermaid()                                     # str: Mermaid diagram
engine.visualize("dag.svg")                          # renders to file

# Proceed to Phase 2
await engine.generate_docs()                         # Phase 2: Doc generation
await engine.run(UserQuery(query="..."))             # Phase 3+4: Plan + Execute
```

The `CompiledAgent` is immutable after construction. The DAG is a complete, validated snapshot of the agent structure. Nothing is executed yet — the compile phase is purely structural.

---

## Full Worked Example

Given the research agent from the spec:

```python
@global_config(prompt="Research assistant", llm_config=LLMConfig(...))
class ResearchAgent:

    @flow(name="research", prompt="...")
    class ResearchFlow:

        @flow(name="search", prompt="...")
        class SearchSubFlow:
            @task(name="execute_search", prompt="...")
            class ExecuteSearchTask:
                @step(order=1, prompt="optimize") async def optimize_query(ctx): ...
                @branch(order=2, name="source_select", ...) async def route_source(ctx): ...
                @step(order=3, prompt="dedup") async def deduplicate(ctx): ...

        @task(name="analyze_and_format", prompt="...")
        class AnalyzeAndFormatTask:
            @task(name="analyze", prompt="...") class AnalyzeTask:
                @step(order=1, prompt="classify") async def classify_intent(ctx): ...
            @task(name="format", prompt="...")  class FormatTask:
                @step(order=1, prompt="draft")  async def draft_answer(ctx): ...
                @step(order=2, prompt="cite")   async def add_citations(ctx): ...
```

The resulting DAG after `FlowForge.compile()`:

```
global  (GLOBAL)
└── global.research  (FLOW)
    ├── global.research.search  (FLOW)
    │   └── global.research.search.execute_search  (TASK, leaf)
    │       ├── global.research.search.execute_search.optimize_query[1]   (STEP)
    │       ├── global.research.search.execute_search.source_select[2]    (BRANCH)
    │       └── global.research.search.execute_search.deduplicate[3]      (STEP)
    └── global.research.analyze_and_format  (TASK, container)
        ├── global.research.analyze_and_format.analyze  (TASK, leaf)
        │   └── global.research.analyze_and_format.analyze.classify_intent[1]  (STEP)
        └── global.research.analyze_and_format.format  (TASK, leaf)
            ├── global.research.analyze_and_format.format.draft_answer[1]  (STEP)
            └── global.research.analyze_and_format.format.add_citations[2]  (STEP)

Total: 13 nodes, no cycles
```

Verify with CLI:
```bash
flowforge validate examples/research_agent.py
# ✓ Valid DAG with 13 nodes.

flowforge viz examples/research_agent.py --mermaid
# graph TD
#   global["global: global"]
#   ...
```

---

## Error Reference

| Error | When it fires | Cause |
|-------|--------------|-------|
| `OrderConflictError` | At `@task` decoration | Two `@step` or `@branch` nodes share the same `order` number inside one leaf task |
| `IOBindingError` | At `@task` decoration | Node A's `output_schema` ≠ Node B's `input_schema` for consecutive A→B pair (when both are non-None) |
| `BranchOutputMismatchError` | At `@branch` decoration | Branch handlers have different return type annotations |
| `CycleDetectedError` | At `FlowForge.compile()` | `depends_on` references form a cycle between flows |
| `CompileError` | At `FlowForge.compile()` | The class passed to `compile()` is not decorated with `@global_config` |

---

## Key Files

| File | Role in compile phase |
|------|-----------------------|
| `flowforge/annotations/decorators.py` | The five decorator factories |
| `flowforge/annotations/metadata.py` | `StepMeta`, `BranchMeta`, `TaskMeta`, `FlowMeta`, `GlobalMeta` dataclasses |
| `flowforge/annotations/validators.py` | Order uniqueness, I/O chain, branch output consistency checks |
| `flowforge/schema/dag.py` | `DAGNode`, `DAGEdge`, `FlowForgeDAG` (networkx wrapper) |
| `flowforge/schema/compiler.py` | DFS traversal: metadata tree → DAG nodes and edges |
| `flowforge/schema/resolver.py` | Topological sort + cycle detection |
| `flowforge/__init__.py` | `FlowForge.compile()` and `CompiledAgent` |
