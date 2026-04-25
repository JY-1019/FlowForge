# Engine & CompiledAgent

---

## FlowForge.compile()

```python
from flowforge import FlowForge

engine = FlowForge.compile(MyAgent)

# With dynamic flow generation:
from flowforge import DynamicRunOptions
options = DynamicRunOptions(persist_generated=True)
engine = FlowForge.compile(MyAgent, dynamic_options=options)
```

Converts a `@global_config`-decorated class into a `CompiledAgent`. Raises `CompileError` if:

- The class is not decorated with `@global_config`
- The DAG has cycles (`CycleDetectedError`)
- Step orders conflict (`OrderConflictError`)

When `dynamic_options` is provided and the agent has `dynamic_flow=True`, the
internal `_dynamic_generator` meta-flow is injected into the DAG. If
`auto_load_generated=True`, previously generated flows and tools are loaded
from `manifest.json` during compile, before planning/running.

The manifest also acts as a cache: before creating a new dynamic flow,
FlowForge checks both the in-memory DAG and the persisted manifest. If a flow
with the target name already exists, generation is skipped and the existing
flow is reused.

---

## CompiledAgent

The object returned by `FlowForge.compile()`. Holds the DAG, docs, and a default execution engine.

For **single-user / CLI** usage, call `run()` directly on this object.
For **multi-user / server** usage, call `create_session()` to get isolated per-user sessions.

---

### Properties

#### `.dag` — `FlowForgeDAG`

Access the compiled DAG directly:

```python
for node in engine.dag.get_all_nodes():
    print(node.id, node.type.value, node.name)

cycles   = engine.dag.detect_cycles()   # [] if valid
ordered  = engine.dag.topological_order()
children = engine.dag.get_children("global.research")
count    = len(engine.dag)
```

#### `.docs` — `dict[str, AnyDoc]`

AI-generated documentation for each node (populated after `generate_docs()`):

```python
for node_id, doc in engine.docs.items():
    print(f"{node_id}: {doc.summary}")
```

#### `.last_trace` — `RunTrace | None`

Trace of the most recent `run()` call. `None` before the first run.

```python
await engine.run(input_data)
print(engine.last_trace.duration_ms)
```

#### `.memory` — `SessionMemory`

Session memory that persists across `run()` calls. Stores compact summaries of previous runs so the LLM can reference earlier results.

```python
engine.memory.clear()  # reset memory
```

#### `.last_dynamic_generation` — `dict[str, Any] | None`

Information about the most recent dynamic flow generation, if any:

```python
result = await engine.run(input_data, planning_mode="autonomous")
dyn = engine.last_dynamic_generation
if dyn and dyn.get("success"):
    print(f"Generated: {dyn['dynamic_flow']}")
    print(dyn["generated_code"])
```

---

### Methods

#### `run(input_data, ...)` → `Any`

Execute the pipeline. Trace is stored in `engine.last_trace`.

```python
result = await engine.run(my_input)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `Any` | (required) | Input passed to the first flow |
| `planning_mode` | `str` | `"deterministic"` | `"deterministic"`, `"autonomous"`, or `"hybrid"` |
| `route` | `str \| list[str] \| None` | `None` | Execute only specific paths |
| `resume_from` | `Checkpoint \| None` | `None` | Resume from a previous run's checkpoint |
| `dynamic_options` | `DynamicRunOptions \| None` | `None` | Override dynamic generation options for this run |

**Planning modes:**

- `"deterministic"` — run all nodes in compiled order (default)
- `"autonomous"` — call the LLM planner first; only selected nodes execute
- `"hybrid"` — same as autonomous but allows minor deviations

**Route examples:**

```python
# Run only the "search" flow
result = await engine.run(data, route="search")

# Run a specific task within a flow
result = await engine.run(data, route="research.analyze")

# Run multiple routes
result = await engine.run(data, route=["analysis", "report"])
```

#### `run_traced(input_data, ...)` → `tuple[Any, RunTrace]`

Execute and explicitly return `(result, RunTrace)`:

```python
result, trace = await engine.run_traced(my_input)
print(trace.succeeded, trace.duration_ms)
```

Accepts the same parameters as `run()`.

#### `generate_docs(force=False, planning_only=False)` → `dict[str, AnyDoc]`

Generate AI documentation for DAG nodes. Uses LLM with caching.

```python
docs = await engine.generate_docs()
docs = await engine.generate_docs(force=True)          # ignore cache
docs = await engine.generate_docs(planning_only=True)  # only GLOBAL + FLOW nodes
```

When `planning_only=True`, only GLOBAL and FLOW level docs are generated — sufficient for the autonomous/hybrid planner and much faster.

#### `create_session()` → `AgentSession`

Create a per-user session with isolated memory and trace. See [AgentSession](#agentsession) below.

```python
session = engine.create_session()
result = await session.run(user_input)
```

---

### Visualization Methods

#### `visualize(output_path, **kwargs)` → `None`

Render the **full DAG structure** to an SVG/PNG file. Falls back to Mermaid if graphviz is not installed.

```python
engine.visualize("dag.svg")
engine.visualize("dag.png", fmt="png")
```

#### `mermaid()` → `str`

Return a Mermaid diagram string of the full DAG:

```python
print(engine.mermaid())
```

#### `visualize_run(output_path, trace=None, fmt="svg")` → `str`

Render the executed subtree of the last (or given) run.

```python
await engine.run(input_data)
path = engine.visualize_run("run.svg")
path = engine.visualize_run("run.png", fmt="png")

# Explicit trace
result, trace = await engine.run_traced(input_data)
path = engine.visualize_run("run.svg", trace=trace)
```

#### `run_mermaid(trace=None)` → `str`

Return a Mermaid diagram for the last (or given) run.

#### `compare_mermaid(trace=None)` → `str`

Return a Markdown string with two Mermaid diagrams: the full DAG and the executed path side-by-side.

```python
await engine.run(input_data)
md = engine.compare_mermaid()
# Paste into any Markdown viewer to compare
```

#### `print_run_summary(trace=None)` → `None`

Print a terminal table summarizing the run:

```python
await engine.run(input_data)
engine.print_run_summary()
```

---

## AgentSession

Per-user session with isolated memory and trace. Created via `engine.create_session()`.

The heavy, immutable resources (DAG, docs, tool registry) are **shared** with the parent `CompiledAgent` — no duplication. Only memory and trace are isolated.

### Usage

```python
# Single-user (CLI, scripts)
engine = FlowForge.compile(MyAgent)
result = await engine.run(input_data)

# Multi-user (FastAPI, etc.)
engine = FlowForge.compile(MyAgent)
await engine.generate_docs(planning_only=True)

# Per-request
session = engine.create_session()
result = await session.run(user_input)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.memory` | `SessionMemory` | Per-session memory, persists across `run()` calls |
| `.last_trace` | `RunTrace \| None` | Trace of the most recent run in this session |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run(input_data, ...)` | `Any` | Same interface as `CompiledAgent.run()` |
| `run_traced(input_data, ...)` | `tuple[Any, RunTrace]` | Same interface as `CompiledAgent.run_traced()` |
| `compare_mermaid(trace=None)` | `str` | Full DAG vs executed path comparison |

---

## FlowForgeDAG

Returned by `engine.dag`. Wraps a `networkx.DiGraph`.

```python
dag = engine.dag

# All nodes
nodes: list[DAGNode] = dag.get_all_nodes()

# Children of a node
children: list[DAGNode] = dag.get_children("global.research")

# Parents of a node
parents: list[DAGNode] = dag.get_parents("global.research.search")

# Nodes by type
from flowforge.schema.dag import NodeType
flows = dag.nodes_by_type(NodeType.FLOW)

# Topological order
ordered: list[DAGNode] = dag.topological_order()

# Cycle detection
cycles: list = dag.detect_cycles()   # empty = no cycles

# Route resolution
node_ids = dag.resolve_route("search")            # flow + descendants
node_ids = dag.resolve_route("research.analyze")   # specific task

# Size
count: int = len(dag)
```

### DAGNode

```python
node.id        # str — dotted-path ID, e.g. "global.research.search"
node.type      # NodeType enum: GLOBAL | FLOW | TASK | STEP
node.name      # str — short name
node.meta      # FlowMeta | TaskMeta | StepMeta | GlobalMeta
node.parent_id # str | None
node.doc       # AnyDoc | None — populated after generate_docs()
```
