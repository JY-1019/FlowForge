# Engine & CompiledAgent

---

## FlowForge.compile()

```python
from flowforge import FlowForge

engine = FlowForge.compile(MyAgent)
```

Converts a `@global_config`-decorated class into a `CompiledAgent`. Raises `CompileError` if:

- The class is not decorated with `@global_config`
- The DAG has cycles (`CycleDetectedError`)
- Step orders conflict (`OrderConflictError`)

---

## CompiledAgent

The object returned by `FlowForge.compile()`. Holds the DAG and the execution engine.

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

---

### Methods

#### `run(input_data)` → `Any`

Execute the pipeline. Trace is stored in `engine.last_trace`.

```python
result = await engine.run(my_input)
```

#### `run_traced(input_data)` → `tuple[Any, RunTrace]`

Execute and explicitly return `(result, RunTrace)`:

```python
result, trace = await engine.run_traced(my_input)
print(trace.succeeded, trace.duration_ms)
```

#### `generate_docs(force=False)` → `dict[str, AnyDoc]`

Generate AI documentation for every DAG node. Uses LLM with caching.

```python
docs = await engine.generate_docs()
docs = await engine.generate_docs(force=True)  # ignore cache
```

---

### Visualization Methods

#### `visualize(output_path, **kwargs)` → `None`

Render the **full DAG structure** (not a run trace) to an SVG/PNG file.

```python
engine.visualize("dag.svg")
engine.visualize("dag.png", fmt="png", show_docs=True)
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

Raises `RuntimeError("No run trace")` if called before any run.

#### `run_mermaid(trace=None)` → `str`

Return a Mermaid diagram for the last (or given) run:

```python
await engine.run(input_data)
mmd = engine.run_mermaid()
print(mmd)
```

#### `print_run_summary(trace=None)` → `None`

Print a terminal table summarizing the run:

```python
await engine.run(input_data)
engine.print_run_summary()
```

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

# Size
count: int = len(dag)
```

### DAGNode

```python
node.id       # str — dotted-path ID, e.g. "global.research.search"
node.type     # NodeType enum: GLOBAL | FLOW | TASK | STEP | BRANCH
node.name     # str — short name
node.meta     # FlowMeta | TaskMeta | StepMeta | BranchMeta | GlobalMeta
node.parent_id # str | None
node.doc      # AnyDoc | None — populated after generate_docs()
```
