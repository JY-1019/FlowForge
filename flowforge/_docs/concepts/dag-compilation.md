# DAG Compilation

`FlowForge.compile(MyAgent)` converts the metadata tree into a `networkx.DiGraph`.

---

## Node ID Scheme

Every node gets a **dotted-path ID** that uniquely identifies it in the DAG:

```
global
global.{flow_name}
global.{flow_name}.{task_name}
global.{flow_name}.{task_name}.{func_name}[{order}]          # step
global.{flow_name}.{task_name}.{branch_name}[{order}]        # branch
```

Example for the ResearchAgent from the usage example:

```
global
global.research
global.research.search
global.research.search.execute_search
global.research.search.execute_search.optimize_query[1]
global.research.search.execute_search.source_select[2]
global.research.search.execute_search.deduplicate[3]
global.research.analyze_and_format
global.research.analyze_and_format.analyze
global.research.analyze_and_format.analyze.classify_intent[1]
global.research.analyze_and_format.format
global.research.analyze_and_format.format.draft_answer[1]
global.research.analyze_and_format.format.add_citations[2]
```

These IDs are used by the `RunTrace` system to map runtime execution back to DAG nodes.

---

## Edge Types

| Type | Meaning | Created by |
|------|---------|------------|
| `parent_child` | Containment (parent → child node) | Compiler's DFS traversal |
| `depends_on` | Cross-flow ordering (A must finish before B) | `@flow(depends_on=[...])` |

---

## Compile-Time Validation

The compiler raises errors immediately if any of these checks fail:

| Error | Trigger |
|-------|---------|
| `OrderConflictError` | Two `@step` or `@branch` share the same `order` in one task |
| `IOBindingError` | `output_schema` of step N ≠ `input_schema` of step N+1 |
| `BranchOutputMismatchError` | Not all branch handlers return the same type |
| `CycleDetectedError` | `depends_on` creates a cycle in the DAG |
| `CompileError` | Class missing `@global_config`, or other structural error |

---

## Inspecting the DAG

```python
engine = FlowForge.compile(MyAgent)

# All nodes
for node in engine.dag.get_all_nodes():
    print(node.id, node.type.value, node.name)

# Nodes by type
flows = engine.dag.nodes_by_type(NodeType.FLOW)

# Children of a node
children = engine.dag.get_children("global.research")

# Topological order
ordered = engine.dag.topological_order()

# Cycle detection
cycles = engine.dag.detect_cycles()   # empty list = no cycles

# Total node count
print(len(engine.dag))
```

---

## Mermaid Diagram

```python
print(engine.mermaid())
```

Output (truncated):
```
graph TD
    global["global\ntrace test agent"]
    global.main_flow["main_flow\nmain"]
    global.main_flow.step_task["step_task\nsteps"]
    ...
    global --> global.main_flow
    global.main_flow --> global.main_flow.step_task
    ...
```

Paste this into [mermaid.live](https://mermaid.live) or use the CLI:

```bash
flowforge viz my_agent.py --mermaid
```
