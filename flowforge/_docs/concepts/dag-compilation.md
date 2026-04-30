# DAG Compilation

`FlowForge.compile(MyAgent)` turns decorator metadata into a `networkx` DAG.
The compiled DAG is the source of truth for validation, route resolution,
planning, execution, and visualization.

## Node Types

FlowForge has four DAG node types:

| NodeType | Created from |
|----------|--------------|
| `GLOBAL` | `@global_config` |
| `FLOW` | `@flow` |
| `TASK` | `@task` |
| `STEP` | `@step` |

Branch dispatching does not create a separate node type. A branch dispatcher is
a normal flow, task, or step whose metadata has `condition` and `branches`.

## Node IDs

```text
global
global.{flow_name}
global.{flow_name}.{task_name}
global.{flow_name}.{task_name}.{step_func_name}[{order}]
```

Example:

```text
global
global.research
global.research.search
global.research.search.expand_query[1]
global.research.search.fetch_sources[2]
```

Nested flows and tasks extend the dotted path in the same way.

## Edge Types

| Edge | Meaning | Source |
|------|---------|--------|
| `parent_child` | A node contains another node | metadata traversal |
| `depends_on` | A flow must run after another flow | `@flow(depends_on=[...])` |

## Order Groups

Ordering is stored on metadata and enforced by the runners:

| `order` value | Behavior |
|---------------|----------|
| `None` | Auto-sequential; insertion order is preserved |
| same explicit integer | Same-order siblings run in parallel |
| `unique=True` | Only that node runs within its same-order group |

Same-order nodes are valid. Multiple same-order nodes with `unique=True` are
invalid.

## Validation

| Error | Trigger |
|-------|---------|
| `CompileError` | Missing `@global_config`, invalid branch target, duplicate flow add |
| `CycleDetectedError` | `depends_on` creates a cycle |
| `OrderConflictError` | More than one `unique=True` node in an order group |
| `IOBindingError` | Consecutive step groups have incompatible schemas |
| `BranchOutputMismatchError` | Step branch handlers advertise inconsistent return types |

## Inspecting The DAG

```python
from flowforge.schema.dag import NodeType

engine = FlowForge.compile(MyAgent)

for node in engine.dag.get_all_nodes():
    print(node.id, node.type.value, node.name)

flows = engine.dag.nodes_by_type(NodeType.FLOW)
children = engine.dag.get_children("global.research")
ordered = engine.dag.topological_order()
cycles = engine.dag.detect_cycles()
```

## Route Resolution

```python
node_ids = engine.dag.resolve_route("research")
node_ids = engine.dag.resolve_route("research.search")
```

`engine.run(..., route=...)` uses the same resolver. A flow route includes the
selected flow and descendants. A task route includes its ancestors plus the
task subtree.
