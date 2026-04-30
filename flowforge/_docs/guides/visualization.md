# Run Visualization

FlowForge records a `RunTrace` for every `run()` and `run_traced()` call. A
trace contains executed node IDs, timing, status, branch choices, errors, and
checkpoints.

## Quick Start

```python
engine = FlowForge.compile(MyAgent)
result, trace = await engine.run_traced(my_input)

engine.print_run_summary(trace)
print(engine.run_mermaid(trace))
engine.visualize_run("run.svg", trace)
```

## Full DAG Diagram

```python
print(engine.mermaid())
engine.visualize("dag.svg")
```

`visualize()` uses Graphviz when available. If Graphviz is not installed, it
falls back to writing Mermaid Markdown.

## Executed Path Diagram

```python
await engine.run(data, route="research")
print(engine.run_mermaid())
engine.visualize_run("research-run.svg")
```

Skipped nodes are shown differently from executed nodes, so route filtering
and planner choices are easy to inspect.

## Compare Full DAG And Run

```python
report = engine.compare_mermaid(include_full_dag=True)
print(report)
```

This returns Markdown containing the full DAG and the executed path.

## Trace Object

```python
for node in trace.nodes:
    print(node.execution_order, node.node_type, node.name, node.duration_ms)
    if node.selected_branch:
        print("branch:", node.condition_value, "->", node.selected_branch)
```

`engine.last_trace` always points to the most recent run trace.
