# CLI Reference

Install FlowForge and the `flowforge` command becomes available:

```bash
pip install flowforge
flowforge --help
```

---

## flowforge validate

Validate the DAG structure of an agent file.

```bash
flowforge validate <AGENT_FILE>
```

Checks:
- Class decorated with `@global_config` exists
- No duplicate `order` values within any task
- No cycles in the dependency graph
- I/O schema compatibility between consecutive steps

**Example:**

```bash
$ flowforge validate my_agent.py
Validating my_agent.py ...
✓ Valid DAG with 11 nodes.
┌────────────────────────────────────────┬────────┬───────────────────┐
│ Node ID                                │ Type   │ Name              │
├────────────────────────────────────────┼────────┼───────────────────┤
│ global                                 │ global │ my_agent          │
│ global.research                        │ flow   │ research          │
│ global.research.analyze                │ task   │ analyze           │
│ global.research.analyze.classify[1]    │ step   │ classify          │
│ ...                                    │ ...    │ ...               │
└────────────────────────────────────────┴────────┴───────────────────┘
```

---

## flowforge viz

Render the full DAG structure to a file or print Mermaid to stdout.

```bash
flowforge viz <AGENT_FILE> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `dag.svg` | Output file path |
| `--format`, `-f` | `svg` | Output format: `svg`, `png`, `pdf` |
| `--show-docs` | `False` | Include AI doc summaries in node labels |
| `--mermaid` | `False` | Print Mermaid diagram to stdout instead |

**Examples:**

```bash
# Render SVG
flowforge viz my_agent.py --output dag.svg

# Render PNG
flowforge viz my_agent.py --output dag.png --format png

# Print Mermaid to stdout
flowforge viz my_agent.py --mermaid

# Include doc summaries
flowforge viz my_agent.py --show-docs --output dag_with_docs.svg
```

---

## flowforge run

Run the agent with a query and optionally show the execution trace.

```bash
flowforge run <AGENT_FILE> --query <QUERY> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--query`, `-q` | *(required)* | Input string passed to `engine.run()` |
| `--trace`, `-t` | `False` | Print per-node execution summary table |
| `--viz` | `False` | Render executed subtree to an SVG file |
| `--viz-output` | `run.svg` | Output path for the subtree SVG |
| `--viz-fmt` | `svg` | Format: `svg`, `png`, `pdf` |
| `--viz-mermaid` | `False` | Print Mermaid subtree to stdout |

**Examples:**

```bash
# Simple run
flowforge run my_agent.py -q "process this document"

# Run with execution trace table
flowforge run my_agent.py -q "hello" --trace

# Run and render subtree SVG
flowforge run my_agent.py -q "hello" --viz --viz-output run.svg

# Run and print Mermaid subtree
flowforge run my_agent.py -q "hello" --viz-mermaid

# Run with JSON input
flowforge run my_agent.py -q '{"text": "hello", "lang": "en"}' --trace
```

**Sample trace output:**

```
Result:
Hello, World!

Run 3a1b2c4d — 8.2 ms — ✓ succeeded
┌───────┬────────┬────────────────┬────────┬─────────┬─────────┐
│ Order │ Type   │ Name           │ ms     │ Branch  │ Status  │
├───────┼────────┼────────────────┼────────┼─────────┼─────────┤
│     1 │ flow   │ research       │   7.1  │         │ ✓       │
│     2 │ task   │ execute_search │   6.0  │         │ ✓       │
│     3 │ step   │ optimize_query │   1.5  │         │ ✓       │
│     4 │ branch │ source_select  │   3.2  │ web     │ ✓       │
│     5 │ step   │ deduplicate    │   0.8  │         │ ✓       │
└───────┴────────┴────────────────┴────────┴─────────┴─────────┘
```

---

## flowforge doc-generate

Generate AI documentation for all nodes in the agent.

```bash
flowforge doc-generate <AGENT_FILE> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--force` | `False` | Ignore cache and regenerate all docs |

Docs are cached by SHA-256 of the prompt content. Re-run without `--force` to use cache.

**Example:**

```bash
# Generate docs (uses cache if available)
flowforge doc-generate my_agent.py

# Force regeneration
flowforge doc-generate my_agent.py --force
```

---

## Global Options

```bash
flowforge --help          # show all commands
flowforge <cmd> --help    # show options for a specific command
```
