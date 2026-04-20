# Installation & Quickstart

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Pydantic | 2.7+ |
| anyio | 4.x |
| networkx | 3.3+ |

---

## Install

```bash
# Minimal install
pip install flowforge

# With docs server (MkDocs Material)
pip install flowforge[docs]

# Development install (editable + test deps)
git clone https://github.com/yourusername/flowforge
cd flowforge
pip install -e ".[dev]"
```

---

## Hello World

The smallest possible FlowForge agent:

```python
# hello_agent.py
import asyncio
from flowforge import global_config, flow, task, step, FlowForge

@global_config(prompt="You are a helpful assistant.")
class HelloAgent:

    @flow(name="hello", prompt="Greet the user")
    class HelloFlow:

        @task(name="greet", prompt="Produce a greeting")
        class GreetTask:

            @step(order=1, prompt="Say hello")
            async def say_hello(ctx):
                return f"Hello, {ctx.input}!"


async def main():
    engine = FlowForge.compile(HelloAgent)
    result = await engine.run("World")
    print(result)   # Hello, World!

asyncio.run(main())
```

Run it:
```bash
python hello_agent.py
# Hello, World!
```

---

## Validate & Visualize via CLI

```bash
# Validate the DAG (cycle detection, order uniqueness)
flowforge validate hello_agent.py

# Print Mermaid diagram
flowforge viz hello_agent.py --mermaid

# Run with execution trace
flowforge run hello_agent.py -q "World" --trace
```

---

## What just happened?

When Python imports your file, each decorator runs bottom-up:

```
1. @step   → attaches StepMeta to say_hello.__flowforge_step_meta__
2. @task   → scans class dict, finds step, builds TaskMeta
3. @flow   → scans class dict, finds task, builds FlowMeta
4. @global_config → finds flow, builds GlobalMeta
```

Then `FlowForge.compile()` converts that metadata tree into a networkx DAG and validates it. After that, `engine.run()` traverses the DAG and executes each node in topological order.

---

## Next Steps

| I want to… | Go to |
|-----------|-------|
| Understand the annotation hierarchy | [Concepts → Annotations](concepts/annotations.md) |
| Learn how data moves between steps | [Concepts → Data Flow](concepts/data-flow.md) |
| Build a real agent with branching | [Guides → First Agent](guides/first-agent.md) |
| Visualize a run | [Guides → Run Visualization](guides/visualization.md) |
| Use the CLI | [API → CLI](api/cli.md) |
