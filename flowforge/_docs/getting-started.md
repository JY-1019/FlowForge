# Installation & Quickstart

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Pydantic | 2.7+ |
| anyio | 4.x |
| networkx | 3.3+ |

## Install

```bash
pip install git+https://github.com/JY-1019/FlowForge.git

# Optional integrations
pip install "flowforge[all] @ git+https://github.com/JY-1019/FlowForge.git"
pip install "flowforge[viz] @ git+https://github.com/JY-1019/FlowForge.git"
pip install "flowforge[openai] @ git+https://github.com/JY-1019/FlowForge.git"
pip install "flowforge[google] @ git+https://github.com/JY-1019/FlowForge.git"
```

For local development:

```bash
git clone https://github.com/JY-1019/FlowForge.git
cd FlowForge
pip install -e ".[dev]"
```

## Hello World

```python
# hello_agent.py
import asyncio
from flowforge import FlowForge, global_config, flow, task, step

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
    print(result)

asyncio.run(main())
```

Run it:

```bash
python hello_agent.py
```

Expected output:

```text
Hello, World!
```

## Add Typed I/O

```python
from pydantic import BaseModel

class Query(BaseModel):
    text: str

class Answer(BaseModel):
    text: str

@task(name="answer", prompt="Answer the question", input_schema=Query, output_schema=Answer)
class AnswerTask:
    @step(order=1, prompt="Draft an answer", input_schema=Query, output_schema=Answer)
    async def draft(ctx):
        return Answer(text=f"Answering: {ctx.input.text}")
```

FlowForge validates and coerces values at the boundaries where schemas are
provided.

## Run A Route

```python
# Run only one flow
result = await engine.run(data, route="hello")

# Run one task inside a flow
result = await engine.run(data, route="hello.greet")
```

`route` overrides `planning_mode` and is useful for debugging or exposing
specific capabilities as API endpoints.

## Generate Docs For Planning

```python
engine = FlowForge.compile(HelloAgent)
await engine.generate_docs(planning_only=True)
result = await engine.run("World", planning_mode="autonomous")
```

`planning_only=True` generates docs for global and flow nodes, which is enough
for the flow-level planner and much cheaper than documenting every step.

## Visualize

```python
print(engine.mermaid())          # full DAG as Mermaid

result, trace = await engine.run_traced("World")
print(engine.run_mermaid(trace)) # executed path
engine.print_run_summary(trace)
```

With Graphviz installed:

```python
engine.visualize("dag.svg")
engine.visualize_run("run.svg", trace)
```

## CLI

```bash
flowforge validate hello_agent.py
flowforge viz hello_agent.py --mermaid
flowforge run hello_agent.py -q "World" --trace
```
