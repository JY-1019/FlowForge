# FlowForge

**Annotation-Based AI Agent Framework**

Build production-grade AI agent pipelines using nothing but Python decorators.
No graphs. No YAML. No LangChain dependency.

---

```python
from flowforge import global_config, flow, task, step, FlowForge
from pydantic import BaseModel

class Query(BaseModel):
    text: str
    lang: str = "en"

@global_config(prompt="You are a research assistant.")
class ResearchAgent:

    @flow(name="research", prompt="Analyze and answer the user query")
    class ResearchFlow:

        @task(name="analyze", prompt="Extract intent and keywords")
        class AnalyzeTask:

            @step(order=1, prompt="Classify the query intent")
            async def classify(ctx): ...

            @step(order=2, prompt="Extract key concepts")
            async def extract_keywords(ctx): ...

# Compile → Run
engine = FlowForge.compile(ResearchAgent)
result = await engine.run(Query(text="What is FlowForge?"))
```

---

## Why FlowForge?

<div class="grid cards" markdown>

-   :material-tag-outline: **Annotation-First**

    ---

    Your entire agent structure lives in Python decorators.
    No imperative graph construction, no separate config files.

-   :material-graph-outline: **DAG-Native**

    ---

    Decorators compile to a networkx DAG automatically.
    Cycle detection, topological sort, and I/O validation happen at import time.

-   :material-shield-check-outline: **Type-Safe I/O**

    ---

    Pydantic models enforce contracts at every annotation boundary.
    Bad data never silently passes between steps.

-   :material-eye-outline: **Run Visualization**

    ---

    After every `engine.run()`, render which nodes executed,
    how long each took, and which branch was chosen — as SVG or Mermaid.

-   :material-robot-outline: **Dual-Prompt System**

    ---

    Every annotation carries a user-written `prompt` (runtime instruction)
    and an AI-generated `doc` (planning metadata) — keeping execution and planning separate.

-   :material-console: **Built-in CLI**

    ---

    `flowforge validate`, `flowforge viz`, `flowforge run --trace` —
    inspect and execute your agents without writing any boilerplate.

</div>

---

## Installation

```bash
pip install git+https://github.com/JY-1019/FlowForge.git

# With all optional extras:
pip install "flowforge[all] @ git+https://github.com/JY-1019/FlowForge.git"
```

Requires **Python 3.11+**.

---

## Core Concepts in 60 Seconds

```
@global_config          ← top-level agent config (LLM, tools, system prompt)
  └─ @flow              ← high-level pipeline stage (nestable, supports branching)
       ├─ @flow         ← sub-pipeline (flows nest recursively)
       └─ @task         ← execution unit (nestable, supports branching)
            └─ @step    ← single action, runs in order=N (supports branching)
```

Branch dispatching is built into `@step`, `@task`, and `@flow` via optional `condition`, `branches`, and `fallback` parameters — there is no separate `@branch` decorator.

Every decorator compiles to a **DAG node** with a dotted-path ID (e.g. `global.research.analyze.classify[1]`).
At runtime the engine traverses the DAG, threads outputs into inputs, and records a full `RunTrace`.

---

## Quick Links

- [Installation & Quickstart →](getting-started.md)
- [Annotation Reference →](concepts/annotations.md)
- [I/O Data Flow →](concepts/data-flow.md)
- [Tools & LLM Calling →](guides/tools-and-llm.md)
- [Run Visualization Guide →](guides/visualization.md)
- [CLI Reference →](api/cli.md)
