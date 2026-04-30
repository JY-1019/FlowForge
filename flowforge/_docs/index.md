# FlowForge

**Annotation-Based AI Agent Framework**

FlowForge lets you build async AI-agent pipelines with Python decorators. You
write normal classes and functions; FlowForge compiles them into a validated
DAG, executes the selected route, records a trace, and can optionally let an
LLM planner choose or generate missing flows.

```python
from flowforge import FlowForge, global_config, flow, task, step

@global_config(prompt="You are a helpful research assistant.")
class ResearchAgent:

    @flow(name="research", prompt="Analyze the query and produce an answer")
    class ResearchFlow:

        @task(name="analyze", prompt="Extract intent and keywords")
        class AnalyzeTask:

            @step(order=1, prompt="Classify the request")
            async def classify(ctx):
                return {"intent": "research", "query": ctx.input["query"]}

        @task(name="answer", prompt="Draft the final response")
        class AnswerTask:

            @step(order=1, prompt="Write a concise answer")
            async def draft(ctx):
                return {"answer": f"Result for: {ctx.input['query']}"}

engine = FlowForge.compile(ResearchAgent)
result = await engine.run({"query": "What is FlowForge?"})
```

## What You Get

<div class="grid cards" markdown>

-   :material-tag-outline: **Decorator-First Structure**

    ---

    Define agents with `@global_config`, `@flow`, `@task`, and `@step`. No
    manual graph wiring, no YAML, no LangChain dependency.

-   :material-graph-outline: **Validated DAG Compilation**

    ---

    Metadata compiles to a `networkx` DAG with stable node IDs, cycle checks,
    route resolution, and schema-aware I/O validation.

-   :material-transit-connection-variant: **Ordered, Parallel, And Unique Nodes**

    ---

    `order=None` means insertion-order sequential execution. Siblings with the
    same explicit `order` run in parallel. `unique=True` makes one node the
    exclusive runner for that order group.

-   :material-source-branch: **Branch Dispatching Without `@branch`**

    ---

    Add `condition`, `branches`, and optional `fallback` to `@step`, `@task`,
    or `@flow`. There is no public `@branch` decorator.

-   :material-tools: **Tools, Skills, And LLM Calls**

    ---

    Register MCP servers, HTTP tools, Python functions, Claude Skills, and
    local Agent Skills. Steps can call `ctx.call_tool(...)` or
    `ctx.call_llm(...)`.

-   :material-map-marker-path: **Route And Planner Execution**

    ---

    Run the whole DAG, a specific route like `"research.answer"`, or use
    `planning_mode="autonomous"` / `"hybrid"` after generating docs.

-   :material-creation-outline: **Dynamic Flow Generation**

    ---

    With `dynamic_flow=True`, the planner can generate missing FlowForge code,
    inject it into the live DAG, and persist it for reuse.

-   :material-eye-outline: **Run Trace Visualization**

    ---

    Every run records executed nodes, status, duration, selected branches, and
    checkpoints. Render the result as Mermaid, Graphviz, SVG, PNG, or a table.

</div>

## Mental Model

```text
@global_config              agent-wide prompt, model, tools, dynamic settings
  └─ @flow                  high-level pipeline stage; flows can nest
       ├─ @flow             sub-pipeline
       └─ @task             work unit; tasks can nest
            └─ @step        async function; ordered atomic action
```

Every node returns a value. That value becomes the next node's input. Pydantic
schemas can validate boundaries when you provide `input_schema` and
`output_schema`.

## Start Here

- [Installation & Quickstart](getting-started.md)
- [Your First Agent](guides/first-agent.md)
- [Annotations In Depth](concepts/annotations.md)
- [Data Flow & I/O](concepts/data-flow.md)
- [Branch Dispatching](guides/branching.md)
- [Tools & LLM Calling](guides/tools-and-llm.md)
- [Dynamic Flow Generation](guides/dynamic-flow.md)
- [Engine & CompiledAgent](api/engine.md)
