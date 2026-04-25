# FlowForge — Input / Output Flow

This document explains how data moves across FlowForge annotation boundaries:
`@global_config -> @flow -> @task -> @step`. Branch dispatching is implemented
as parameters on `@step`, `@task`, and `@flow`; there is no separate public
`@branch` decorator in the current API.

---

## 1. End-To-End Flow

```text
engine.run(user_input)
      |
      v
FlowRunner.run(flow_meta, global_ctx, input=user_input)
      |
      | current_output starts as flow_input
      |
      +--> child flows, if present
      |       previous child output -> next child input
      |
      +--> tasks, if present
              previous task output -> next task input
                    |
                    v
             TaskRunner.run(task_meta, flow_ctx, input=current_output)
                    |
                    +--> leaf task: ordered step chain
                    |       order 1 output -> order 2 input -> ...
                    |
                    +--> container task: child task chain
                            child task 1 output -> child task 2 input -> ...
```

Every runner returns the latest output it produced. That returned value becomes
the input to the next sibling node in the same execution group.

---

## 2. Annotation-Level I/O Contracts

### 2.1 `@global_config`

`@global_config` does not consume or emit data by itself. It declares global
metadata: system prompt, default LLM config, tools, dynamic-generation settings,
and root flows.

The value passed to `engine.run(input_data)` becomes the input of the first
root flow.

```python
@global_config(prompt="...")
class MyAgent:
    @flow(name="first_flow", prompt="...")
    class FirstFlow:
        ...
```

```text
engine.run("hello")
  -> FlowRunner.run(FirstFlow, global_ctx, flow_input="hello")
```

### 2.2 `@flow`

A flow is a transformation boundary. It receives a value, runs child flows and
tasks, and returns one output.

| Situation | Flow input | Flow output |
|-----------|------------|-------------|
| Root flow | The value passed to `engine.run()` | The last child/task output |
| Child flow | Parent flow input or previous sibling flow output | The last child/task output |
| Child of a parallel parent group | Parent flow input | The final result selected from the parallel group |

Conceptual runtime:

```python
current_output = flow_input

for child_flow in meta.child_flows:
    current_output = await run_flow(child_flow, current_output)

for task_meta in meta.tasks:
    current_output = await run_task(task_meta, current_output)

return current_output
```

`input_schema` and `output_schema` on `@flow` primarily document the contract
and guide compile/planning behavior. Runtime Pydantic validation is strongest
at step boundaries.

### 2.3 `@task`

Tasks are execution units inside flows. A task can be a container task or a
leaf task.

Container task:

```text
task_input
  -> child task 1
       -> output -> child task 2
            -> output -> task_output
```

Leaf task:

```text
task_input
  -> step order=1
       -> output -> step order=2
            -> output -> task_output
```

Conceptual runtime:

```python
current_input = task_input

for node_meta in sorted(meta.steps, key=lambda s: s.order):
    current_input = await step_runner.run(node_meta, task_ctx, current_input)

return current_input
```

When multiple siblings share the same explicit `order`, they run as a parallel
group and receive the same input. If one sibling is marked `unique=True`, only
that sibling runs for that order group.

### 2.4 `@step`

A step is the smallest executable unit. It receives `ctx.input` and returns the
value that will be passed to the next step or sibling node.

```python
@step(order=1, prompt="Validate the input")
async def validate(ctx):
    data = ctx.input
    return {"valid": True, "data": data}
```

`StepContext` exposes:

| Attribute | Description |
|-----------|-------------|
| `ctx.input` | Output from the previous node, or the task input for the first step |
| `ctx.step_prompt` | Prompt declared on the `@step` decorator |
| `ctx.previous_results` | Prior results in the current task, keyed by order |
| `ctx.merged_tools` | Tools merged from global -> flow -> task -> step |
| `ctx.llm_config` | Effective LLM configuration |
| `ctx.task_ctx`, `ctx.flow_ctx`, `ctx.global_ctx` | Parent contexts |

Runtime validation:

```text
incoming step_input
  -> validate with input_schema, if present
  -> create StepContext(ctx.input=validated_input)
  -> await step function
  -> validate result with output_schema, if present
  -> record in task_ctx.step_results
  -> return result
```

### 2.5 Branch Dispatch

Branch dispatch uses `condition`, `branches`, and `fallback` on `@step`,
`@task`, or `@flow`.

```python
@step(
    order=2,
    prompt="Route by source",
    condition=BranchCondition(field="source", enum=["web", "db"]),
    branches={"web": web_handler, "db": db_handler},
    fallback=web_handler,
)
async def route(ctx):
    ...
```

The runner:

1. validates the input schema if one is declared;
2. reads `condition.field` from the input;
3. selects `branches[str(value)]` when it exists;
4. otherwise uses `fallback` when provided;
5. otherwise executes the decorated function itself;
6. validates the output schema if one is declared;
7. records the result for later steps.

During a branch handler call, `StepContext.selected_branch` and
`StepContext.condition_value` are populated.

All branch handlers should return compatible output types so the next node can
consume the result consistently.

---

## 3. Data Flow Diagram

```text
engine.run(UserQuery)
|
| UserQuery ----------------------------------------------------------> flow_input
|                                                                        |
|                         FlowRunner("research")                         |
|                         current_output = flow_input                    |
|                                                                        |
|   child flow / task output --------------------------------------------+
|                                                                        |
|                         TaskRunner("analyze")                          |
|                         current_input = task_input                     |
|                                                                        |
|   step[1] output -> step[2] input -> step[3] input -> task_output       |
|                                                                        |
+---------------------------------------------------------------------> result
```

---

## 4. Schema Validation Boundaries

Runtime validation happens when a step declares an `input_schema` or
`output_schema`.

```python
class RawDoc(BaseModel):
    text: str


class CleanDoc(BaseModel):
    text: str
    word_count: int


@step(order=1, prompt="Clean input", input_schema=RawDoc, output_schema=CleanDoc)
async def clean(ctx):
    text = ctx.input.text.strip()
    return {"text": text, "word_count": len(text.split())}
```

If validation fails, the runner wraps the failure in `ExecutionError` and the
task/flow retry policy decides what happens next.

---

## 5. LLM Calls And Tool References

`ctx.call_llm(prompt)` renders `{field}` placeholders from `ctx.input`, removes
`<tool-name>` markers, resolves the referenced tools from `ctx.merged_tools`,
and calls the configured provider.

```python
@step(order=1, prompt="You are a research assistant")
async def research(ctx):
    return await ctx.call_llm(
        "Research {topic} with <web_search> and summarize the findings."
    )
```

Tool references can point to:

- `FunctionTool`;
- `HTTPTool`;
- `MCPServer`;
- `ClaudeSkill`;
- `AgentSkill`.

`ClaudeSkill` is sent through Anthropic's native `container.skills` mechanism.
`AgentSkill` loads a local `SKILL.md` into the system prompt and works across
providers.

---

## 6. Parallel Order Groups

Siblings with the same explicit `order` are treated as a parallel group.
Each node receives the same input, and the group result is forwarded after the
group completes.

```python
@task(name="parallel_fetch", prompt="Fetch from several sources")
class ParallelFetch:
    @step(order=1, prompt="Fetch API A")
    async def fetch_a(ctx): ...

    @step(order=1, prompt="Fetch API B")
    async def fetch_b(ctx): ...

    @step(order=2, prompt="Merge results")
    async def merge(ctx): ...
```

`unique=True` can be used within a same-order group when exactly one sibling
should run.

---

## 7. Looping Tasks

Tasks can re-run their leaf step chain until `loop_condition(output)` returns
`True`, or until `max_loops` is exhausted.

```python
@task(
    name="retry_until_valid",
    prompt="Produce valid output",
    max_loops=5,
    loop_condition=lambda output: output.get("valid", False),
)
class RetryTask:
    ...
```

The loop condition receives the task output. `True` accepts the output and
stops looping. `False` discards it and re-runs the task chain.
