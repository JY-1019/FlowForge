# Decorator API Reference

---

## @global_config

```python
@global_config(
    prompt: str,
    llm_config: LLMConfig | None = None,
    tools: list[ToolReference] | None = None,
    dynamic_flow: bool = False,
    include_builtin_tools: bool = False,
)
class MyAgent: ...
```

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `prompt` | `str` | Yes | System-level prompt prepended to every LLM call |
| `llm_config` | `LLMConfig` | | Default model/temperature/token settings |
| `tools` | `list[ToolReference]` | | Global tool registrations (MCP, function, HTTP, Claude Skill, Agent Skill). Available to **all** flows, tasks, and steps. |
| `dynamic_flow` | `bool` | | Enable dynamic flow generation. Injects the internal `_dynamic_generator` meta-flow. See [Dynamic Flow Guide](../guides/dynamic-flow.md). |
| `include_builtin_tools` | `bool` | | Add FlowForge builtin tools to the global tool list. |

Attaches `GlobalMeta` to the class as `cls.__flowforge_global_meta__`.

---

## @flow

```python
@flow(
    name: str,
    prompt: str,
    input_schema: Type[BaseModel] | None = None,
    output_schema: Type[BaseModel] | None = None,
    depends_on: list[str] = [],
    parallel: bool = False,
    max_retries: int = 3,
    order: int | None = None,
    unique: bool = False,
    tools: list[ToolReference] | None = None,
    condition: BranchCondition | None = None,
    branches: dict[str, type] | None = None,
    fallback: type | None = None,
)
class MyFlow: ...
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique name within parent scope |
| `prompt` | `str` | *(required)* | Role description used at runtime |
| `input_schema` | `Type[BaseModel]` | `None` | Validates flow input at execution time |
| `output_schema` | `Type[BaseModel]` | `None` | Validates flow output at execution time |
| `depends_on` | `list[str]` | `[]` | Names of flows that must finish before this one |
| `parallel` | `bool` | `False` | Legacy flag: execute immediate child flows concurrently |
| `max_retries` | `int` | `3` | Number of retries on `ExecutionError` with backoff |
| `order` | `int \| None` | `None` | Execution position within parent. Same order = parallel |
| `unique` | `bool` | `False` | Sole representative of its order group |
| `tools` | `list[ToolReference]` | `None` | Tool configs or globally registered tool names available to descendants |
| `condition` | `BranchCondition` | `None` | Branch condition (turns flow into dispatcher) |
| `branches` | `dict[str, type]` | `None` | `{value: FlowClass}` mapping for branching |
| `fallback` | `type` | `None` | Default branch when no key matches |

Attaches `FlowMeta` to the class as `cls.__flowforge_flow_meta__`.

---

## @task

```python
@task(
    name: str,
    prompt: str,
    input_schema: Type[BaseModel] | None = None,
    output_schema: Type[BaseModel] | None = None,
    order: int | None = None,
    unique: bool = False,
    tools: list[ToolReference] | None = None,
    on_error: str = "raise",
    max_loops: int = 1,
    loop_condition: Callable[..., bool] | None = None,
    condition: BranchCondition | None = None,
    branches: dict[str, type] | None = None,
    fallback: type | None = None,
)
class MyTask: ...
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique name within parent scope |
| `prompt` | `str` | *(required)* | Task role description |
| `input_schema` | `Type[BaseModel]` | `None` | Input validation |
| `output_schema` | `Type[BaseModel]` | `None` | Output validation |
| `order` | `int \| None` | `None` | Execution position within parent. Same order = parallel |
| `unique` | `bool` | `False` | Sole representative of its order group |
| `tools` | `list[ToolReference]` | `None` | Tool configs or globally registered tool names available to child tasks/steps |
| `on_error` | `str` | `"raise"` | `"raise"` propagates errors; `"skip_remaining"` returns the last successful step output |
| `max_loops` | `int` | `1` | Maximum attempts for the task step chain |
| `loop_condition` | `Callable \| None` | `None` | `(output) -> bool`; `True` accepts the result |
| `condition` | `BranchCondition` | `None` | Branch condition (turns task into dispatcher) |
| `branches` | `dict[str, type]` | `None` | `{value: TaskClass}` mapping for branching |
| `fallback` | `type` | `None` | Default branch when no key matches |

A **leaf task** contains `@step` functions directly (no child tasks).
A **container task** contains child `@task` classes (no direct steps).
A **branch task** has `condition` set and delegates to one of `branches`.

Attaches `TaskMeta` to the class as `cls.__flowforge_task_meta__`.

---

## @step

```python
@step(
    order: int,
    prompt: str,
    input_schema: Type[BaseModel] | None = None,
    output_schema: Type[BaseModel] | None = None,
    tool_mode: bool = False,
    timeout_seconds: int = 60,
    unique: bool = False,
    approval: bool = False,
    tools: list[ToolReference] | None = None,
    condition: BranchCondition | None = None,
    branches: dict[str, Callable] | None = None,
    fallback: Callable | None = None,
    pass_criteria: str | None = None,
    pass_criteria_max_retries: int = 3,
)
async def my_step(ctx: StepContext) -> Any: ...
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `order` | `int` | *(required)* | Execution slot within the task. Same order = parallel group |
| `prompt` | `str` | *(required)* | What this step does. Included in the hierarchical **system prompt** when `ctx.call_llm()` is called |
| `input_schema` | `Type[BaseModel]` | `None` | Coerces and validates input before calling function |
| `output_schema` | `Type[BaseModel]` | `None` | Coerces and validates return value |
| `tool_mode` | `bool` | `False` | Metadata flag for tool-oriented steps; current runners still execute ordered steps normally |
| `timeout_seconds` | `int` | `60` | Wall-clock timeout; raises `asyncio.TimeoutError` |
| `unique` | `bool` | `False` | Sole representative of its order group |
| `approval` | `bool` | `False` | Pause before execution and raise `ApprovalRequired` with a checkpoint |
| `tools` | `list[ToolReference]` | `None` | Tool configs or globally registered tool names available **only in this step** |
| `condition` | `BranchCondition` | `None` | Branch condition (turns step into dispatcher) |
| `branches` | `dict[str, Callable]` | `None` | `{value: handler}` mapping for branching |
| `fallback` | `Callable` | `None` | Default handler when no key matches |
| `pass_criteria` | `str \| None` | `None` | LLM-judged criteria for accepting the output |
| `pass_criteria_max_retries` | `int` | `3` | Retry attempts when criteria fail |

### StepContext Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ctx.input` | `Any` | Previous step's output (or task input if order=1) |
| `ctx.step_prompt` | `str` | The step annotation prompt (one section of the `call_llm` system prompt) |
| `ctx.tools` | `ToolRegistry` | Global tool registry |
| `ctx.merged_tools` | `list[ToolConfig]` | All tools merged from global → flow → task → step |
| `ctx.previous_results` | `dict[int, Any]` | All prior step results keyed by order |
| `ctx.task_ctx` | `TaskContext` | Parent task context |
| `ctx.flow_ctx` | `FlowContext` | Parent flow context |
| `ctx.global_ctx` | `GlobalContext` | Global context |
| `ctx.llm_config` | `LLMConfig` | LLM configuration for this run |
| `ctx.condition_value` | `Any` | Resolved condition value (branch steps only) |
| `ctx.selected_branch` | `str` | Selected branch key (branch steps only) |
| `ctx.pass_criteria_feedback` | `list` | Feedback from previous failed criteria checks |

### StepContext Methods

#### `await ctx.call_llm(prompt: str) → Any`

Call the LLM with a templated user prompt.

- Global, flow, task, and step prompts are assembled into the **system prompt**
- `prompt` argument is the **user prompt**
- `{field}` in the prompt is replaced with `ctx.input.field`
- `<tool_name>` in the prompt includes that tool in the API call
- `<skill_name>` works the same for `ClaudeSkill`; Anthropic requests receive `container.skills`
- `<agent-skill>` works for local `AgentSkill`; FlowForge loads the matching `SKILL.md` into the model context
- Returns the LLM response content

```python
@step(order=1, prompt="You are a search assistant")
async def search(ctx):
    return await ctx.call_llm(
        "Find information about {query} using <web_search>"
    )
```

Attaches `StepMeta` to the function as `fn.__flowforge_step_meta__`.

---

## Branch Dispatching

There is **no** separate `@branch` decorator. Branch behaviour is built into `@step`, `@task`, and `@flow` via optional `condition`, `branches`, and `fallback` parameters.

### Step-level branching

```python
@step(
    order=2,
    prompt="route by source",
    condition=BranchCondition(field="source", enum=["web", "db"]),
    branches={"web": web_handler, "db": db_handler},
    fallback=web_handler,
)
async def route(ctx): ...
```

### Task-level branching

```python
@task(
    name="dispatch",
    prompt="dispatch by mode",
    condition=BranchCondition(field="mode", enum=["fast", "slow"]),
    branches={"fast": FastTask, "slow": SlowTask},
)
class DispatchTask: ...
```

### Flow-level branching

```python
@flow(
    name="dispatch",
    prompt="dispatch by type",
    condition=BranchCondition(field="type", enum=["a", "b"]),
    branches={"a": FlowA, "b": FlowB},
)
class DispatchFlow: ...
```

---

## Tool Inheritance Hierarchy

Tools registered at each level are available to all descendants:

```
@global_config(tools=[A, B])        ← A, B available everywhere
  └─ @flow(tools=[C])               ← A, B, C available in this flow
       └─ @task(tools=[D])          ← A, B, C, D available in this task
            └─ @step(tools=[E])     ← A, B, C, D, E available in this step
```

Access all merged tools via `ctx.merged_tools` inside any step.

See the [Tools & LLM Calling Guide](../guides/tools-and-llm.md) for detailed usage.
