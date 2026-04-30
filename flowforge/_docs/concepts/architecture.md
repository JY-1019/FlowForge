# Architecture Overview

FlowForge runs through four main phases:

```text
Decorate -> Compile -> Plan -> Execute
```

When `dynamic_flow=True`, dynamic generation can run between planning and
execution, or as a recovery path after a planner-selected route fails because
a needed upstream capability is missing.

## 1. Decorate

Python applies inner decorators before outer decorators:

```python
@global_config(prompt="agent")       # 4. scans root flows
class Agent:
    @flow(name="f", prompt="flow")   # 3. scans child flows/tasks
    class F:
        @task(name="t", prompt="task")  # 2. scans steps/child tasks
        class T:
            @step(order=1, prompt="step")  # 1. attaches StepMeta
            async def s(ctx): ...
```

Each decorator attaches metadata to the original class or function:

| Decorator | Metadata | Attribute |
|-----------|----------|-----------|
| `@step` | `StepMeta` | `__flowforge_step_meta__` |
| `@task` | `TaskMeta` | `__flowforge_task_meta__` |
| `@flow` | `FlowMeta` | `__flowforge_flow_meta__` |
| `@global_config` | `GlobalMeta` | `__flowforge_global_meta__` |

There is no `BranchMeta`. A branching node is still a `StepMeta`,
`TaskMeta`, or `FlowMeta` with `condition` set.

## 2. Compile

`FlowForge.compile(MyAgent)` traverses the metadata tree and builds a
`FlowForgeDAG`.

```text
GlobalMeta
  -> FlowMeta
     -> FlowMeta
     -> TaskMeta
        -> TaskMeta
        -> StepMeta
```

The compiler emits:

| Item | Meaning |
|------|---------|
| `DAGNode` | One node per global, flow, task, or step |
| `parent_child` edge | Containment relation |
| `depends_on` edge | Explicit flow dependency |

Node IDs use a dotted path:

```text
global
global.research
global.research.answer
global.research.answer.draft[1]
```

## 3. Validate

Validation catches structural errors early:

| Check | Error |
|-------|-------|
| missing `@global_config` | `CompileError` |
| `depends_on` cycle | `CycleDetectedError` |
| same-order duplicate `unique=True` | `OrderConflictError` |
| incompatible consecutive schemas | `IOBindingError` |
| inconsistent branching step handler return annotations | `BranchOutputMismatchError` |

Same explicit `order` is allowed and means parallel execution. The conflict is
only multiple `unique=True` nodes in the same order group.

## 4. Plan

Deterministic mode runs the compiled order. Autonomous and hybrid modes use
generated docs:

```python
engine = FlowForge.compile(MyAgent)
await engine.generate_docs(planning_only=True)
result = await engine.run(data, planning_mode="autonomous")
```

The planner receives flow-level doc summaries and chooses the root flow routes
to execute. Passing `route=...` skips planning and runs the explicit route.

## 5. Dynamic Generation

When enabled:

```python
@global_config(prompt="agent", dynamic_flow=True)
class Agent: ...

engine = FlowForge.compile(Agent, dynamic_options=DynamicRunOptions(...))
```

If the planner reports a gap, FlowForge can:

1. Build a dynamic generation request from the user input and planner result.
2. Generate FlowForge decorator code.
3. Compile and inject the new flow into the live DAG.
4. Persist generated code and manifest entries when configured.
5. Re-plan and execute the expanded DAG.

## 6. Execute

Execution creates context objects and runs root flows:

```text
GlobalContext
  -> FlowRunner
     -> TaskRunner
        -> StepRunner
```

Each run records a `RunTrace` with executed node IDs, status, timing, branch
selection, errors, and checkpoints.

## Sessions And Memory

`CompiledAgent` owns a default engine and memory for simple single-user usage.
For servers, create isolated sessions:

```python
agent = FlowForge.compile(MyAgent)
session = agent.create_session()
result = await session.run(user_input)
```

Sessions share the compiled DAG and docs but have separate memory and traces.
