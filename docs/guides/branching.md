# Conditional Routing (Branch Dispatching)

Branch dispatching is built into `@step`, `@task`, and `@flow` via the `condition`, `branches`, and `fallback` parameters. There is no separate `@branch` decorator.

---

## Step-Level Branching

The most common pattern — route execution within a task's step chain:

```python
from flowforge import step
from flowforge.types import BranchCondition

async def handle_premium(ctx): ...
async def handle_standard(ctx): ...
async def handle_trial(ctx): ...

@step(
    order=2,
    prompt="Route user to the correct handler based on subscription plan",
    condition=BranchCondition(field="plan", enum=["premium", "standard", "trial"]),
    branches={
        "premium":  handle_premium,
        "standard": handle_standard,
        "trial":    handle_trial,
    },
    fallback=handle_standard,
)
async def route_plan(ctx): ...
```

---

## Task-Level Branching

Route to entirely different `@task`-decorated classes:

```python
from flowforge import task
from flowforge.types import BranchCondition

@task(
    name="dispatch",
    prompt="Route to the correct processing pipeline",
    condition=BranchCondition(field="mode", enum=["fast", "slow"]),
    branches={"fast": FastTask, "slow": SlowTask},
)
class DispatchTask: ...
```

---

## Flow-Level Branching

Route to entirely different `@flow`-decorated classes:

```python
from flowforge import flow
from flowforge.types import BranchCondition

@flow(
    name="dispatch",
    prompt="Route by request type",
    condition=BranchCondition(field="type", enum=["a", "b"]),
    branches={"a": FlowA, "b": FlowB},
)
class DispatchFlow: ...
```

---

## How Routing Works

1. The `condition.field` is looked up on `ctx.input` (via `getattr` or `dict.get`)
2. The resolved value is converted to a string and matched against `branches` keys
3. If matched → that handler runs
4. If not matched → `fallback` runs (or the decorated function body if no fallback)
5. The handler's return value flows to the next `order` step

```python
class UserRequest(BaseModel):
    user_id: str
    plan: str      # "premium" | "standard" | "trial" | anything else

@step(order=1, prompt="Validate user request", output_schema=UserRequest)
async def validate(ctx): ...

@step(
    order=2,
    prompt="Route by plan",
    condition=BranchCondition(field="plan", enum=["premium", "standard", "trial"]),
    branches={"premium": handle_premium, "standard": handle_standard, "trial": handle_trial},
    fallback=handle_standard,
)
async def route_plan(ctx):
    # ctx.input is the UserRequest from step order=1
    # ctx.condition_value = ctx.input.plan  (e.g. "premium")
    # ctx.selected_branch = "premium"       (set by runner)
    ...
```

---

## Fallback Behavior

| Situation | Behavior |
|-----------|----------|
| `value` is `None` | `fallback` runs; `selected_branch = "__fallback__"` |
| `value` not in `branches` | `fallback` runs; `selected_branch = "__fallback__"` |
| `fallback=None` and no match | Decorated function body runs |

---

## Handler Context

Handlers receive a `StepContext` with branch-related attributes populated:

```python
class SearchResult(BaseModel):
    results: list[dict]
    source: str

async def handle_web(ctx):
    # ctx.input           → previous step's output
    # ctx.condition_value  → "web"
    # ctx.selected_branch  → "web"
    results = await some_web_api(ctx.input.query)
    return SearchResult(results=results, source="web")
```

---

## Branch + Step Chain

```python
@task(name="search_and_rank", prompt="Search and rank results")
class SearchAndRankTask:

    @step(order=1, prompt="Analyze query to determine best source",
          output_schema=AnalyzedQuery)
    async def analyze(ctx): ...

    @step(
        order=2,
        prompt="Route to the correct search backend",
        condition=BranchCondition(field="source_preference", enum=["web", "db"]),
        branches={"web": search_web, "db": search_db},
        fallback=search_web,
    )
    async def route_search(ctx): ...

    @step(order=3, prompt="Rank and deduplicate the search results",
          input_schema=SearchResult)
    async def rank_results(ctx):
        results: SearchResult = ctx.input   # output of whichever branch ran
        ...
```

---

## Tracing Branch Selection

After a run, the `RunTrace` records which branch was selected:

```python
result, trace = await engine.run_traced(input_data)

for node in trace.nodes:
    if node.selected_branch:
        print(f"Step '{node.name}': selected '{node.selected_branch}'")
```

Or via CLI:

```bash
flowforge run agent.py -q '...' --trace
```
