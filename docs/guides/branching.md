# Conditional Routing with @branch

`@branch` lets you route execution to different handlers based on a field value in the input.

---

## Basic Branch

```python
from flowforge.types import BranchCondition

async def handle_premium(ctx): ...
async def handle_standard(ctx): ...
async def handle_trial(ctx): ...

@branch(
    order=2,
    name="plan_router",
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

@branch(order=2, name="plan_router", ..., fallback=handle_standard)
async def route_plan(ctx):
    # ctx.input is the UserRequest from step order=1
    # ctx.condition_value = ctx.input.plan  (e.g. "premium")
    # ctx.selected_branch = "premium"       (set by BranchRunner)
    ...
```

---

## Fallback Behavior

| Situation | Behavior |
|-----------|----------|
| `value` is `None` | `fallback` runs; `selected_branch = "__fallback__"` |
| `value` not in `branches` | `fallback` runs; `selected_branch = "__fallback__"` |
| `fallback=None` and no match | Decorated branch body runs |

---

## Handler Context

Handlers receive the same `BranchContext` as the decorator's own body:

```python
class SearchResult(BaseModel):
    results: list[dict]
    source: str

async def handle_web(ctx):
    # ctx.input          → previous step's output
    # ctx.condition_value → "web"
    # ctx.selected_branch → "web"
    # ctx.task_ctx        → parent TaskContext
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

    @branch(
        order=2, name="source_router",
        condition=BranchCondition(field="source_preference",
                                  enum=["web", "db"]),
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
    if node.node_type == "branch":
        print(f"Branch '{node.name}': selected '{node.selected_branch}'")

# Branch 'source_router': selected 'web'
# Branch 'plan_router':   selected '__fallback__'
```

Or via CLI:

```bash
flowforge run agent.py -q '...' --trace
```

Output includes a `branch` column showing the selected arm.
