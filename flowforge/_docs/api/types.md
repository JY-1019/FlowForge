# Types Reference

All types are importable from `flowforge.types` or `flowforge`.

---

## LLMConfig

```python
from flowforge.types import LLMConfig

config = LLMConfig(
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=4096,
    api_key=None,   # falls back to ANTHROPIC_API_KEY env var
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model identifier |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0–1.0) |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `api_key` | `str \| None` | `None` | API key (uses env var if None) |

---

## BranchCondition

```python
from flowforge.types import BranchCondition

condition = BranchCondition(
    field="format",              # attribute/key to inspect on ctx.input
    enum=["json", "csv", "xml"], # valid values
)
```

| Field | Type | Description |
|-------|------|-------------|
| `field` | `str` | Attribute name (for Pydantic models) or dict key |
| `enum` | `list[str]` | Valid values that trigger branch routing |

---

## MCPServer

```python
from flowforge.types import MCPServer

mcp = MCPServer(
    url="https://api.example.com/mcp",
    name="example_api",
    description="Example MCP server",
)
```

---

## FunctionTool

```python
from flowforge.types import FunctionTool

def my_tool(query: str) -> str:
    return f"Result for: {query}"

tool = FunctionTool(
    name="my_search",
    description="Search for information",
    func=my_tool,
)
```

---

## HTTPTool

```python
from flowforge.types import HTTPTool

http_tool = HTTPTool(
    name="weather_api",
    url="https://api.weather.com/v1/current",
    method="GET",
    description="Get current weather data",
)
```

---

## ToolConfig (Union)

`ToolConfig = MCPServer | FunctionTool | HTTPTool`

Used in `@global_config(tools=[...])`:

```python
@global_config(
    prompt="...",
    tools=[
        MCPServer("https://search.api.com/mcp"),
        FunctionTool(name="compute", func=my_compute_fn, description="..."),
    ]
)
class MyAgent: ...
```
