# Types Reference

Core public types such as `LLMConfig`, `BranchCondition`, `MCPServer`,
`ClaudeSkill`, `AgentSkill`, `DynamicRunOptions`, and `DependencyPolicy` are
re-exported from `flowforge`. Tool-specific classes are always available from
`flowforge.types`.

---

## LLMConfig

```python
from flowforge.types import LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-sonnet-4-6",
    temperature=0.3,
    max_tokens=4096,
    api_key=None,
    base_url=None,
    verify_ssl=True,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `"anthropic" \| "openai" \| "google"` | `"anthropic"` | LLM backend |
| `model` | `str` | `"claude-sonnet-4-6"` | Model identifier |
| `temperature` | `float` | `0.3` | Sampling temperature |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `api_key` | `str \| None` | `None` | Provider API key; SDK env vars are used when omitted |
| `base_url` | `str \| None` | `None` | Custom base URL, useful for OpenAI-compatible endpoints |
| `verify_ssl` | `bool` | `True` | Whether provider HTTP clients verify SSL certificates |

Convenience constructors:

```python
LLMConfig.for_claude()
LLMConfig.for_openai(model="gpt-4o")
LLMConfig.for_gemini(model="gemini-2.0-flash")
```

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
    headers={"Authorization": "Bearer ..."},  # optional
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

## ClaudeSkill

Anthropic-native Claude Agent Skill. Unlike `FunctionTool`, `MCPServer`, or
`HTTPTool`, this is not executed by FlowForge's local tool loop. When selected
with `<skill_name>` inside `ctx.call_llm()`, FlowForge attaches it to the
Anthropic Messages API request via `container.skills` and enables the required
code execution beta tool.

```python
from flowforge import ClaudeSkill

pptx = ClaudeSkill(name="pptx")  # Anthropic-managed PowerPoint Skill

custom = ClaudeSkill(
    name="finance_model",
    type="custom",
    skill_id="skill_01AbCdEfGhIjKlMnOpQrStUv",
    version="latest",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `""` | FlowForge reference name used in `<name>` prompts |
| `skill_id` | `str` | `""` | Anthropic Skill ID. Defaults to `name` when omitted |
| `type` | `"anthropic" \| "custom"` | `"anthropic"` | Skill source |
| `version` | `str` | `"latest"` | Skill version passed to Claude |
| `description` | `str` | `""` | Optional local description |

!!! note
    Document-generation Claude Skills such as `pptx` may create files inside
    Claude's container and return `file_id` values. FlowForge surfaces those
    IDs in the text response; examples can then download the files via
    Anthropic's Files API.

---

## AgentSkill

Provider-neutral local Agent Skill loaded from a standard `SKILL.md` folder.
When selected with `<skill-name>` inside `ctx.call_llm()`, FlowForge reads the
local `SKILL.md` and appends the activated Skill instructions to the model
context. This works with Anthropic, OpenAI, and Google providers because it is
prompt-based rather than provider-native.

```python
from flowforge import AgentSkill

code_review = AgentSkill(path=".agents/skills/code-review")
```

Expected local layout:

```text
.agents/skills/code-review/
└── SKILL.md
```

`SKILL.md` may include standard frontmatter:

```markdown
---
name: code-review
description: Review code changes for regressions and missing tests.
---

Prioritize concrete bugs, behavior changes, and test gaps.
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | required | Skill directory or direct `SKILL.md` path |
| `name` | `str` | directory name | FlowForge reference name used in `<name>` prompts. Hyphenated names like `<code-review>` are supported |
| `description` | `str` | `""` | Optional description override |
| `max_chars` | `int` | `12000` | Maximum instruction body characters injected into the prompt |

---

## ToolConfig And ToolReference

`ToolConfig = MCPServer | FunctionTool | HTTPTool | ClaudeSkill | AgentSkill`

`ToolReference = ToolConfig | str`

Used in `@global_config(tools=[...])`:

```python
@global_config(
    prompt="...",
    tools=[
        MCPServer("https://search.api.com/mcp"),
        FunctionTool(name="compute", func=my_compute_fn, description="..."),
        ClaudeSkill(name="pptx"),
        AgentSkill(path=".agents/skills/code-review"),
    ]
)
class MyAgent: ...
```

Child `@flow`, `@task`, and `@step` annotations can reference a globally
registered tool by string name:

```python
@flow(name="fetch", prompt="Fetch public data", tools=["web_fetch_url"])
class FetchFlow:
    @task(name="load", prompt="Load URL through the web tool", tools=["web_fetch_url"])
    class Load:
        @step(order=1, prompt="Call the web_fetch_url tool", tools=["web_fetch_url"])
        async def fetch(ctx):
            return await ctx.call_tool("web_fetch_url", url=ctx.input["url"])
```

String references are resolved against concrete tool configs already present
in the global/parent tool chain. Dynamic generated flows use this form to
declare intended tool scope without rebuilding `FunctionTool` objects.

Use a Claude Skill from a step with the same angle-bracket syntax as other
LLM tools:

```python
@step(order=1, prompt="Create a presentation")
async def make_deck(ctx):
    return await ctx.call_llm("Create a deck from this report. <pptx>")
```

Use a local Agent Skill from a step:

```python
@step(order=1, prompt="Review code")
async def review(ctx):
    return await ctx.call_llm("Review this change. <code-review>")
```

---

## DynamicRunOptions

Controls dynamic flow generation. You can pass it to both
`FlowForge.compile()` and `engine.run()`.

```python
from flowforge import DynamicRunOptions
from flowforge.types import DependencyPolicy

options = DynamicRunOptions(
    project_root=".",
    generated_dir="generated",
    persist_generated=True,
    auto_load_generated=True,
    include_builtin_tools=True,
    generated_step_timeout_seconds=300,
    dependency_policy=DependencyPolicy(allow_install=True),
)

engine = FlowForge.compile(MyAgent, dynamic_options=options)
result = await engine.run(input_data, dynamic_options=options)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable or disable dynamic generation. If `False`, gaps are not generated |
| `project_root` | `str \| None` | `None` | Project root. `None` means `cwd()` |
| `generated_dir` | `str` | `"flowforge/generated"` | Directory for generated code. Must stay under `project_root` |
| `persist_generated` | `bool` | `True` | Save generated code and update `manifest.json` |
| `auto_load_generated` | `bool` | `True` | Load previously generated flows from `manifest.json` at compile time |
| `include_builtin_tools` | `bool` | `True` | Enable the built-in tool pack for web, JSON, files, and documents |
| `allow_tool_generation` | `bool` | `False` | Allow generation of new `FunctionTool` code when needed |
| `allow_codegen_tool_use` | `bool` | `False` | Allow generated code to use LLM tool selection |
| `generated_step_timeout_seconds` | `int` | `300` | Minimum timeout applied to steps in generated dynamic flows |
| `allowed_shell_modes` | `list[str]` | `["readonly", "project_exec"]` | Shell modes: `"readonly"`, `"workspace_write"`, `"project_exec"`, `"install_dependency"` |
| `shell_timeout_seconds` | `int` | `60` | Shell execution timeout |
| `shell_output_max_chars` | `int` | `4000` | Maximum captured shell output |
| `mcp_server_commands` | `dict[str, list[str]]` | `{}` | MCP server command map available during dynamic generation |
| `mcp_server_urls` | `dict[str, str]` | `{}` | MCP server endpoint URLs keyed by server name |
| `mcp_server_tools` | `dict[str, list[str]]` | `{}` | Known MCP tool names keyed by server name |
| `mcp_server_headers` | `dict[str, dict[str, str]]` | `{}` | Optional MCP request headers keyed by server name |
| `mcp_start_timeout_seconds` | `int` | `15` | MCP server startup timeout |
| `project_context_max_chars` | `int` | `4000` | Maximum project context characters included in codegen prompts |
| `codegen_tool_catalog_max_tools` | `int` | `12` | Maximum relevant tools included in codegen prompts |
| `codegen_tool_catalog_max_chars` | `int` | `6000` | Maximum characters for the codegen tool catalog |
| `max_requirements` | `int` | `8` | Maximum number of planner gap requirements |
| `dependency_policy` | `DependencyPolicy` | default | Package installation policy |

When `allow_codegen_tool_use=False`, executable tools are not passed to the
code generation LLM. Prompt-only skills such as `ClaudeSkill` and `AgentSkill`
may still be attached to guide code generation.

**Caching behavior:**

- `persist_generated=True` + `auto_load_generated=True` (default): generated flows are reused across processes
- Within the same session, the DAG and manifest are checked to avoid regeneration
- Before generating, FlowForge checks both the current DAG and `manifest.json` for a flow with the same name

See the [Dynamic Flow Generation Guide](../guides/dynamic-flow.md#dynamicrunoptions) for details.

---

## DependencyPolicy

Controls package installation for generated code. Pass it through
`DynamicRunOptions.dependency_policy`.

```python
from flowforge.types import DependencyPolicy

policy = DependencyPolicy(
    allow_install=True,                 # allow the pip_install tool
    allowed_managers=["pip", "uv"],
    allowed_packages=["httpx"],         # allowlist; empty means allow all
    denied_packages=["subprocess"],     # denylist; always wins
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_install` | `bool` | `False` | Whether the `pip_install` tool may be used |
| `allowed_managers` | `list[str]` | `["pip", "uv", "npm", "pnpm", "yarn"]` | Dependency managers dynamic code may request |
| `allowed_packages` | `list[str]` | `[]` | Package allowlist. Empty means all packages are allowed |
| `denied_packages` | `list[str]` | `[]` | Package denylist. Takes precedence over the allowlist |
