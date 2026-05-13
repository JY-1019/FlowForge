# Tools & LLM Calling

How to register tools at every level of the annotation hierarchy and call the LLM from inside a step.

---

## Overview

FlowForge supports **hierarchical tool registration**: tools declared at a higher level are automatically available to all descendants.

```
@global_config(tools=[...])          ← available everywhere
  └─ @flow(tools=[...])             ← available to all tasks/steps in this flow
       └─ @task(tools=[...])        ← available to all steps in this task
            └─ @step(tools=[...])   ← available only in this step
```

Inside a step function, use `ctx.call_llm(prompt)` to make an AI API call. The global, flow, task, and step annotation prompts are assembled into the **system prompt**, and the argument to `call_llm()` becomes the **user prompt**.

Child annotations may also reference globally registered tools by name:

```python
@global_config(tools=[FunctionTool(func=fetch_url, name="web_fetch_url")])
class Agent:
    @flow(name="clone", prompt="Clone a public page", tools=["web_fetch_url"])
    class Clone:
        @task(name="inspect", prompt="Inspect the target page", tools=["web_fetch_url"])
        class Inspect:
            @step(order=1, prompt="Fetch the page with the web tool", tools=["web_fetch_url"])
            async def fetch(ctx):
                return await ctx.call_tool("web_fetch_url", url=ctx.input["url"])
```

This name-reference form is intended for generated flows and scoped prompts;
the actual tool config still lives at `@global_config`. Runtime adapters
registered in the global `ToolRegistry` are also visible to `<tool>` prompt
references and `ctx.call_tool()`.

---

## Tool Types

FlowForge provides five tool configurations:

```python
from flowforge.types import MCPServer, FunctionTool, HTTPTool, ClaudeSkill, AgentSkill

# MCP Server. input_schema is optional but helps non-MCP schema discovery.
mcp = MCPServer(
    url="https://api.example.com/mcp",
    name="search",
    description="Web search",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

# Python function
def my_func(query: str) -> str:
    return f"result for {query}"

func_tool = FunctionTool(func=my_func, name="my_func", description="Custom function")

# HTTP API. input_schema tells the LLM what JSON/query params to send.
http = HTTPTool(
    url="https://api.example.com/translate",
    name="translate",
    method="POST",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

# Claude Agent Skill (Anthropic provider only)
pptx = ClaudeSkill(name="pptx")

# Local Agent Skills SKILL.md folder (provider-neutral)
review = AgentSkill(path=".agents/skills/code-review")
```

`ClaudeSkill` is provider-native: FlowForge does not execute it locally.
When referenced as `<pptx>` in `ctx.call_llm()`, it is passed to Anthropic as
`container.skills` with the required code execution beta tool.

`AgentSkill` is provider-neutral. When referenced as `<code-review>`,
FlowForge reads the local `SKILL.md` and injects the activated instructions
into the model context. This follows the Agent Skills progressive-disclosure
shape without relying on a provider-native Skills API.

### Choosing Skill Types

| Type | Provider support | Best for | What FlowForge sends |
|------|------------------|----------|----------------------|
| `ClaudeSkill` | Anthropic only | Hosted Skills such as `pptx`, `xlsx`, `docx`, `pdf`, or Anthropic custom `skill_id`s | `container.skills` + required beta flags |
| `AgentSkill` | Anthropic, OpenAI, Google | Local standard Agent Skills folders authored as `SKILL.md` | Activated Skill instructions appended to the system prompt |

Use `ClaudeSkill` when you need Claude's native server-side Skill runtime,
especially for document-generation Skills that create downloadable files.
Use `AgentSkill` when users keep Skills locally and you want the same FlowForge
API across providers.

---

## Registering Tools

### Global Level

Tools on `@global_config` are available to **every** flow, task, and step in the agent.

```python
@global_config(
    prompt="Research assistant",
    tools=[
        MCPServer(url="https://search.example.com/mcp", name="web_search"),
        MCPServer(url="https://db.example.com/mcp", name="db_search"),
        ClaudeSkill(name="pptx"),
        AgentSkill(path=".agents/skills/code-review"),
    ]
)
class MyAgent:
    ...
```

### Flow Level

Tools on `@flow` are available to all tasks and steps **within that flow** (including nested sub-flows).

```python
@flow(
    name="data_pipeline",
    prompt="Process and transform data",
    tools=[HTTPTool(url="https://api.example.com/validate", name="validator")]
)
class DataPipeline:
    @task(name="process", prompt="process data")
    class ProcessTask:
        @step(order=1, prompt="validate input")
        async def validate(ctx):
            # ctx.merged_tools includes "validator" from the parent flow
            ...
```

If `validator` was already registered globally, the flow can scope it by name
instead: `tools=["validator"]`.

### Task Level

Tools on `@task` are available to all steps **within that task** (including child task steps).

```python
@task(
    name="analysis",
    prompt="Analyze documents",
    tools=[MCPServer(url="https://nlp.example.com/mcp", name="nlp_engine")]
)
class AnalysisTask:
    @step(order=1, prompt="extract entities")
    async def extract(ctx):
        # ctx.merged_tools includes "nlp_engine" from the parent task
        ...
```

### Step Level

Tools on `@step` are available **only in that step**.

```python
@step(
    order=2,
    prompt="translate text",
    tools=[HTTPTool(url="https://translate.example.com", name="translate")]
)
async def translate_step(ctx):
    result = await ctx.call_llm(
        "Translate '{text}' to {target_lang} using <translate>"
    )
    return {"translated": result}
```

### Claude Skill Example

```python
@global_config(
    prompt="Document automation assistant",
    llm_config=LLMConfig.for_claude(),
    tools=[ClaudeSkill(name="pptx")],
)
class MyAgent:
    @flow(name="deck", prompt="Create decks")
    class DeckFlow:
        @task(name="make", prompt="Make a presentation")
        class MakeDeck:
            @step(order=1, prompt="Generate a PowerPoint")
            async def make(ctx):
                return await ctx.call_llm(
                    "Create a 5-slide presentation from this input. <pptx>"
                )
```

Claude Skills require Anthropic's Messages API skill support. FlowForge adds
the required `code-execution-2025-08-25` and `skills-2025-10-02` beta flags
for calls that include `ClaudeSkill`.

!!! note "Generated files"
    Document Skills such as `pptx` can return server-side `file_id` values
    instead of a local file path. FlowForge includes those IDs in the text
    response so the caller can download the artifact through Anthropic's Files
    API. See `examples/claude_skill_pptx_agent.py` for a full download flow.

### Local Agent Skill Example

Create `.agents/skills/code-review/SKILL.md`:

```markdown
---
name: code-review
description: Review code changes for correctness, regressions, and missing tests.
---

Prioritize concrete bugs, behavior changes, and test gaps. Return concise
findings first.
```

Register and use it:

```python
@global_config(
    prompt="Engineering assistant",
    llm_config=LLMConfig.for_openai(),
    tools=[AgentSkill(path=".agents/skills/code-review")],
)
class MyAgent:
    @flow(name="review", prompt="Review changes")
    class ReviewFlow:
        @task(name="review", prompt="Review")
        class ReviewTask:
            @step(order=1, prompt="Review with the local Agent Skill")
            async def review(ctx):
                return await ctx.call_llm("Review this patch. <code-review>")
```

`AgentSkill(path=...)` accepts either the Skill directory or a direct
`SKILL.md` path. If `name` is omitted, FlowForge uses the directory name, so
hyphenated standard names such as `<code-review>` work naturally.

### Custom Claude Skill Proof Example

`examples/claude_skill_custom_text_agent.py` demonstrates a tiny custom Claude
Skill that returns a marker directly in the Python process output:

```text
- marker: FLOWFORGE_CUSTOM_SKILL_USED
- skill_name: flowforge-proof
```

This example is useful for checking that FlowForge is passing Skills into the
Anthropic API correctly without dealing with file downloads.

---

## Tool Merging

At runtime, `ctx.merged_tools` returns all tools available to the current step, merged in order:

1. **Global** tools (from `@global_config`)
2. **Flow** tools (from the enclosing `@flow`)
3. **Task** tools (from the enclosing `@task`)
4. **Step** tools (from the current `@step`)

```python
@step(order=1, prompt="search and translate")
async def my_step(ctx):
    all_tools = ctx.merged_tools
    # Contains tools from global + flow + task + step
    for tool in all_tools:
        print(tool.name)
```

---

## Calling the LLM: `ctx.call_llm()`

`ctx.call_llm(prompt)` is the primary way to make an AI API call from inside a step.

### Prompt Roles

| Source | Role | Description |
|--------|------|-------------|
| `@global_config`, `@flow`, `@task`, `@step` prompts | **System prompt** | Hierarchical instructions for the agent, flow, task, and current step |
| `ctx.call_llm("...")` | **User prompt** | The actual task instruction; sent as the user message |

### Template Syntax: `{variable}`

Use `{field_name}` in the user prompt to interpolate values from `ctx.input`:

```python
@step(order=1, prompt="You are a search optimizer")
async def optimize(ctx):
    # ctx.input has fields: query, language
    result = await ctx.call_llm(
        "Optimize the search query '{query}' for {language} speakers"
    )
    return {"optimized_query": result}
```

If `ctx.input` is a Pydantic model with `query="AI agents"` and `language="Korean"`, the rendered prompt becomes:

> Optimize the search query 'AI agents' for Korean speakers

**Rules:**
- `{field_name}` is replaced with the value of `ctx.input.field_name`
- Works with both Pydantic models and plain dicts
- Missing fields are left as literal text (e.g. `{unknown}` stays as `{unknown}`)

### Tool References: `<tool_name>`

Use `<tool_name>` in the user prompt to include a specific tool in the LLM call:

```python
@step(order=1, prompt="You are a research assistant")
async def research(ctx):
    result = await ctx.call_llm(
        "Find information about {topic} using <web_search> and summarize it"
    )
    return {"summary": result}
```

**What happens:**
1. `<web_search>` is parsed and removed from the prompt text
2. The tool named `web_search` is looked up in `ctx.merged_tools`
3. The tool is included in the LLM API call's `tools` parameter
4. The LLM can then use tool calling to invoke it

**Rules:**
- Multiple tools: `"Use <search> and <translate> to process"`
- Hyphenated Agent Skill names: `"Review with <code-review>"`
- Duplicate references are deduplicated
- Unknown tool names are silently skipped
- The `<...>` markers are removed from the final prompt sent to the LLM

---

## Tool-Use Loop

When tools are provided in a `call_llm()` call, FlowForge automatically runs a **multi-turn tool-use loop**. The LLM can call tools, receive results, and call more tools until it's ready to produce a final answer.

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────┐
│  LLM generates response                     │
│                                             │
│  ┌─ tool_use block? ──────────────────────┐ │
│  │ Yes → Execute tool → Feed result back  │ │
│  │       (loop continues)                 │ │
│  │                                        │ │
│  │ No  → Return final text or             │ │
│  │       structured_output                │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### How It Works

1. **Schema fetching** -- For MCP tools, FlowForge automatically connects to the MCP server and fetches the real tool schemas (`tools/list`). This gives the LLM accurate parameter names and types.
2. **LLM call** -- The user prompt + all tool definitions are sent to the LLM.
3. **Tool execution** -- When the LLM returns `tool_use` blocks, FlowForge dispatches each call to the correct handler:
    - **MCPServer** -- JSON-RPC over Streamable HTTP (MCP protocol)
    - **FunctionTool** -- Direct Python function call (sync or async)
    - **HTTPTool** -- HTTP request via httpx
4. **Result feedback** -- Tool results are sent back to the LLM as `tool_result` messages. Oversized results are truncated according to `LLMConfig.max_tool_result_chars` to keep the active context bounded.
5. **Repeat** -- Steps 2-4 repeat until the LLM returns a final text answer or a `structured_output` tool call (when `output_schema` is set). Max 25 rounds.

### Example: MCP Tool Loop

```python
playwright_nav = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_navigate",
    description="Navigate to a URL",
)
playwright_snap = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_snapshot",
    description="Get page content as text",
)

@step(
    order=2,
    prompt="Use Playwright to browse web pages and extract content.",
    output_schema=PageContent,
    tools=[playwright_nav, playwright_snap],
)
async def browse(ctx):
    # The LLM will:
    # 1. Call browser_navigate(url=...) → FlowForge executes via MCP
    # 2. Call browser_snapshot() → FlowForge executes via MCP
    # 3. Call structured_output(...) → FlowForge validates ResearchResult
    return await ctx.call_llm("""
        Navigate to '{url}' using <browser_navigate>,
        then read the page with <browser_snapshot>.
        Extract the main content.
    """)
```

### Structured Output with Tools

When `output_schema` is set on a step that also has tools, FlowForge uses a two-phase approach:

1. **Tools first** -- `tool_choice` is set to `auto`, allowing the LLM to call MCP/function/HTTP tools freely.
2. **Structured output last** -- A synthetic `structured_output` tool is injected. The LLM calls it with its final answer after all tool work is done.

This ensures the LLM can gather data from tools before being forced to produce structured output.

```python
@step(
    order=1,
    prompt="Research assistant",
    output_schema=ResearchResult,    # Forces structured output
    tools=[web_search_mcp],          # But allows tool calls first
)
async def research(ctx):
    return await ctx.call_llm("Research {topic} using <web_search>")
    # LLM flow: web_search() → web_search() → structured_output({...})
```

!!! note "Without tools"
    When `output_schema` is set but no tools are provided, `tool_choice` is forced directly to `structured_output` for a single-turn call (no loop needed).

### MCP Server Requirements

FlowForge communicates with MCP servers using the **Streamable HTTP transport** (MCP protocol version 2025-03-26):

- The server must accept POST requests with JSON-RPC payloads
- FlowForge handles the `initialize` handshake and session management automatically
- Tool schemas are fetched via `tools/list` before the first LLM call
- Tool calls are dispatched via `tools/call`

```bash
# Example: Start Playwright MCP server
npx @playwright/mcp@latest --port 8931

# The MCPServer config points to the server endpoint
MCPServer(url="http://localhost:8931/mcp", name="browser_navigate")
```

Dynamic flows can also register MCP tools at runtime when the server is
declared in `DynamicRunOptions`:

```python
options = DynamicRunOptions(
    mcp_server_commands={"playwright": ["npx", "-y", "@playwright/mcp@latest", "--port", "8931"]},
    mcp_server_urls={"playwright": "http://localhost:8931/mcp"},
    mcp_server_tools={"playwright": ["browser_navigate", "browser_snapshot"]},
)

await ctx.call_tool("mcp_start_server", server_name="playwright")
await ctx.call_tool("mcp_register_server", server_name="playwright")
result = await ctx.call_llm("Open the target URL. <browser_navigate>")
```

If the generated flow calls `mcp_register_server` first, FlowForge still checks
the declared endpoint. When a command exists and the endpoint is down,
registration starts the MCP server automatically before adding the MCP tool
configs to the current run.

### FunctionTool in the Loop

Python functions also participate in the tool-use loop:

```python
async def fetch_weather(city: str = "") -> str:
    # async or sync functions both work
    return f"Weather in {city}: 22C, sunny"

weather_tool = FunctionTool(
    func=fetch_weather,
    name="get_weather",
    description="Get current weather for a city",
)

@step(order=1, prompt="Weather assistant", tools=[weather_tool])
async def report(ctx):
    return await ctx.call_llm("What's the weather in {city}? Use <get_weather>")
```

---

## Error Handling: `on_error`

By default, when a step fails the error propagates through the task and flow, triggering retries. For pipelines where partial results are acceptable (e.g., an MCP tool is unavailable), use `on_error="skip_remaining"` on the task:

```python
@task(
    name="scrape_and_analyze",
    prompt="Scrape and analyze a web page",
    on_error="skip_remaining",  # Don't retry, return last good result
)
class ScrapeTask:

    @step(order=1, prompt="validate URL")
    async def validate(ctx):
        # If this fails → error propagates (no prior output)
        return {"url": ctx.input.url, "valid": True}

    @step(order=2, prompt="browse page", tools=[playwright_mcp])
    async def browse(ctx):
        # If this fails → return validate's output, skip step 3
        return await ctx.call_llm("Navigate to {url} using <browser_navigate>")

    @step(order=3, prompt="analyze content")
    async def analyze(ctx):
        # If step 2 failed, this never runs
        return await ctx.call_llm("Analyze: {content}")
```

| `on_error` | Behavior |
|------------|----------|
| `"raise"` (default) | Propagate error to flow → triggers retry logic |
| `"skip_remaining"` | Stop at failed step, return last successful output. If the **first** step fails, error still propagates (no prior output exists) |

---

## AI Step vs Code Step

A step does **not** have to call the LLM. The distinction is simple:

### Code Step (no AI call)

```python
@step(order=1, prompt="validate and clean input data")
async def validate(ctx):
    data = ctx.input
    if not data.get("text"):
        raise ValueError("text is required")
    return {"text": data["text"].strip(), "valid": True}
```

The annotation `prompt` still serves as documentation for the AI planner, but no API call is made. The step is pure Python code.

### AI Step (with `ctx.call_llm()`)

```python
@step(order=2, prompt="You are a summarization expert")
async def summarize(ctx):
    result = await ctx.call_llm(
        "Summarize the following text in {language}: {text}"
    )
    return {"summary": result}
```

The annotation prompts are sent as the system prompt, and the `call_llm()` argument is the user prompt.

### Mixing Both in a Task

```python
@task(name="process", prompt="Process and summarize documents")
class ProcessTask:

    @step(order=1, prompt="validate document format")
    async def validate(ctx):
        # Code step — just Python logic
        doc = ctx.input
        return {"text": doc["content"], "format": doc["type"]}

    @step(
        order=2,
        prompt="You are a document summarizer",
        tools=[MCPServer(url="https://nlp.example.com", name="nlp")]
    )
    async def summarize(ctx):
        # AI step — calls the LLM with tool access
        return await ctx.call_llm(
            "Summarize this {format} document: {text}. Use <nlp> for entity extraction."
        )
```

---

## Complete Example

```python
from flowforge import global_config, flow, task, step, FlowForge
from flowforge.types import LLMConfig, MCPServer, HTTPTool
from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str
    language: str = "en"

class SearchResult(BaseModel):
    answer: str
    sources: list[str] = []


# Global tool — available everywhere
search_mcp = MCPServer(
    url="https://search.example.com/mcp",
    name="web_search",
    description="Search the web"
)

# Task-level tool — available only in TranslateTask
translate_api = HTTPTool(
    url="https://translate.example.com/api",
    name="translate",
    description="Translate text between languages"
)


@global_config(
    prompt="You are a multilingual research assistant.",
    llm_config=LLMConfig(model="claude-sonnet-4-6"),
    tools=[search_mcp],
)
class ResearchAgent:

    @flow(name="research", prompt="Search and answer questions")
    class ResearchFlow:

        @task(name="search", prompt="Execute search")
        class SearchTask:

            @step(order=1, prompt="Validate the search query")
            async def validate(ctx):
                # Code step — no AI
                q = ctx.input
                if isinstance(q, dict):
                    return {"query": q.get("query", ""), "language": q.get("language", "en")}
                return {"query": q.query, "language": q.language}

            @step(order=2, prompt="You are a search query optimizer")
            async def optimize(ctx):
                # AI step — uses global web_search tool
                return await ctx.call_llm(
                    "Optimize '{query}' for <web_search> in {language}"
                )

        @task(
            name="format",
            prompt="Format the final answer",
            tools=[translate_api],
        )
        class FormatTask:

            @step(order=1, prompt="You format search results into readable answers")
            async def format_answer(ctx):
                # AI step — can use both global web_search AND task-level translate
                return await ctx.call_llm(
                    "Format this answer for {language} readers. "
                    "Use <translate> if the answer needs translation."
                )


# Compile and run
import asyncio

engine = FlowForge.compile(ResearchAgent)
result = asyncio.run(engine.run(SearchQuery(query="what is FlowForge?")))
```

---

## API Reference

### `ctx.merged_tools` → `list[ToolConfig]`

Returns all tools available to this step, merged from global → flow → task → step.

### `ctx.call_llm(prompt: str) → Any`

Call the LLM with a templated user prompt. When tools are referenced via `<tool_name>`, a tool-use loop automatically executes tool calls and feeds results back until the LLM produces a final answer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | User prompt with `{var}` and `<tool>` syntax |

**Returns:** The LLM response. When `output_schema` is set on the step, FlowForge validates the response into that Pydantic model before returning it to step code. Without `output_schema`, it returns plain text or parsed JSON when possible.

`stream=True` is for plain text streaming only. It is rejected when the step declares `output_schema`, because streaming cannot satisfy the structured output contract.

**Template syntax:**
- `{field}` → replaced with `ctx.input.field`
- `<tool>` → tool included in the API call, marker removed from text

**Tool-use loop:** When the LLM returns `tool_use` blocks, each tool is executed (MCP, function, or HTTP) and the result is fed back. This repeats until the LLM returns text or structured output (max 25 rounds). Each tool result is capped by `LLMConfig.max_tool_result_chars` by default.

### `ctx.call_tool(tool_name: str, **kwargs) → Any`

Call a tool directly from step code. FlowForge searches annotation-scoped
tools first, then the global `ToolRegistry`. It supports `FunctionTool`,
`HTTPTool`, `MCPServer`, and registry `ToolAdapter` instances. Function tools
and adapters that declare a `ctx` parameter receive the current `StepContext`
automatically.

The return value is wrapped for ergonomic access, so dict results still support
normal key lookup while also behaving well inside strings and model prompts.

### `ctx.step_prompt` → `str`

The annotation prompt from `@step(prompt="...")`. It is one section of the hierarchical system prompt in `call_llm()`.

### `ctx.input` → `Any`

The step's input data (output of previous step, or task input for first step).
