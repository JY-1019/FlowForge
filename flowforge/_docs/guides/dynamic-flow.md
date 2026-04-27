# Dynamic Flow Generation

Dynamic flow generation lets FlowForge create missing FlowForge flows at
runtime. When `@global_config(dynamic_flow=True)` is enabled, autonomous or
hybrid planning can detect that the compiled DAG does not cover part of the
user request. FlowForge then runs an internal meta-flow that generates,
validates, optionally persists, injects, and executes the new flow.

---

## Quick Start

```python
from flowforge import DynamicRunOptions, FlowForge, global_config
from flowforge.types import LLMConfig


@global_config(
    prompt="General-purpose assistant",
    llm_config=LLMConfig.for_claude(),
    dynamic_flow=True,
)
class MyAgent:
    pass


options = DynamicRunOptions(
    project_root=".",
    generated_dir="flowforge/generated",
    persist_generated=True,
    auto_load_generated=True,
    include_builtin_tools=True,
)

engine = FlowForge.compile(MyAgent, dynamic_options=options)
result = await engine.run(
    "Make a table of the five tallest mountains in the world",
    planning_mode="autonomous",
)
```

Dynamic generation is only considered when:

- the agent is declared with `dynamic_flow=True`;
- `DynamicRunOptions.enabled` is `True`;
- the run uses `planning_mode="autonomous"` or `"hybrid"`;
- the planner reports one or more uncovered requirements.

---

## DynamicRunOptions

`DynamicRunOptions` controls generation, persistence, tool availability, shell
access, dependency installation, and project context size. You can pass it to
`FlowForge.compile()` and override it again at `engine.run()` time.

```python
from flowforge import DynamicRunOptions
from flowforge.types import DependencyPolicy

options = DynamicRunOptions(
    enabled=True,
    project_root=".",
    generated_dir="flowforge/generated",
    persist_generated=True,
    auto_load_generated=True,
    include_builtin_tools=True,
    allow_tool_generation=False,
    allow_codegen_tool_use=False,
    generated_step_timeout_seconds=300,
    allowed_shell_modes=["readonly", "project_exec"],
    shell_timeout_seconds=60,
    shell_output_max_chars=4000,
    mcp_server_commands={},
    mcp_server_urls={},
    mcp_server_tools={},
    mcp_server_headers={},
    mcp_start_timeout_seconds=15,
    project_context_max_chars=4000,
    codegen_tool_catalog_max_tools=12,
    codegen_tool_catalog_max_chars=6000,
    max_requirements=8,
    dependency_policy=DependencyPolicy(
        allow_install=False,
        allowed_packages=[],
        denied_packages=[],
    ),
)
```

### Core Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable or disable dynamic generation |
| `project_root` | `str \| None` | `None` | Project root. `None` uses `cwd()` |
| `generated_dir` | `str` | `"flowforge/generated"` | Directory for generated code. Must stay inside `project_root` |

### Persistence And Cache

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `persist_generated` | `bool` | `True` | Save generated flow/tool code to disk and record it in `manifest.json` |
| `auto_load_generated` | `bool` | `True` | Load previously generated flows/tools from `manifest.json` during `FlowForge.compile()` |

When both values are `True`, generated flows survive process restarts:

```text
First run:
  compile() -> no generated flow in the DAG
  run()     -> planner detects a gap -> flow generated -> manifest updated

Later run:
  compile() -> manifest is loaded -> generated flow is in the DAG
  run()     -> planner sees the requirement as covered -> no regeneration
```

FlowForge checks both the current DAG and the persisted manifest before
creating a new dynamic flow. If a flow with the same name already exists in
either place, generation is skipped and the existing flow is reused.

### Tool Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `include_builtin_tools` | `bool` | `True` | Enable the built-in tool pack for web, JSON, files, and document artifacts |
| `allow_tool_generation` | `bool` | `False` | Allow generation of new `FunctionTool` code when needed |
| `allow_codegen_tool_use` | `bool` | `False` | Allow generated code to call `ctx.call_llm()` with tool references |
| `generated_step_timeout_seconds` | `int` | `300` | Minimum timeout applied to every generated flow step when dynamic code is compiled or loaded from the manifest |

`allow_codegen_tool_use=False` keeps executable tools out of the codegen LLM
call. Prompt-only skills such as `ClaudeSkill` and `AgentSkill` may still be
attached so generated code can follow their instructions without running tools
during code generation.

### Shell Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed_shell_modes` | `list[str]` | `["readonly", "project_exec"]` | Allowed shell modes: `"readonly"`, `"workspace_write"`, `"project_exec"`, `"install_dependency"` |
| `shell_timeout_seconds` | `int` | `60` | Shell execution timeout |
| `shell_output_max_chars` | `int` | `4000` | Maximum shell output included in prompts/results |

### MCP Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcp_server_commands` | `dict[str, list[str]]` | `{}` | Named MCP server commands available during generation |
| `mcp_server_urls` | `dict[str, str]` | `{}` | MCP server name to Streamable HTTP endpoint URL |
| `mcp_server_tools` | `dict[str, list[str]]` | `{}` | Known tool names exposed by each MCP server for dynamic registration |
| `mcp_server_headers` | `dict[str, dict[str, str]]` | `{}` | Optional headers per MCP server, for authenticated gateways |
| `mcp_start_timeout_seconds` | `int` | `15` | Timeout for starting MCP servers |

Dynamic flows can start and register declared MCP servers:

```python
result = await ctx.call_tool("mcp_start_server", server_name="playwright")
registered = await ctx.call_tool("mcp_register_server", server_name="playwright")
```

After registration, later steps can scope the newly registered MCP tool names
with `tools=["browser_navigate"]` and expose them to the LLM with
`ctx.call_llm("Navigate to the target. <browser_navigate>")`.

For remote MCP services such as Figma, declare `mcp_server_urls` and
`mcp_server_tools` without `mcp_server_commands`. The generated flow can call
`mcp_register_server` directly and then use tools such as
`get_design_context`, `get_variable_defs`, or `get_metadata`.

### Codegen Context

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project_context_max_chars` | `int` | `4000` | Maximum project context included in the generation prompt |
| `codegen_tool_catalog_max_tools` | `int` | `12` | Maximum relevant tools shown in the codegen catalog. `<=0` disables selection |
| `codegen_tool_catalog_max_chars` | `int` | `6000` | Maximum characters for the codegen tool catalog. `<=0` disables truncation |
| `max_requirements` | `int` | `8` | Maximum planner requirements considered for generation |

The generator builds a compact tool catalog from `required_tools`, artifact
detection, query keywords, Agent Skills, Claude Skills, and declared MCP server
metadata. This keeps token use low while still forcing required tools into the
prompt.

### DependencyPolicy

`DependencyPolicy` controls package installation by generated code.

```python
from flowforge.types import DependencyPolicy

policy = DependencyPolicy(
    allow_install=True,
    allowed_packages=["httpx"],
    denied_packages=["subprocess"],
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_install` | `bool` | `False` | Whether the `pip_install` built-in may be used |
| `allowed_packages` | `list[str]` | `[]` | Package allowlist. Empty means all packages are allowed |
| `denied_packages` | `list[str]` | `[]` | Package denylist. Takes precedence over the allowlist |

---

## Generation Lifecycle

### 1. Planning Finds A Gap

In autonomous mode, the planner decomposes the user request into requirements:

```json
{
  "requirements": [
    {
      "id": "search_papers",
      "description": "Search arXiv papers",
      "covered": false,
      "suggested_flow_name": "search_arxiv_papers",
      "suggested_flow_prompt": "Search arXiv and return matching papers"
    },
    {
      "id": "write_report",
      "description": "Write a report",
      "covered": true,
      "matched_flow": "report_pipeline"
    }
  ]
}
```

Each uncovered requirement can trigger `_dynamic_generator`, unless the target
flow already exists in the DAG or manifest.

### 2. The Meta-Flow Builds A Brief

The internal `_dynamic_generator` flow is built with normal FlowForge
decorators:

```text
_dynamic_generator
└─ task: generate_dynamic_flow
   ├─ step[1] analyse_gap
   ├─ step[2] prepare_codegen
   └─ step[3] generate_and_inject
```

The brief includes:

- the uncovered requirement;
- the current DAG summary;
- downstream contract information when a generated upstream flow must feed an
  existing downstream flow;
- available tools and their parameter schemas;
- safety and style constraints for generated code.

### 3. Code Is Generated And Validated

`DynamicFlowGenerator` asks the LLM to produce FlowForge decorator code. Before
execution, the generated code must pass AST safety validation.

Blocked examples include:

- calls such as `os.system`, `os.popen`, `subprocess.run`, and `shutil.rmtree`;
- imports such as `socket` and `multiprocessing`;
- dynamic execution with `eval`, `exec`, or `compile`.

If compilation fails, FlowForge can feed the error back to the LLM and retry.

### 4. The Flow Is Injected

After validation and compilation, the generated flow is injected into the DAG.
If `persist_generated=True`, the source is also written to disk and registered
in `manifest.json`.

### 5. Planning Runs Again

The engine replans with the generated flow now present. Normal execution then
continues with the selected route.

---

## Patterns

### Pattern 1: Static Downstream, Dynamic Upstream

Use this when the report, rendering, or delivery pipeline is known, but the
source-gathering step may vary by request.

```python
@flow(name="report_pipeline", prompt="Write a report from prepared source data")
class ReportPipeline:
    ...


@global_config(
    prompt="Research assistant",
    dynamic_flow=True,
)
class Agent:
    ReportPipeline = ReportPipeline
```

At runtime, the planner can decide that a missing search flow is required,
generate it, and then execute:

```text
search_arxiv_papers -> report_pipeline
```

### Pattern 2: Empty Agent

Use this for exploratory agents where every capability can be generated:

```python
@global_config(
    prompt="General-purpose agent",
    dynamic_flow=True,
)
class Agent:
    pass
```

The planner decomposes complex requests into requirements and generates the
flows it needs.

### Pattern 3: Contract-First Chaining

When a generated upstream flow must feed an existing downstream flow,
FlowForge extracts the downstream `input_schema` and asks codegen to return a
compatible output. The compiled flow is then checked for compatibility before
it is used.

---

## Manifest Persistence And Caching

When `persist_generated=True`, generated code is saved under `generated_dir`:

```text
flowforge/generated/
├── manifest.json
├── manifest.json.lock
├── flows/
│   └── search_papers.py
└── tools/
    └── custom_tool.py
```

Example manifest:

```json
{
  "version": 1,
  "flows": [
    {
      "name": "top_mountains_table",
      "file": "flowforge/generated/flows/top_mountains_table.py",
      "class_name": "TopMountainsTableFlow",
      "downstream_flow_route": "",
      "bridge": "",
      "created_at": "2026-04-26T10:30:00+00:00"
    }
  ],
  "tools": []
}
```

`manifest.json.lock` protects concurrent writes with `fcntl.flock`.

Option behavior:

| Option | Behavior |
|--------|----------|
| `persist_generated=True` | Write generated code to disk and record it in the manifest |
| `persist_generated=False` | Inject only into the current in-memory DAG |
| `auto_load_generated=True` | Import manifest records during `compile()` |
| `auto_load_generated=False` | Leave previous generated files on disk but do not auto-load them |

---

## Built-In Tools

When `include_builtin_tools=True`, generated flows can use a built-in tool
pack.

Generated flows are instructed to scope intended tools in decorators with
string references, for example:

```python
@flow(name="clone_site", prompt="Clone a public site", tools=["web_fetch_url"])
class CloneSite:
    @task(name="inspect", prompt="Inspect the target page", tools=["web_fetch_url"])
    class Inspect:
        @step(order=1, prompt="Fetch the page with the web tool", tools=["web_fetch_url"])
        async def fetch(ctx):
            return await ctx.call_tool("web_fetch_url", url=ctx.input["target_url"])
```

For LLM-mediated tool use, generated steps must include the angle-bracket
reference in the runtime prompt, such as
`await ctx.call_llm("Inspect this page with <web_fetch_url>")`.

### Utility Tools

| Tool | Description | Gate |
|------|-------------|------|
| `pip_install` | Install Python packages with `pip install` | `DependencyPolicy(allow_install=True)` |
| `python_import_check` | Check whether a Python module can be imported | Always available |
| `web_fetch_url` | Fetch text from a URL | Always available |
| `json_select_fields` | Select fields from JSON | Always available |
| `shell_readonly` | Run read-only shell inspection commands | `allowed_shell_modes` includes `"readonly"` |
| `shell_project_exec` | Run project commands such as `npm run build` or `python -m pytest` | `allowed_shell_modes` includes `"project_exec"` |
| `shell_workspace_write` | Run limited workspace-writing commands such as `mkdir`, `touch`, `cp`, and `mv` | `allowed_shell_modes` includes `"workspace_write"` |
| `shell_install_dependency` | Run dependency installation commands such as `npm install`, `pnpm add`, `yarn add`, or `python -m pip install` | `allowed_shell_modes` includes `"install_dependency"` and `DependencyPolicy.allow_install=True` |

### File Tools

| Tool | Description | Gate |
|------|-------------|------|
| `files_read_text` | Read text files | Path must stay under `project_root` |
| `files_write_text` | Write text files | Path must stay under `project_root` |
| `files_list_dir` | List directory entries | Path must stay under `project_root` |

### Document And Artifact Tools

| Tool | Description | Gate |
|------|-------------|------|
| `pdf_read_text` | Extract text from PDF files | Requires `pypdf` |
| `pptx_create` | Create PowerPoint decks with layouts, tables, themes, and speaker notes | Requires `python-pptx` |
| `csv_read` | Read CSV rows as dictionaries | Always available |
| `csv_write` | Write CSV files | Always available |
| `docx_create` | Create Word documents | Requires `python-docx` |
| `markdown_write` | Write Markdown files | Always available |
| `chart_create` | Create PNG charts (`bar`, `line`, `pie`, `scatter`) | Requires `matplotlib` |

All file tools resolve paths through a project-root guard so generated code
cannot write outside the configured project boundary.

---

## Direct Tool Calls From Steps

Generated and hand-written steps can call registered `FunctionTool`s directly
with `ctx.call_tool()`:

```python
@step(order=1, prompt="Render a presentation")
async def render(ctx):
    slides = json.dumps([
        {"layout": "cover", "title": "Report", "subtitle": "2026"},
        {"layout": "content", "title": "Summary", "body": "Key points..."},
        {
            "layout": "table",
            "title": "Data",
            "table": {"headers": ["Metric", "Value"], "rows": [["A", "10"]]},
        },
    ])
    return await ctx.call_tool("pptx_create", path="report.pptx", slides=slides)
```

`ctx.call_tool()` searches the merged tools in global -> flow -> task -> step
order and executes the matching local function tool.

### `pptx_create` Layouts

| Layout | Required fields | Optional fields |
|--------|-----------------|-----------------|
| `cover` | `title` | `subtitle` |
| `content` | `title`, `body` | `notes` |
| `section` | `title` | `subtitle` |
| `comparison` | `title`, `left_title`, `left_body`, `right_title`, `right_body` | `notes` |
| `table` | `title`, `table.headers`, `table.rows` | `notes` |
| `blank` | none | none |

---

## LLM Response Cleanup

`ctx.call_llm()` automatically strips Markdown code fences from responses and
attempts JSON parsing when possible. This makes generated flows more robust
when they call `json.loads()` on model output.

---

## Examples

- `examples/dynamic_paper_report_agent.py` — static downstream pipeline plus
  dynamic upstream paper search.
- `examples/dynamic_bare_agent.py` — zero static flows and zero static tools;
  everything is generated at runtime.
- `examples/dynamic_clone_coding_agent.py` — zero static flows, a local
  `AgentSkill`, Anthropic's `frontend-design` Skill guidance loaded as a
  local Agent Skill for codegen,
  built-in web/file/shell tools, and an npm-based frontend project generated
  under `~/test`.
- `examples/dynamic_skill_mcp_agent.py` — zero static flows, Agent Skill
  guidance, optional Claude `pptx` Skill usage, compact tool catalog settings,
  and dynamic MCP server registration for Playwright or Figma.
