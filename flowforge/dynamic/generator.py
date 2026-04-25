"""Dynamic flow generation — LLM writes FlowForge code, framework compiles it.

The ``DynamicFlowGenerator`` is the core of the ``dynamic_flow`` feature.
It takes a user query and the current DAG state, determines whether the
query is covered by existing flows, and if not, asks an LLM to produce
new FlowForge decorator code.  The generated code is compiled in a safe
sandbox and, on success, injected into the live DAG.

Design decisions
----------------
* **Template-guided generation**: the LLM fills in decorator parameters and
  ``ctx.call_llm()`` instructions rather than writing arbitrary Python.
  This dramatically reduces failure rates.
* **Compile → retry loop**: if the generated code fails to compile, the
  error message is fed back to the LLM for self-correction (up to 3 tries).
* **Module-level classes**: generated code always defines classes at module
  scope (never inside functions) to avoid Python scoping issues.
* **AST safety check**: before executing generated code, an AST scan rejects
  dangerous patterns (``os.system``, ``subprocess``, ``__import__``, etc.).
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.annotations.metadata import FlowMeta, GlobalMeta, TaskMeta
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.types import LLMConfig, ToolConfig
    from flowforge.doc.models import AnyDoc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST safety validation — reject dangerous patterns before exec_module()
# ---------------------------------------------------------------------------

# Attribute calls that are never allowed in generated code.
_BLOCKED_ATTR_CALLS: set[str] = {
    "os.system",
    "os.popen",
    "os.exec",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawn",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "shutil.rmtree",
}

# Module-level imports that are never allowed.
_BLOCKED_IMPORTS: set[str] = {
    "subprocess",
    "shutil",
    "ctypes",
    "socket",
}

# Built-in function names that are never allowed as top-level calls.
_BLOCKED_BUILTINS: set[str] = {
    "__import__",
    "exec",
    "eval",
    "compile",
}


def _validate_generated_ast(code: str) -> str | None:
    """Parse *code* and reject dangerous patterns.

    Returns ``None`` when the code is safe, or a human-readable error
    string describing the first violation found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        # --- blocked imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    return (
                        f"import of '{alias.name}' is not allowed in "
                        f"generated code (line {node.lineno})"
                    )
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    return (
                        f"import from '{node.module}' is not allowed in "
                        f"generated code (line {node.lineno})"
                    )

        # --- blocked builtins (bare calls like eval(...)) ---
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                return (
                    f"call to '{func.id}()' is not allowed in "
                    f"generated code (line {node.lineno})"
                )

            # --- blocked attribute calls (os.system(...), subprocess.run(...)) ---
            if isinstance(func, ast.Attribute):
                parts = _resolve_attr_chain(func)
                if parts:
                    full = ".".join(parts)
                    for blocked in _BLOCKED_ATTR_CALLS:
                        if full == blocked or full.endswith(f".{blocked}"):
                            return (
                                f"call to '{full}()' is not allowed in "
                                f"generated code (line {node.lineno})"
                            )

    return None


def _resolve_attr_chain(node: ast.Attribute) -> list[str] | None:
    """Walk an ``ast.Attribute`` chain and return dotted names, e.g. ``['os', 'system']``."""
    parts: list[str] = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return parts
    return None


# ---------------------------------------------------------------------------
# Schema introspection (used to bridge dynamic flow ↔ existing flow schemas)
# ---------------------------------------------------------------------------

def _entry_input_schema(flow_meta: "FlowMeta") -> type | None:
    """Return the first Pydantic ``input_schema`` along *flow_meta*'s entry path.

    Walks down from the flow → first ordered child flow / task → first step
    and stops at the first non-``None`` ``input_schema``.  Returns ``None``
    when nothing along the path declares one (auto-binding) or when the
    entry path traverses a branch dispatcher (unpredictable).
    """
    if flow_meta.input_schema is not None:
        return flow_meta.input_schema
    if getattr(flow_meta, "is_branch", False):
        return None

    def _task_key(t: "TaskMeta", fallback: int) -> tuple[int, int]:
        return (t.order if t.order is not None else fallback, fallback)

    def _flow_key(f: "FlowMeta", fallback: int) -> tuple[int, int]:
        return (f.order if f.order is not None else fallback, fallback)

    children: list[tuple[tuple[int, int], str, object]] = []
    for idx, cf in enumerate(flow_meta.child_flows):
        children.append((_flow_key(cf, idx), "flow", cf))
    offset = len(flow_meta.child_flows)
    for idx, t in enumerate(flow_meta.tasks):
        children.append((_task_key(t, offset + idx), "task", t))
    if not children:
        return None
    children.sort(key=lambda item: item[0])

    _, kind, first = children[0]
    if kind == "flow":
        return _entry_input_schema(first)  # type: ignore[arg-type]
    return _task_entry_input_schema(first)  # type: ignore[arg-type]


def _task_entry_input_schema(task_meta: "TaskMeta") -> type | None:
    if task_meta.input_schema is not None:
        return task_meta.input_schema
    if task_meta.is_branch:
        return None
    if task_meta.is_leaf:
        if not task_meta.steps:
            return None
        first_step = min(task_meta.steps, key=lambda s: s.order)
        return first_step.input_schema
    if not task_meta.child_tasks:
        return None
    first_child = min(
        task_meta.child_tasks,
        key=lambda t: (t.order if t.order is not None else 0),
    )
    return _task_entry_input_schema(first_child)


def _exit_output_schema(flow_meta: "FlowMeta") -> type | None:
    """Return the declared ``output_schema`` at *flow_meta*'s exit path.

    Walks down to the last ordered child flow / task / step and returns its
    ``output_schema`` (or the flow's own ``output_schema`` when declared).
    Returns ``None`` when nothing is declared along the exit path.
    """
    if flow_meta.output_schema is not None:
        return flow_meta.output_schema
    if getattr(flow_meta, "is_branch", False):
        return None

    children: list[tuple[tuple[int, int], str, object]] = []
    for idx, cf in enumerate(flow_meta.child_flows):
        key = (cf.order if cf.order is not None else idx, idx)
        children.append((key, "flow", cf))
    offset = len(flow_meta.child_flows)
    for idx, t in enumerate(flow_meta.tasks):
        key = (t.order if t.order is not None else offset + idx, offset + idx)
        children.append((key, "task", t))
    if not children:
        return None
    children.sort(key=lambda item: item[0])

    _, kind, last = children[-1]
    if kind == "flow":
        return _exit_output_schema(last)  # type: ignore[arg-type]
    return _task_exit_output_schema(last)  # type: ignore[arg-type]


def _task_exit_output_schema(task_meta: "TaskMeta") -> type | None:
    if task_meta.output_schema is not None:
        return task_meta.output_schema
    if task_meta.is_branch:
        return None
    if task_meta.is_leaf:
        if not task_meta.steps:
            return None
        last_step = max(task_meta.steps, key=lambda s: s.order)
        return last_step.output_schema
    if not task_meta.child_tasks:
        return None
    last_child = max(
        task_meta.child_tasks,
        key=lambda t: (t.order if t.order is not None else 0),
    )
    return _task_exit_output_schema(last_child)


def _schema_to_contract(model_cls: type | None) -> dict[str, Any] | None:
    """Return ``model_cls.model_json_schema()`` or ``None`` when not a model."""
    if model_cls is None:
        return None
    try:
        schema = model_cls.model_json_schema()
    except Exception:
        return None
    return schema


def _summarise_schema_mismatch(
    produced: dict[str, Any] | None,
    expected: dict[str, Any],
) -> str:
    """Return a short, human-readable mismatch description for self-correction."""
    if produced is None:
        return (
            "the generated flow did not declare any output_schema and no explicit "
            "output contract was produced"
        )

    expected_required = set(expected.get("required", []) or [])
    produced_props = set((produced.get("properties") or {}).keys())
    missing = sorted(expected_required - produced_props)
    extra = sorted(produced_props - set((expected.get("properties") or {}).keys()))

    parts: list[str] = []
    if missing:
        parts.append(f"missing required top-level keys: {missing}")
    if extra:
        parts.append(f"unexpected top-level keys: {extra}")
    if not parts:
        parts.append("top-level key names match but value types differ")
    return "; ".join(parts)

# Maximum compile-retry attempts when LLM-generated code fails validation.
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GAP_ANALYSIS_SYSTEM = textwrap.dedent("""\
    You are FlowForge's gap analyser.  Given a user query and a list of
    existing flows (with their doc summaries), determine whether the query
    can be handled by any combination of existing flows.

    Respond with JSON:
    {
        "covered": true/false,
        "reason": "short explanation",
        "suggested_flow_name": "snake_case name for the new flow (only if covered=false)",
        "suggested_flow_prompt": "1-2 sentence description (only if covered=false)"
    }
""")

_CODEGEN_SYSTEM = textwrap.dedent("""\
    You are FlowForge's code generator.  Generate Python code that defines a
    new FlowForge flow using decorators.

    RULES:
    1. Import only from flowforge: `from flowforge import flow, task, step`
       (and `import json` when needed for serialisation)
    2. All classes at MODULE LEVEL (never inside functions)
    3. Step functions use `async def name(ctx):` signature
    4. Use `await ctx.call_llm("instruction")` for AI-powered steps that need
       LLM reasoning, summarisation, or analysis
    5. Use `return {{"key": "value"}}` for code-only steps
    6. The top-level class must be decorated with @flow
    7. Do NOT use @global_config — only generate the @flow and its children
    8. flow name must be: {flow_name}
    9. Include input handling via `ctx.input` (dict or Pydantic model)

    TOOL USAGE — TWO METHODS:
    There are two ways to use available tools in generated code:

    METHOD A — `ctx.call_tool(name, **kwargs)` (DIRECT programmatic call):
      Use this when the step's logic is deterministic and you know exactly
      which tool to call with which parameters. The tool is called directly
      as a Python function — no LLM reasoning overhead.
      ```python
      @step(order=1, prompt="fetch data from URL")
      async def fetch(ctx):
          result = await ctx.call_tool("web_fetch_url", url="https://...", timeout_seconds=30)
          return result
      ```

    METHOD B — `ctx.call_llm("instruction <tool_name>")` (LLM-mediated call):
      Use this when the step needs the LLM to decide HOW to use the tool,
      which parameters to pass, or to interpret the results. The `<tool_name>`
      syntax makes the tool available to the LLM as a callable tool.
      ```python
      @step(order=1, prompt="search and analyse papers")
      async def search(ctx):
          return await ctx.call_llm(
              "Search for papers about {{query}} using <web_fetch_url> "
              "and return the results as a structured JSON."
          )
      ```

    TOOL SELECTION STRATEGY:
    10. FIRST check the "Available tools" list in the user message. If a
        builtin tool already covers the capability you need (e.g. web_fetch_url
        for HTTP GET, pptx_create for PowerPoint, csv_write for CSV, etc.),
        USE IT via ctx.call_tool() or <tool_name>. Do NOT re-implement what
        a tool already does.
    11. If NO existing tool covers the need AND the dynamic policy allows
        tool generation, write lightweight Python code in the step body.
        Prefer stdlib modules. Do NOT import blocked modules (subprocess,
        shutil, ctypes, socket).
    12. For file I/O, ALWAYS prefer builtin tools (files_read_text,
        files_write_text, csv_read, csv_write, pptx_create, docx_create,
        markdown_write, chart_create, pdf_read_text) over raw `open()` calls.
        These tools enforce project_root sandboxing.
    13. For web requests, ALWAYS use the `web_fetch_url` tool instead of
        writing urllib/httpx/requests code directly.
    14. For package installation, use `pip_install` tool when the dependency
        policy allows it.
    15. Use builtin runtime tools (python_import_check, shell_*) only when the
        generated flow's actual job needs them. Do NOT add project inspection
        or test-running steps unless explicitly requested.
    16. If a tool returns structured JSON needed by downstream flows, instruct
        the model to return ONLY that JSON payload without extra commentary
    17. When you need earlier step outputs, use `ctx.previous_results` or
        `ctx.step_results` instead of inventing new context fields
    18. Generate ONLY the missing flow named {flow_name}; do NOT recreate
        downstream capabilities that other existing flows already cover
    19. Do NOT reference existing flows via `<flow_name>` tool syntax.
        Flows are composed by the planner/engine, not called as tools.

    OUTPUT CONTRACT (hard requirement):
    - The final step of this flow MUST return a plain Python ``dict`` whose
      shape matches the JSON Schema supplied in the user message under the
      heading "## Downstream input contract". Missing / extra top-level keys,
      wrong nesting, or wrong value types WILL cause downstream validation
      errors and trigger a compile-retry.
    - If no OUTPUT CONTRACT is supplied the flow may return any JSON-
      serialisable dict, but keys should still be self-describing.
    - Do NOT wrap the returned dict in extra envelopes like
      ``{{"result": {{ ... }}}}`` unless the contract demands it.
    - Keep schemas inline via plain dicts; do NOT ``import pydantic`` or
      declare Pydantic models — the framework handles validation for you.

    STRUCTURE:
    A @flow can contain BOTH child @flow classes AND @task classes.
    A @task can contain child @task classes OR @step functions (not both).
    Only leaf @task classes contain @step functions.

    SIMPLE TEMPLATE (single task, direct tool call):
    ```python
    from flowforge import flow, task, step
    import json

    @flow(name="{flow_name}", prompt="{flow_prompt}")
    class {class_name}:
        @task(name="main_task", prompt="description")
        class MainTask:
            @step(order=1, prompt="fetch data")
            async def fetch_data(ctx):
                result = await ctx.call_tool("web_fetch_url", url="https://api.example.com/data")
                if not result.get("ok"):
                    return {{"error": result.get("error", "fetch failed")}}
                return json.loads(result["body"])

            @step(order=2, prompt="analyse and summarise the data")
            async def analyse(ctx):
                data = ctx.previous_results.get(1)
                return await ctx.call_llm(
                    f"Analyse this data and produce a summary:\\n{{json.dumps(data)}}"
                )
    ```

    COMPLEX TEMPLATE (child flows + tasks):
    ```python
    from flowforge import flow, task, step

    @flow(name="sub_process", prompt="a sub-process")
    class SubProcessFlow:
        @task(name="sub_task", prompt="sub task")
        class SubTask:
            @step(order=1, prompt="sub step")
            async def sub_step(ctx):
                return await ctx.call_llm("do sub work on {{field}}")

    @flow(name="{flow_name}", prompt="{flow_prompt}")
    class {class_name}:
        # Child flow — define at module level, reference as class attribute
        SubProcessFlow = SubProcessFlow

        # Direct task — sibling to the child flow
        @task(name="finalize", prompt="finalize after sub-process")
        class FinalizeTask:
            @step(order=1, prompt="wrap up")
            async def wrap_up(ctx):
                return await ctx.call_llm("summarize results from {{field}}")
    ```

    Choose the simplest structure that fits the request.
    Generate ONLY the Python code, no markdown fences, no explanation.
""")

_FIX_SYSTEM = textwrap.dedent("""\
    The following FlowForge code failed to compile.  Fix the error and
    return ONLY the corrected Python code (no markdown, no explanation).

    Error:
    {error}

    Original code:
    {code}
""")

_TOOL_CODEGEN_SYSTEM = textwrap.dedent("""\
    You are FlowForge's tool generator. Generate a small Python module that
    exposes exactly one function tool.

    RULES:
    1. Prefer Python standard library code.
    2. Do not install dependencies in the generated code.
    3. If a dependency is needed, include a module-level DEPENDENCIES list.
    4. Define a function named {tool_name} with typed parameters.
    5. Return JSON-serialisable values only.
    6. Generate ONLY Python code, no markdown fences, no explanation.
""")


class DynamicFlowGenerator:
    """Generates, compiles, and injects new flows at runtime.

    Parameters
    ----------
    llm_config:
        LLM settings used for code generation and gap analysis.
    dag:
        The current compiled DAG (read-only — injection is done by the caller).
    docs:
        Existing node docs for gap analysis.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        dag: FlowForgeDAG,
        docs: dict[str, AnyDoc] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        dynamic_options: Any = None,
    ) -> None:
        self._llm_config = llm_config
        self._dag = dag
        self._docs = docs if docs is not None else {}
        self._tool_configs = tool_configs or []
        self._dynamic_options = dynamic_options

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyse_gap(self, user_query: str | Any) -> dict[str, Any]:
        """Check if the user query is covered by existing flows.

        Returns a dict with ``covered`` (bool), ``reason``, and — when
        ``covered`` is False — ``suggested_flow_name`` and
        ``suggested_flow_prompt``.
        """
        from flowforge.llm.caller import call_with_tool
        from flowforge.schema.dag import NodeType

        query_str = str(user_query)

        # Build a summary of existing flows for the LLM.
        flow_summaries: list[str] = []
        for node in self._dag.get_all_nodes():
            if node.type != NodeType.FLOW:
                continue
            doc = self._docs.get(node.id)
            summary = getattr(doc, "summary", "") if doc else ""
            prompt_text = getattr(node.meta, "prompt", "")
            flow_summaries.append(
                f"- {node.id}: {prompt_text}"
                + (f" (doc: {summary})" if summary else "")
            )

        flows_text = "\n".join(flow_summaries) if flow_summaries else "(no flows)"

        user_prompt = (
            f"User query: {query_str}\n\n"
            f"Existing flows:\n{flows_text}"
        )

        tool_schema = {
            "name": "gap_analysis",
            "description": "Analyse whether the user query is covered by existing flows",
            "input_schema": {
                "type": "object",
                "properties": {
                    "covered": {
                        "type": "boolean",
                        "description": "True if existing flows can handle the query",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of the analysis",
                    },
                    "suggested_flow_name": {
                        "type": "string",
                        "description": "snake_case name for the new flow (only if covered=false)",
                    },
                    "suggested_flow_prompt": {
                        "type": "string",
                        "description": "1-2 sentence description of the new flow (only if covered=false)",
                    },
                },
                "required": ["covered", "reason"],
            },
        }

        result = await call_with_tool(
            prompt=user_prompt,
            tool_schema=tool_schema,
            llm_config=self._llm_config,
            system_prompt=_GAP_ANALYSIS_SYSTEM,
            max_tokens=512,
        )
        logger.info("gap analysis result: %s", result)
        return result

    async def generate_flow_code(
        self,
        flow_name: str,
        flow_prompt: str,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        project_context: str | None = None,
    ) -> str:
        """Ask the LLM to generate FlowForge decorator code for a new flow.

        Parameters
        ----------
        flow_name, flow_prompt, user_query:
            Natural-language description of the missing flow.
        downstream_contract:
            JSON Schema the generated flow's final step output MUST satisfy.
            When provided, it is embedded in the user prompt as a hard
            contract so the LLM produces a matching dict shape.
        downstream_flow_name:
            Human-readable name of the existing flow that will consume this
            output (used only for explanatory context in the prompt).
        project_context:
            Optional lightweight implementation context prepared by the
            dynamic meta-flow before code generation.

        Returns
        -------
        str
            The generated Python source code.
        """
        from flowforge.execution.llm import call_llm_api
        import json

        # Build a PascalCase class name from the snake_case flow name.
        class_name = "".join(
            part.capitalize() for part in flow_name.split("_")
        ) + "Flow"

        system = _CODEGEN_SYSTEM.format(
            flow_name=flow_name,
            flow_prompt=flow_prompt,
            class_name=class_name,
        )

        tool_catalog = self._format_tool_catalog()

        contract_block = ""
        if downstream_contract:
            pretty = json.dumps(downstream_contract, ensure_ascii=False, indent=2)
            downstream_hint = (
                f" (consumed by the existing flow `{downstream_flow_name}`)"
                if downstream_flow_name
                else ""
            )
            contract_block = (
                f"## Downstream input contract{downstream_hint}\n"
                f"The final step of `{flow_name}` MUST return a dict whose "
                f"shape matches this JSON Schema exactly. Use identical "
                f"top-level field names and types:\n"
                f"```json\n{pretty}\n```\n\n"
            )

        # Auto-detect output artifacts from the user query.
        artifacts = detect_output_artifacts(
            str(user_query),
            available_tools=self._tool_names(),
        )
        artifact_block = _format_artifact_instructions(artifacts)

        artifact_section = f"{artifact_block}\n\n" if artifact_block else ""

        user_prompt = (
            f"Missing flow scope that triggered this flow generation:\n"
            f"{user_query}\n\n"
            f"Flow name: {flow_name}\n"
            f"Flow purpose: {flow_prompt}\n"
            f"Class name: {class_name}\n\n"
            f"{contract_block}"
            f"{artifact_section}"
            f"{self._format_project_context(project_context)}"
            f"Available tools:\n{tool_catalog}\n\n"
            f"Dynamic policy:\n{self._format_dynamic_policy()}\n\n"
            f"Generate ONLY the complete FlowForge code for this missing flow."
        )

        code = await call_llm_api(
            system_prompt=system,
            user_prompt=user_prompt,
            llm_config=self._llm_config,
            tool_configs=self._codegen_tool_configs(),
            max_tool_rounds=3,
        )

        # Strip markdown fences if the LLM included them.
        code = _strip_markdown_fences(str(code))
        return code

    async def generate_tool_code(
        self,
        tool_name: str,
        tool_prompt: str,
        user_query: str | Any,
        dependency_hints: list[str] | None = None,
    ) -> str:
        """Generate Python code for a missing FunctionTool.

        Tool generation is separate from flow generation: the planner can
        identify that a capability is missing, then this method produces a
        small importable module. Installation remains policy-gated.
        """
        from flowforge.execution.llm import call_llm_api

        dependencies = dependency_hints or []
        policy_error = self._check_dependency_policy(dependencies)
        policy_text = self._format_dynamic_policy()
        if policy_error:
            policy_text += f"\nDependency warning: {policy_error}"

        code = await call_llm_api(
            system_prompt=_TOOL_CODEGEN_SYSTEM.format(tool_name=tool_name),
            user_prompt=(
                f"Tool name: {tool_name}\n"
                f"Tool purpose: {tool_prompt}\n"
                f"User request: {user_query}\n"
                f"Dependency hints: {dependencies}\n\n"
                f"Dynamic policy:\n{policy_text}\n\n"
                "Generate the complete Python module for this tool."
            ),
            llm_config=self._llm_config,
        )
        return _strip_markdown_fences(str(code))

    def compile_tool_code(self, code: str, tool_name: str) -> ToolConfig:
        """Compile generated tool code into a ``FunctionTool`` or ToolConfig.

        The code undergoes AST safety validation before execution.
        """
        from flowforge.errors import CompileError
        from flowforge.types import FunctionTool

        # AST safety check.
        safety_error = _validate_generated_ast(code)
        if safety_error is not None:
            raise CompileError(f"Generated tool code failed safety check: {safety_error}")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="flowforge_dynamic_tool_",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        module_name = f"_flowforge_dynamic_tool_{temp_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            if spec is None or spec.loader is None:
                raise CompileError(f"Cannot load generated tool from {temp_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            dependencies = getattr(module, "DEPENDENCIES", [])
            policy_error = self._check_dependency_policy(list(dependencies or []))
            if policy_error:
                raise CompileError(policy_error)

            if hasattr(module, "TOOL_CONFIG"):
                return getattr(module, "TOOL_CONFIG")
            if hasattr(module, tool_name):
                return FunctionTool(func=getattr(module, tool_name), name=tool_name)

            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(module, attr_name)
                if callable(obj):
                    return FunctionTool(func=obj, name=tool_name)
            raise CompileError("Generated tool code does not contain a function.")
        except CompileError:
            raise
        except Exception as e:
            raise CompileError(f"Failed to compile generated tool code: {e}") from e
        finally:
            # Clean up temp file; keep module in sys.modules for runtime use.
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def generate_and_persist_tool(
        self,
        tool_name: str,
        tool_prompt: str,
        user_query: str | Any,
        dependency_hints: list[str] | None = None,
    ) -> ToolConfig:
        """Generate, compile, and optionally persist a new tool."""
        if self._dynamic_options is not None and not getattr(
            self._dynamic_options, "allow_tool_generation", False,
        ):
            from flowforge.errors import CompileError

            raise CompileError(
                "Dynamic tool generation is disabled by DynamicRunOptions."
            )

        code = await self.generate_tool_code(
            tool_name=tool_name,
            tool_prompt=tool_prompt,
            user_query=user_query,
            dependency_hints=dependency_hints,
        )
        tool = self.compile_tool_code(code, tool_name)

        if self._dynamic_options is not None and getattr(
            self._dynamic_options, "persist_generated", False,
        ):
            from flowforge.dynamic.manifest import persist_tool_code

            persist_tool_code(
                tool_name=tool_name,
                code=code,
                options=self._dynamic_options,
                symbol=tool_name,
                dependencies=dependency_hints or [],
            )
        return tool

    def compile_flow_code(self, code: str) -> FlowMeta:
        """Compile generated Python code into a ``FlowMeta``.

        The code is first validated with an AST safety check (rejecting
        dangerous patterns like ``os.system``, ``subprocess``, ``eval``, etc.),
        then written to a temporary file, imported as a module, and scanned
        for a ``@flow``-decorated class.

        Parameters
        ----------
        code:
            Python source code containing a ``@flow``-decorated class.

        Returns
        -------
        FlowMeta
            The metadata extracted from the decorated class.

        Raises
        ------
        CompileError
            If the code fails AST safety validation, no ``@flow``-decorated
            class is found, or the code has errors.
        """
        from flowforge.annotations.decorators import _FLOW_ATTR
        from flowforge.errors import CompileError

        # AST safety check — reject dangerous patterns before execution.
        safety_error = _validate_generated_ast(code)
        if safety_error is not None:
            raise CompileError(f"Generated code failed safety check: {safety_error}")

        # Write to a temp file and import it.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="flowforge_dynamic_",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        module_name = f"_flowforge_dynamic_{temp_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            if spec is None or spec.loader is None:
                raise CompileError(f"Cannot load generated module from {temp_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            # Scan for @flow-decorated class.
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if isinstance(obj, type) and hasattr(obj, _FLOW_ATTR):
                    return getattr(obj, _FLOW_ATTR)

            raise CompileError(
                "Generated code does not contain a @flow-decorated class."
            )
        except CompileError:
            raise
        except Exception as e:
            raise CompileError(f"Failed to compile generated code: {e}") from e
        finally:
            # Clean up the temp file.  The module stays in sys.modules so
            # that step functions defined inside it remain importable when
            # the runner invokes them at execution time.  Removing it would
            # break ``__module__`` references on the generated classes.
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def generate_and_compile(
        self,
        flow_name: str,
        flow_prompt: str,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        project_context: str | None = None,
    ) -> tuple[FlowMeta, str]:
        """Generate code, compile it, and retry on errors.

        Makes up to ``_MAX_RETRIES`` attempts.  On each failure the error
        message is fed back to the LLM for self-correction.  When
        ``downstream_contract`` is supplied it is threaded into the codegen
        prompt so the LLM knows the exact output shape it must produce.

        Returns
        -------
        tuple[FlowMeta, str]
            The compiled metadata for the new flow and the final source code.
        """
        from flowforge.errors import CompileError

        code = await self.generate_flow_code(
            flow_name, flow_prompt, user_query,
            downstream_contract=downstream_contract,
            downstream_flow_name=downstream_flow_name,
            project_context=project_context,
        )
        logger.info("generated code for flow '%s' (%d chars)", flow_name, len(code))
        logger.debug("generated code:\n%s", code)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                meta = self.compile_flow_code(code)
                compatibility_error = self.check_contract_compatibility(
                    meta, downstream_contract,
                )
                if compatibility_error is not None:
                    raise CompileError(compatibility_error)
                logger.info(
                    "flow '%s' compiled successfully on attempt %d",
                    flow_name, attempt + 1,
                )
                return meta, code
            except (CompileError, Exception) as e:
                last_error = e
                logger.warning(
                    "compile attempt %d/%d failed for '%s': %s",
                    attempt + 1, _MAX_RETRIES, flow_name, e,
                )
                if attempt + 1 < _MAX_RETRIES:
                    code = await self.fix_code(
                        code, str(e), user_query,
                        downstream_contract=downstream_contract,
                        downstream_flow_name=downstream_flow_name,
                        project_context=project_context,
                    )
                    logger.debug("fixed code (attempt %d):\n%s", attempt + 2, code)

        raise CompileError(
            f"Failed to compile dynamic flow '{flow_name}' after "
            f"{_MAX_RETRIES} attempts. Last error: {last_error}"
        )

    async def generate_compile_and_persist(
        self,
        flow_name: str,
        flow_prompt: str,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        downstream_flow_route: str = "",
        project_context: str | None = None,
    ) -> tuple[FlowMeta, str]:
        """Generate, compile, and optionally persist a flow.

        This is the single entry point used by the meta-flow.  It combines
        ``generate_and_compile()`` with persistence via the manifest, so
        the meta-flow does not need to duplicate the retry loop.

        Returns
        -------
        tuple[FlowMeta, str]
            The compiled metadata and the final source code.
        """
        meta, code = await self.generate_and_compile(
            flow_name=flow_name,
            flow_prompt=flow_prompt,
            user_query=user_query,
            downstream_contract=downstream_contract,
            downstream_flow_name=downstream_flow_name,
            project_context=project_context,
        )

        if (
            self._dynamic_options is not None
            and self._dynamic_options.persist_generated
        ):
            from flowforge.dynamic.manifest import persist_flow_code

            persist_flow_code(
                flow_name=meta.name,
                code=code,
                options=self._dynamic_options,
                class_name=meta.cls.__name__,
                inject_before=downstream_flow_route,
                downstream_flow_route=downstream_flow_route,
            )

        return meta, code

    def resolve_downstream_contract(
        self,
        downstream_flow_route: str | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Look up the downstream flow's entry ``input_schema`` as JSON Schema.

        Returns a ``(contract, flow_name)`` tuple.  Either element may be
        ``None`` when the route is missing, unknown, or the downstream flow
        declares no ``input_schema`` along its entry path.
        """
        if not downstream_flow_route:
            return None, None

        from flowforge.annotations.metadata import FlowMeta
        from flowforge.schema.dag import NodeType

        route = downstream_flow_route.removeprefix("global.")
        node = self._dag.get_node(f"global.{route}")
        if node is None or node.type != NodeType.FLOW:
            logger.info(
                "downstream_flow_route %r did not resolve to a flow node; "
                "skipping contract injection", downstream_flow_route,
            )
            return None, None

        flow_meta = node.meta
        if not isinstance(flow_meta, FlowMeta):
            return None, None

        entry_schema = _entry_input_schema(flow_meta)
        if entry_schema is None:
            logger.info(
                "downstream flow '%s' declares no input_schema on its entry "
                "path; proceeding without a strict contract", flow_meta.name,
            )
            return None, flow_meta.name

        contract = _schema_to_contract(entry_schema)
        return contract, flow_meta.name

    async def run_full_pipeline(
        self,
        user_query: str | Any,
        downstream_flow_route: str | None = None,
    ) -> tuple[FlowMeta, dict[str, Any], str]:
        """Full pipeline: gap analysis → code gen → compile.

        When ``downstream_flow_route`` is provided (typically forwarded from
        the planner's gap metadata) the generator introspects that flow's
        entry ``input_schema`` and passes the resulting JSON Schema to the
        LLM as a hard output contract.

        Returns
        -------
        tuple[FlowMeta, dict, str]
            The compiled FlowMeta, the gap analysis result, and the generated
            source code.

        Raises
        ------
        CompileError
            If code generation / compilation fails after retries.
        ValueError
            If gap analysis says the query is already covered.
        """
        # Step 1: Gap analysis
        gap = await self.analyse_gap(user_query)

        if gap.get("covered", True):
            raise ValueError(
                f"Query is already covered by existing flows: {gap.get('reason')}"
            )

        flow_name = gap.get("suggested_flow_name", "dynamic_flow")
        flow_prompt = gap.get("suggested_flow_prompt", str(user_query))

        # Sanitise the flow name.
        flow_name = _sanitise_name(flow_name)

        contract, downstream_name = self.resolve_downstream_contract(
            downstream_flow_route,
        )

        # Step 2+3: Generate + compile (with retry loop)
        meta, code = await self.generate_and_compile(
            flow_name, flow_prompt, user_query,
            downstream_contract=contract,
            downstream_flow_name=downstream_name,
        )

        return meta, gap, code

    # ------------------------------------------------------------------
    # Public helpers (used by meta-flow and external callers)
    # ------------------------------------------------------------------

    async def fix_code(
        self,
        code: str,
        error: str,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        project_context: str | None = None,
    ) -> str:
        """Ask the LLM to fix broken code given the compile error."""
        from flowforge.execution.llm import call_llm_api
        import json

        system = _FIX_SYSTEM.format(error=error, code=code)

        contract_hint = ""
        if downstream_contract:
            pretty = json.dumps(downstream_contract, ensure_ascii=False, indent=2)
            suffix = f" (consumed by `{downstream_flow_name}`)" if downstream_flow_name else ""
            contract_hint = (
                f"\n\nDownstream input contract{suffix} — the final step must "
                f"return a dict matching this JSON Schema:\n```json\n{pretty}\n```"
            )

        fixed = await call_llm_api(
            system_prompt=system,
            user_prompt=(
                f"Fix this FlowForge code. The user wanted: {user_query}"
                f"{contract_hint}"
                f"{self._format_project_context(project_context)}"
            ),
            llm_config=self._llm_config,
            tool_configs=self._codegen_tool_configs(),
            max_tool_rounds=3,
        )
        return _strip_markdown_fences(str(fixed))

    def check_contract_compatibility(
        self,
        meta: "FlowMeta",
        downstream_contract: dict[str, Any] | None,
    ) -> str | None:
        """Return a mismatch description string, or ``None`` when compatible.

        Runs a conservative static check: if the generated flow declares an
        ``output_schema`` anywhere on its exit path we compare its JSON Schema
        against ``downstream_contract``.  When the generated flow declares no
        schema we cannot statically detect a mismatch here — the framework
        will still validate at runtime using the original Pydantic models,
        but with accurate error messages since the contract was given to the
        LLM up-front.
        """
        if not downstream_contract:
            return None

        produced_model = _exit_output_schema(meta)
        produced_schema = _schema_to_contract(produced_model)

        if produced_schema is None:
            # Nothing declared — defer to runtime validation.  Don't fail the
            # compile attempt; the LLM prompt already contained the contract.
            return None

        expected_required = set(downstream_contract.get("required", []) or [])
        produced_props = set((produced_schema.get("properties") or {}).keys())
        missing = expected_required - produced_props
        if missing:
            return (
                "Output contract mismatch: "
                + _summarise_schema_mismatch(produced_schema, downstream_contract)
                + ". Regenerate the flow so its final step returns a dict whose "
                  "top-level keys match the downstream input contract exactly."
            )
        return None

    def _format_tool_catalog(self) -> str:
        """Return a detailed list of available tools with parameters.

        For ``FunctionTool`` entries the catalog includes parameter names,
        types, and defaults so the LLM can emit correct ``ctx.call_tool()``
        calls or ``<tool_name>`` references without guessing.
        """
        if not self._tool_configs:
            return "(no tools available)"

        from flowforge.types import MCPServer, FunctionTool, HTTPTool, ClaudeSkill
        import inspect

        lines: list[str] = []
        for tool in self._tool_configs:
            name = ""
            description = ""
            kind = "tool"
            params_info = ""

            if isinstance(tool, MCPServer):
                name = tool.name
                description = tool.description
                kind = "mcp"
            elif isinstance(tool, FunctionTool):
                name = tool.name or (
                    tool.func.__name__ if hasattr(tool.func, "__name__") else ""
                )
                description = tool.description
                kind = "function"
                params_info = _format_func_params(tool.func)
            elif isinstance(tool, HTTPTool):
                name = tool.name
                description = tool.description
                kind = "http"
            elif isinstance(tool, ClaudeSkill):
                name = tool.name or tool.skill_id
                description = tool.description
                kind = "claude-skill"

            if not name:
                continue
            desc = description or "No description provided."
            entry = f"- {name} ({kind}): {desc}"
            if params_info:
                entry += f"\n    Parameters: {params_info}"
                entry += f"\n    Call: await ctx.call_tool(\"{name}\", {_format_call_example(tool.func)})"
            elif kind == "claude-skill":
                entry += f"\n    Call: await ctx.call_llm(\"instruction <{name}>\")"
            lines.append(entry)

        return "\n".join(lines) if lines else "(no named tools available)"

    def build_code_generation_context(
        self,
        *,
        flow_name: str,
        flow_prompt: str,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
    ) -> str:
        """Build a lightweight implementation brief for dynamic codegen.

        This intentionally does **not** inspect the project filesystem or run
        shell commands. It gives the code generator enough coding guidance to
        produce useful FlowForge code while keeping dynamic generation fast.
        """
        tool_names = self._tool_names()

        # Classify tools into categories for the brief.
        _RUNTIME_PREFIXES = ("shell_", "python_import_check", "mcp_", "pip_install")
        _DOCUMENT_TOOLS = {
            "pdf_read_text", "pptx_create", "csv_read", "csv_write",
            "docx_create", "markdown_write", "chart_create",
        }
        _UTILITY_TOOLS = {
            "web_fetch_url", "json_select_fields",
            "files_read_text", "files_write_text", "files_list_dir",
        }

        runtime_tools = [n for n in tool_names if n.startswith(_RUNTIME_PREFIXES)]
        document_tools = [n for n in tool_names if n in _DOCUMENT_TOOLS]
        utility_tools = [n for n in tool_names if n in _UTILITY_TOOLS]
        domain_tools = [
            n for n in tool_names
            if n not in set(runtime_tools + document_tools + utility_tools)
        ]

        lines = [
            "Implementation brief assembled by the dynamic meta-flow.",
            "No project scan or test run was performed for this brief.",
            f"- Missing flow name: {flow_name}",
            f"- Missing flow purpose: {flow_prompt}",
            f"- User request/scope: {user_query}",
            "- Prefer one @flow with one @task and 1-3 @step functions.",
            "- Use Python code for small deterministic shaping only.",
            "- Use ctx.call_tool(name, **kwargs) for DIRECT tool calls (deterministic).",
            "- Use ctx.call_llm('instruction <tool_name>') when LLM reasoning is needed.",
            "- ALWAYS prefer existing builtin tools over writing custom code.",
            "- If a domain tool exists for external data, prefer it over raw HTTP code.",
        ]
        if domain_tools:
            lines.append(f"- Domain tools to prefer: {', '.join(domain_tools)}")
        if utility_tools:
            lines.append(
                f"- Utility tools (web, json, files): {', '.join(utility_tools)}"
            )
        if document_tools:
            lines.append(
                f"- Document tools (use ctx.call_tool for these): {', '.join(document_tools)}"
            )
        if runtime_tools:
            lines.append(
                "- Runtime tools (use only when truly needed): "
                + ", ".join(runtime_tools)
            )
        if downstream_contract:
            target = f" for `{downstream_flow_name}`" if downstream_flow_name else ""
            lines.append(
                "- Final output must match the downstream input contract"
                f"{target}; do not wrap it in an extra result envelope."
            )
        else:
            lines.append(
                "- No downstream schema is present, so return a compact, "
                "self-describing JSON dict."
            )
        return "\n".join(lines)

    def _format_dynamic_policy(self) -> str:
        if self._dynamic_options is None:
            return "No explicit DynamicRunOptions supplied."

        policy = self._dynamic_options.dependency_policy
        return "\n".join([
            f"- generated_dir: {self._dynamic_options.generated_dir}",
            f"- persist_generated: {self._dynamic_options.persist_generated}",
            f"- allow_tool_generation: {self._dynamic_options.allow_tool_generation}",
            f"- allow_codegen_tool_use: {self._dynamic_options.allow_codegen_tool_use}",
            f"- allowed_shell_modes: {self._dynamic_options.allowed_shell_modes}",
            f"- dependency_install_allowed: {policy.allow_install}",
            f"- allowed_dependency_managers: {policy.allowed_managers}",
            f"- allowed_packages: {policy.allowed_packages or '(not restricted)'}",
            f"- denied_packages: {policy.denied_packages or '(none)'}",
        ])

    def _codegen_tool_configs(self) -> list[ToolConfig] | None:
        if self._dynamic_options is None:
            return None
        if self._dynamic_options.allow_codegen_tool_use:
            return self._tool_configs
        return None

    def _tool_names(self) -> list[str]:
        from flowforge.types import MCPServer, FunctionTool, HTTPTool, ClaudeSkill

        names: list[str] = []
        for tool in self._tool_configs:
            name = ""
            if isinstance(tool, MCPServer):
                name = tool.name
            elif isinstance(tool, FunctionTool):
                name = tool.name or (
                    tool.func.__name__ if hasattr(tool.func, "__name__") else ""
                )
            elif isinstance(tool, HTTPTool):
                name = tool.name
            elif isinstance(tool, ClaudeSkill):
                name = tool.name or tool.skill_id
            if name:
                names.append(name)
        return names

    def _format_project_context(self, project_context: str | None) -> str:
        if not project_context:
            return ""
        text = str(project_context).strip()
        max_chars = 4000
        if self._dynamic_options is not None:
            max_chars = self._dynamic_options.project_context_max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return (
            "Code generation context prepared by the dynamic meta-flow "
            "(no project scan unless explicitly supplied):\n"
            f"{text}\n\n"
        )

    def _check_dependency_policy(self, dependencies: list[str]) -> str | None:
        if not dependencies or self._dynamic_options is None:
            return None

        policy = self._dynamic_options.dependency_policy
        if dependencies and not policy.allow_install:
            return (
                "Generated tool declares dependencies but dependency installation "
                "is disabled by DynamicRunOptions."
            )
        if policy.allowed_packages:
            disallowed = [
                dep for dep in dependencies if dep not in policy.allowed_packages
            ]
            if disallowed:
                return f"Dependencies are not allowed by policy: {disallowed}"
        denied = [dep for dep in dependencies if dep in policy.denied_packages]
        if denied:
            return f"Dependencies are denied by policy: {denied}"
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output artifact detection — auto-detect file output intent from user query
# ---------------------------------------------------------------------------

# Maps user intent keywords → (tool_name, file_extension, description)
_ARTIFACT_RULES: list[tuple[list[str], str, str, str]] = [
    (
        ["ppt", "pptx", "프레젠테이션", "발표 자료", "슬라이드", "presentation", "slide", "powerpoint", "deck"],
        "pptx_create",
        ".pptx",
        "PowerPoint presentation",
    ),
    (
        ["csv", "스프레드시트", "spreadsheet", "표 파일"],
        "csv_write",
        ".csv",
        "CSV spreadsheet",
    ),
    (
        ["docx", "doc", "워드", "word", "문서 파일"],
        "docx_create",
        ".docx",
        "Word document",
    ),
    (
        ["차트", "chart", "그래프", "graph", "plot", "시각화", "visualization"],
        "chart_create",
        ".png",
        "chart image",
    ),
    (
        ["마크다운", "markdown", "md 파일", ".md"],
        "markdown_write",
        ".md",
        "Markdown document",
    ),
    (
        ["pdf"],
        "pdf_read_text",
        ".pdf",
        "PDF text extraction (read-only)",
    ),
    (
        ["이미지", "image", "그림", "picture", "사진", "photo", "png", "jpg",
         "일러스트", "illustration", "배너", "banner", "썸네일", "thumbnail"],
        "image_create",
        ".png",
        "image file",
    ),
]


def detect_output_artifacts(
    user_query: str,
    available_tools: list[str] | None = None,
) -> list[dict[str, str]]:
    """Detect which file-output builtin tools the user's request implies.

    Scans the *user_query* for keywords that suggest the user wants a
    specific output format (PPT, CSV, chart, etc.) and returns a list of
    matching artifact descriptors.

    Each descriptor is a dict with:
    - ``tool``: builtin tool name (e.g. ``"pptx_create"``)
    - ``extension``: file extension (e.g. ``".pptx"``)
    - ``description``: human-readable format name

    Only tools that are actually available (present in *available_tools*)
    are returned. When *available_tools* is ``None``, all matches are
    returned.

    Returns
    -------
    list[dict[str, str]]
        Detected artifacts, may be empty.
    """
    query_lower = user_query.lower()
    detected: list[dict[str, str]] = []
    seen_tools: set[str] = set()

    for keywords, tool_name, ext, desc in _ARTIFACT_RULES:
        if tool_name in seen_tools:
            continue
        if available_tools is not None and tool_name not in available_tools:
            continue
        for kw in keywords:
            if kw in query_lower:
                detected.append({
                    "tool": tool_name,
                    "extension": ext,
                    "description": desc,
                })
                seen_tools.add(tool_name)
                break

    return detected


def _format_artifact_instructions(artifacts: list[dict[str, str]]) -> str:
    """Build a prompt block instructing the LLM to add render steps."""
    if not artifacts:
        return ""

    lines = [
        "## Required output artifacts",
        "The user's request implies the following file output(s). You MUST "
        "include a FINAL step in the generated flow that creates each artifact "
        "using `await ctx.call_tool(...)`. Do NOT skip this step.",
        "",
    ]
    for i, art in enumerate(artifacts, 1):
        lines.append(
            f"{i}. **{art['description']}** ({art['extension']}): "
            f"use `await ctx.call_tool(\"{art['tool']}\", ...)`"
        )

    lines.extend([
        "",
        "Pattern for file-generating final step:",
        "```python",
        "@step(order=N, prompt=\"render output artifact\")",
        "async def render_output(ctx):",
        "    import json",
        "    data = ctx.previous_results.get(N-1)  # previous step output",
        "    result = await ctx.call_tool(\"<tool_name>\", path=\"output/file.ext\", ...)",
        "    return {\"artifact_path\": result.get(\"path\", \"\"), \"ok\": result.get(\"ok\", False)}",
        "```",
        "",
        "IMPORTANT: The render step should transform the previous step's output "
        "into the format expected by the tool. For example, for pptx_create, "
        "convert the data into a JSON array of slide objects with 'title' and "
        "'bullets' keys.",
    ])
    return "\n".join(lines)


def _format_func_params(func: Any) -> str:
    """Return a compact parameter signature string for a tool function."""
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return "(unknown)"

    parts: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls", "ctx"):
            continue
        annotation = ""
        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            annotation = f": {ann.__name__}" if hasattr(ann, "__name__") else f": {ann}"
        default = ""
        if param.default is not inspect.Parameter.empty:
            default = f" = {param.default!r}"
        parts.append(f"{pname}{annotation}{default}")
    return f"({', '.join(parts)})"


def _format_call_example(func: Any) -> str:
    """Return a compact keyword-argument example for ctx.call_tool()."""
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return "..."

    parts: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls", "ctx"):
            continue
        if param.default is inspect.Parameter.empty:
            parts.append(f'{pname}=...')
        # Skip optional params in the example
    return ", ".join(parts) if parts else "..."


def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` fences if the LLM wrapped the code."""
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _sanitise_name(name: str) -> str:
    """Ensure the name is a valid Python identifier in snake_case."""
    import re
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    if not name:
        name = "dynamic_flow"
    # Ensure it doesn't start with a digit.
    if name[0].isdigit():
        name = f"flow_{name}"
    return name
