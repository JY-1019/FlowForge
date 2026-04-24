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
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.annotations.metadata import FlowMeta, GlobalMeta
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.types import LLMConfig
    from flowforge.doc.models import AnyDoc

logger = logging.getLogger(__name__)

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
    2. All classes at MODULE LEVEL (never inside functions)
    3. Step functions use `async def name(ctx):` signature
    4. Use `await ctx.call_llm("instruction")` for AI-powered steps
    5. Use `return {{"key": "value"}}` for code-only steps
    6. The top-level class must be decorated with @flow
    7. Do NOT use @global_config — only generate the @flow and its children
    8. flow name must be: {flow_name}
    9. Include input handling via `ctx.input` (dict or Pydantic model)
    10. Do NOT import pydantic or define schemas — keep it simple

    STRUCTURE:
    A @flow can contain BOTH child @flow classes AND @task classes.
    A @task can contain child @task classes OR @step functions (not both).
    Only leaf @task classes contain @step functions.

    SIMPLE TEMPLATE (single task):
    ```python
    from flowforge import flow, task, step

    @flow(name="{flow_name}", prompt="{flow_prompt}")
    class {class_name}:
        @task(name="main_task", prompt="description")
        class MainTask:
            @step(order=1, prompt="what this step does")
            async def step_one(ctx):
                result = await ctx.call_llm("instruction based on {{field}}")
                return result
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
    ) -> None:
        self._llm_config = llm_config
        self._dag = dag
        self._docs = docs or {}

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
    ) -> str:
        """Ask the LLM to generate FlowForge decorator code for a new flow.

        Returns the generated Python source code as a string.
        """
        from flowforge.execution.llm import call_llm_api

        # Build a PascalCase class name from the snake_case flow name.
        class_name = "".join(
            part.capitalize() for part in flow_name.split("_")
        ) + "Flow"

        system = _CODEGEN_SYSTEM.format(
            flow_name=flow_name,
            flow_prompt=flow_prompt,
            class_name=class_name,
        )

        user_prompt = (
            f"User query that triggered this flow generation:\n"
            f"{user_query}\n\n"
            f"Flow name: {flow_name}\n"
            f"Flow purpose: {flow_prompt}\n"
            f"Class name: {class_name}\n\n"
            f"Generate the complete FlowForge code."
        )

        code = await call_llm_api(
            system_prompt=system,
            user_prompt=user_prompt,
            llm_config=self._llm_config,
        )

        # Strip markdown fences if the LLM included them.
        code = _strip_markdown_fences(str(code))
        return code

    def compile_flow_code(self, code: str) -> FlowMeta:
        """Compile generated Python code into a ``FlowMeta``.

        The code is written to a temporary file, imported as a module, and
        scanned for a ``@flow``-decorated class.

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
            If no ``@flow``-decorated class is found or the code has errors.
        """
        from flowforge.annotations.decorators import _FLOW_ATTR
        from flowforge.errors import CompileError

        # Write to a temp file and import it.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="flowforge_dynamic_",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            module_name = f"_flowforge_dynamic_{temp_path.stem}"
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
            # Clean up the temp file and sys.modules entry.
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            sys.modules.pop(module_name, None)

    async def generate_and_compile(
        self,
        flow_name: str,
        flow_prompt: str,
        user_query: str | Any,
    ) -> FlowMeta:
        """Generate code, compile it, and retry on errors.

        Makes up to ``_MAX_RETRIES`` attempts.  On each failure the error
        message is fed back to the LLM for self-correction.

        Returns
        -------
        FlowMeta
            The compiled metadata for the new flow.
        """
        from flowforge.errors import CompileError

        code = await self.generate_flow_code(flow_name, flow_prompt, user_query)
        logger.info("generated code for flow '%s' (%d chars)", flow_name, len(code))
        logger.debug("generated code:\n%s", code)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                meta = self.compile_flow_code(code)
                logger.info(
                    "flow '%s' compiled successfully on attempt %d",
                    flow_name, attempt + 1,
                )
                return meta
            except (CompileError, Exception) as e:
                last_error = e
                logger.warning(
                    "compile attempt %d/%d failed for '%s': %s",
                    attempt + 1, _MAX_RETRIES, flow_name, e,
                )
                if attempt + 1 < _MAX_RETRIES:
                    code = await self._fix_code(code, str(e), user_query)
                    logger.debug("fixed code (attempt %d):\n%s", attempt + 2, code)

        raise CompileError(
            f"Failed to compile dynamic flow '{flow_name}' after "
            f"{_MAX_RETRIES} attempts. Last error: {last_error}"
        )

    async def run_full_pipeline(
        self,
        user_query: str | Any,
    ) -> tuple[FlowMeta, dict[str, Any]]:
        """Full pipeline: gap analysis → code gen → compile.

        Returns
        -------
        tuple[FlowMeta, dict]
            The compiled FlowMeta and the gap analysis result.

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

        # Step 2+3: Generate + compile (with retry loop)
        meta = await self.generate_and_compile(flow_name, flow_prompt, user_query)

        return meta, gap

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fix_code(
        self, code: str, error: str, user_query: str | Any,
    ) -> str:
        """Ask the LLM to fix broken code given the compile error."""
        from flowforge.execution.llm import call_llm_api

        system = _FIX_SYSTEM.format(error=error, code=code)
        fixed = await call_llm_api(
            system_prompt=system,
            user_prompt=f"Fix this FlowForge code. The user wanted: {user_query}",
            llm_config=self._llm_config,
        )
        return _strip_markdown_fences(str(fixed))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
