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
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

# Imported at module scope both for internal use and to preserve historical
# private test imports from flowforge.dynamic.generator.
from flowforge.dynamic.contracts import (
    _entry_input_schema,
    _exit_output_schema,
    _schema_to_contract,
    _summarise_schema_mismatch,
)
from flowforge.dynamic.generated_code import (
    _normalise_generated_flow_code,
    _strip_markdown_fences,
)
from flowforge.dynamic.prompts import (
    _CODEGEN_SYSTEM,
    _FIX_SYSTEM,
    _GAP_ANALYSIS_SYSTEM,
    _PLAN_SYNTHESIS_SYSTEM,
    _TOOL_CODEGEN_SYSTEM,
)
from flowforge.dynamic.safety import _validate_generated_ast

if TYPE_CHECKING:
    from flowforge.annotations.metadata import FlowMeta, GlobalMeta, TaskMeta
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.types import LLMConfig, ToolConfig
    from flowforge.doc.models import AnyDoc

logger = logging.getLogger(__name__)


# Maximum compile-retry attempts when LLM-generated code fails validation.
_MAX_RETRIES = 3


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
        # Cache: (query_str, flow_signature) -> gap_analysis result.
        # The flow signature changes whenever flows are added or their docs
        # change, which correctly invalidates a cached "covered=true" result.
        self._gap_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Truncation budgets for the gap-analysis prompt.  These keep the
    # request token cost bounded even on projects with hundreds of flows.
    _GAP_PER_FLOW_MAX_CHARS = 240
    _GAP_FLOWS_TOTAL_MAX_CHARS = 4000

    async def analyse_gap(self, user_query: str | Any) -> dict[str, Any]:
        """Check if the user query is covered by existing flows.

        Returns a dict with ``covered`` (bool), ``reason``, and — when
        ``covered`` is False — ``suggested_flow_name`` and
        ``suggested_flow_prompt``.
        """
        from flowforge.llm.caller import call_with_tool
        from flowforge.schema.dag import NodeType

        query_str = str(user_query)

        # Build a summary of existing flows for the LLM, with per-entry and
        # total length caps so the prompt cost stays bounded.
        flow_entries: list[str] = []
        for node in self._dag.get_all_nodes():
            if node.type != NodeType.FLOW:
                continue
            doc = self._docs.get(node.id)
            summary = getattr(doc, "summary", "") if doc else ""
            prompt_text = getattr(node.meta, "prompt", "")
            entry = (
                f"- {node.id}: {prompt_text}"
                + (f" (doc: {summary})" if summary else "")
            )
            if len(entry) > self._GAP_PER_FLOW_MAX_CHARS:
                entry = entry[: self._GAP_PER_FLOW_MAX_CHARS - 1].rstrip() + "…"
            flow_entries.append(entry)

        flow_signature_basis = "\n".join(flow_entries)
        flows_text = flow_signature_basis if flow_entries else "(no flows)"
        if len(flows_text) > self._GAP_FLOWS_TOTAL_MAX_CHARS:
            flows_text = (
                flows_text[: self._GAP_FLOWS_TOTAL_MAX_CHARS].rstrip()
                + "\n[…flow list truncated by FlowForge gap-analysis budget]"
            )

        # Cache check: same query against the same flow set short-circuits
        # the LLM round-trip entirely.
        cache_key = (query_str, flow_signature_basis)
        cached = self._gap_cache.get(cache_key)
        if cached is not None:
            logger.info("gap analysis cache hit for query (%d chars)", len(query_str))
            return dict(cached)

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
        if isinstance(result, dict):
            self._gap_cache[cache_key] = dict(result)
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
        tool_catalog = self._format_tool_catalog(
            user_query=user_query,
            artifacts=artifacts,
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

        # Strip markdown fences if the LLM included them and repair small,
        # common decorator omissions before the compile/retry loop.
        code = _normalise_generated_flow_code(str(code), flow_name, flow_prompt)
        return code

    # ------------------------------------------------------------------
    # Plan-driven synthesis (Phase 4 of the dynamic pipeline)
    # ------------------------------------------------------------------

    async def generate_flow_code_from_plan(
        self,
        *,
        plan: Any,
        selection: Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        project_context: str | None = None,
    ) -> str:
        """Synthesise FlowForge code from a plan + capability decision.

        Unlike :meth:`generate_flow_code`, the LLM is told *what* every
        step must do (plan) and *which* capability it must use
        (selection).  The system prompt focuses purely on translation
        rules, which keeps the prompt small and the failure modes narrow.
        """
        from flowforge.execution.llm import call_llm_api
        import json

        plan_text = _format_plan_for_synthesis(plan)
        capability_text = _format_selection_for_synthesis(selection)

        contract_block = ""
        if downstream_contract:
            pretty = json.dumps(
                downstream_contract, ensure_ascii=False, indent=2,
            )
            downstream_hint = (
                f" (consumed by `{downstream_flow_name}`)"
                if downstream_flow_name else ""
            )
            contract_block = (
                f"## Downstream input contract{downstream_hint}\n"
                f"The final step of `{plan.flow_name}` MUST return a dict "
                f"matching this JSON Schema exactly:\n"
                f"```json\n{pretty}\n```\n\n"
            )

        user_prompt = (
            f"## Workflow plan\n{plan_text}\n\n"
            f"## Capability decision\n{capability_text}\n\n"
            f"{contract_block}"
            f"{self._format_project_context(project_context)}"
            "Translate the plan + capabilities into a FlowForge module.  "
            "Return ONLY the Python code."
        )

        # Generous budget — agentic synthesis is slow regardless and a
        # truncated module is worse than a slow one.
        synthesis_config = self._llm_config_for_synthesis(min_tokens=12000)

        code = await call_llm_api(
            system_prompt=_PLAN_SYNTHESIS_SYSTEM,
            user_prompt=user_prompt,
            llm_config=synthesis_config,
            tool_configs=self._codegen_tool_configs(),
            max_tool_rounds=3,
        )
        return _normalise_generated_flow_code(
            str(code), plan.flow_name, plan.flow_prompt,
        )

    def _llm_config_for_synthesis(self, min_tokens: int):
        """Return an ``LLMConfig`` with at least ``min_tokens`` budget."""
        cfg = self._llm_config
        current = getattr(cfg, "max_tokens", 0) or 0
        if current >= min_tokens:
            return cfg
        if hasattr(cfg, "model_copy"):
            return cfg.model_copy(update={"max_tokens": min_tokens})
        # Fallback for non-pydantic configs in tests.
        try:
            cfg.max_tokens = min_tokens  # type: ignore[attr-defined]
        except Exception:
            pass
        return cfg

    async def generate_and_compile_from_plan(
        self,
        *,
        plan: Any,
        selection: Any,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        project_context: str | None = None,
    ) -> tuple[FlowMeta, str]:
        """Synthesise → compile → retry from a plan + capability decision."""
        from flowforge.errors import CompileError

        code = await self.generate_flow_code_from_plan(
            plan=plan,
            selection=selection,
            downstream_contract=downstream_contract,
            downstream_flow_name=downstream_flow_name,
            project_context=project_context,
        )
        logger.info(
            "synthesised code for flow '%s' (%d chars)",
            plan.flow_name, len(code),
        )
        logger.debug("synthesised code:\n%s", code)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                meta = self.compile_flow_code(code)
                compatibility_error = self.check_contract_compatibility(
                    meta, downstream_contract,
                )
                if compatibility_error is not None:
                    raise CompileError(compatibility_error)
                tool_ref_error = self.check_tool_ref_validity(code)
                if tool_ref_error is not None:
                    raise CompileError(tool_ref_error)
                tool_usage_error = self.check_required_tool_usage(
                    code, user_query,
                )
                if tool_usage_error is not None:
                    raise CompileError(tool_usage_error)
                quality_error = self.check_generated_code_quality(
                    code, user_query,
                )
                if quality_error is not None:
                    raise CompileError(quality_error)
                logger.info(
                    "synthesised flow '%s' compiled on attempt %d",
                    plan.flow_name, attempt + 1,
                )
                return meta, code
            except (CompileError, Exception) as exc:
                last_error = exc
                logger.warning(
                    "synthesis attempt %d/%d failed for '%s': %s",
                    attempt + 1, _MAX_RETRIES, plan.flow_name, exc,
                )
                if attempt + 1 < _MAX_RETRIES:
                    code = await self.fix_code(
                        code, str(exc), user_query,
                        downstream_contract=downstream_contract,
                        downstream_flow_name=downstream_flow_name,
                        project_context=project_context,
                    )
                    code = _normalise_generated_flow_code(
                        code, plan.flow_name, plan.flow_prompt,
                    )
                    logger.debug(
                        "fixed synthesised code (attempt %d):\n%s",
                        attempt + 2, code,
                    )

        raise CompileError(
            f"Failed to synthesise dynamic flow '{plan.flow_name}' after "
            f"{_MAX_RETRIES} attempts. Last error: {last_error}"
        )

    async def generate_compile_and_persist_from_plan(
        self,
        *,
        plan: Any,
        selection: Any,
        user_query: str | Any,
        downstream_contract: dict[str, Any] | None = None,
        downstream_flow_name: str | None = None,
        downstream_flow_route: str = "",
        project_context: str | None = None,
    ) -> tuple[FlowMeta, str]:
        """Plan-driven equivalent of :meth:`generate_compile_and_persist`."""
        meta, code = await self.generate_and_compile_from_plan(
            plan=plan,
            selection=selection,
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
                    meta = getattr(obj, _FLOW_ATTR)
                    from flowforge.dynamic.defaults import apply_dynamic_defaults

                    apply_dynamic_defaults(meta, self._dynamic_options)
                    return meta

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
                tool_ref_error = self.check_tool_ref_validity(code)
                if tool_ref_error is not None:
                    raise CompileError(tool_ref_error)
                tool_usage_error = self.check_required_tool_usage(
                    code, user_query,
                )
                if tool_usage_error is not None:
                    raise CompileError(tool_usage_error)
                quality_error = self.check_generated_code_quality(
                    code, user_query,
                )
                if quality_error is not None:
                    raise CompileError(quality_error)
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
                    code = _normalise_generated_flow_code(
                        code, flow_name, flow_prompt,
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
                f"\n\nAvailable tools:\n{self._format_tool_catalog()}"
                f"\n\nDynamic policy:\n{self._format_dynamic_policy()}"
            ),
            llm_config=self._llm_config,
            tool_configs=self._codegen_tool_configs(),
            max_tool_rounds=3,
        )
        fixed_code = _strip_markdown_fences(str(fixed))
        return fixed_code

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

    def check_tool_ref_validity(self, code: str) -> str | None:
        """Reject generated code that references unknown tool/skill names.

        Scans every ``<name>`` marker inside ``ctx.call_llm(...)`` strings
        and every ``ctx.call_tool("name", ...)`` call.  If a name does not
        match a registered ``ToolConfig`` (FunctionTool, MCPServer, HTTPTool,
        ClaudeSkill, AgentSkill), returns an error message that the retry
        loop feeds back to the LLM.

        Empty registry → no validation (the loop relies on other checks).
        """
        registered = set(self._tool_names())
        registered.discard("")
        if not registered:
            return None

        _, runtime_used = _collect_generated_tool_usage(
            code,
            known_tool_refs=registered,
        )
        unknown = sorted(name for name in runtime_used if name not in registered)
        if not unknown:
            return None

        catalog = ", ".join(sorted(registered)) or "(none)"
        return (
            "Generated flow references unknown tool/skill names: "
            + ", ".join(unknown)
            + f". Use ONLY these registered names verbatim — do not invent "
            f"or paraphrase: {catalog}"
        )

    def check_required_tool_usage(
        self,
        code: str,
        user_query: str | Any,
    ) -> str | None:
        """Validate required tool declarations and runtime usage.

        Dynamic examples can pass ``{"required_tools": [...]}`` in the user
        query.  When present, generated code must record those tools in
        decorator ``tools=[...]`` scopes.  Executable tools must also be used
        at runtime via ``ctx.call_tool("name", ...)`` or LLM-mediated
        ``ctx.call_llm("... <name> ...")``.
        """
        required = _extract_required_tools(user_query)
        if not required:
            return None

        scoped, runtime_used = _collect_generated_tool_usage(
            code,
            known_tool_refs=set(required) | set(self._tool_names()),
        )
        missing_scope = [name for name in required if name not in scoped]

        prompt_only = self._prompt_only_tool_names()
        runtime_required = [name for name in required if name not in prompt_only]
        missing_runtime = [
            name for name in runtime_required
            if name not in runtime_used
        ]

        messages: list[str] = []
        if missing_scope:
            messages.append(
                "Required tools are missing from decorator tools=[...] scopes: "
                + ", ".join(missing_scope)
            )
        if missing_runtime:
            messages.append(
                "Required executable tools are not invoked at runtime via "
                "ctx.call_tool(...) or ctx.call_llm(... <tool> ...): "
                + ", ".join(missing_runtime)
            )
        if not messages:
            return None
        return (
            "Generated flow did not satisfy required tool usage. "
            + " ".join(messages)
        )

    def check_generated_code_quality(
        self,
        code: str,
        user_query: str | Any,
    ) -> str | None:
        """Reject common dynamic-code patterns that compile but ship junk.

        The compiler already catches syntax errors, unknown tool names, and
        missing required tools.  This pass catches higher-level anti-patterns
        observed in generated examples: reading stable request fields from
        ``ctx.input`` after step 1, swallowing JSON parse errors into empty
        dicts, substituting fallback stub file bodies, and writing files
        without raising when a tool returns ``ok=False``.
        """
        return _check_generated_code_quality(code, user_query)

    def _format_tool_catalog(
        self,
        user_query: str | Any | None = None,
        artifacts: list[dict[str, str]] | None = None,
    ) -> str:
        """Return a detailed list of available tools with parameters.

        For ``FunctionTool`` entries the catalog includes parameter names,
        types, and defaults so the LLM can emit correct ``ctx.call_tool()``
        calls or ``<tool_name>`` references without guessing.
        """
        if not self._tool_configs:
            return "(no tools available)"

        from flowforge.types import MCPServer, FunctionTool, HTTPTool, ClaudeSkill, AgentSkill
        import inspect

        lines: list[str] = []
        selected_tools, omitted = self._select_tool_configs_for_codegen(
            user_query=user_query,
            artifacts=artifacts,
        )
        for tool in selected_tools:
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
            elif isinstance(tool, AgentSkill):
                name = tool.name
                description = tool.description
                kind = "agent-skill"

            if not name:
                continue
            desc = description or "No description provided."
            entry = f"- {name} ({kind}): {desc}"
            entry += f"\n    Decorator scope: tools=[\"{name}\"]"
            if params_info:
                entry += f"\n    Parameters: {params_info}"
                entry += f"\n    Call: await ctx.call_tool(\"{name}\", {_format_call_example(tool.func)})"
            elif kind in {"claude-skill", "agent-skill"}:
                entry += f"\n    Call: await ctx.call_llm(\"instruction <{name}>\")"
            lines.append(entry)

        text = "\n".join(lines) if lines else "(no named tools available)"
        if omitted:
            text += (
                "\n- ... "
                f"{omitted} lower-relevance tool(s) omitted from this compact "
                "catalog to save tokens. Add them to required_tools to force inclusion."
            )

        max_chars = 0
        if self._dynamic_options is not None:
            max_chars = self._dynamic_options.codegen_tool_catalog_max_chars
        if max_chars > 0 and len(text) > max_chars:
            text = (
                text[:max_chars].rstrip()
                + "\n...[tool catalog truncated by codegen_tool_catalog_max_chars]"
            )
        return text

    def _select_tool_configs_for_codegen(
        self,
        user_query: str | Any | None = None,
        artifacts: list[dict[str, str]] | None = None,
    ) -> tuple[list[ToolConfig], int]:
        """Select the most relevant tools for codegen to reduce prompt size."""
        if not user_query or self._dynamic_options is None:
            return list(self._tool_configs), 0

        required = set(_extract_required_tools(user_query))
        eligible_tools = [
            tool for tool in self._tool_configs
            if _tool_config_name(tool) != "claude_skill"
            or "claude_skill" in required
        ]

        max_tools = self._dynamic_options.codegen_tool_catalog_max_tools
        if max_tools <= 0 or len(eligible_tools) <= max_tools:
            omitted = len(self._tool_configs) - len(eligible_tools)
            return list(eligible_tools), omitted

        required.update(
            art.get("tool", "")
            for art in (artifacts or [])
            if art.get("tool")
        )

        query_text = str(user_query).lower()
        query_tokens = {
            token
            for token in _word_tokens(query_text)
            if len(token) >= 3
        }
        wants_mcp = any(
            term in query_text
            for term in ("mcp", "figma", "playwright", "browser", "design")
        )

        scored: list[tuple[int, int, ToolConfig]] = []
        for index, tool in enumerate(eligible_tools):
            name = _tool_config_name(tool)
            desc = getattr(tool, "description", "") or ""
            haystack = f"{name} {desc}".lower()

            score = 0
            if name in required:
                score += 1000
            if name in self._prompt_only_tool_names():
                score += 450
            if name in {"mcp_start_server", "mcp_register_server"} and wants_mcp:
                score += 420
            if name in {"web_fetch_url", "files_write_text", "json_select_fields"}:
                score += 120
            if name in {"files_read_text", "files_list_dir"}:
                score += 80
            for token in query_tokens:
                if token in haystack:
                    score += 25
            if name.startswith("shell_") and any(
                term in query_text for term in ("install", "build", "npm", "pnpm", "yarn")
            ):
                score += 150
            if name in {"pptx_create", "markdown_write", "csv_write", "docx_create"}:
                if any(term in query_text for term in ("ppt", "deck", "markdown", "csv", "docx", "report")):
                    score += 120

            scored.append((score, -index, tool))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [tool for score, _, tool in scored if score > 0][:max_tools]
        if len(selected) < min(max_tools, 4):
            selected_names = {_tool_config_name(tool) for tool in selected}
            for _, _, tool in scored:
                name = _tool_config_name(tool)
                if name in selected_names:
                    continue
                selected.append(tool)
                selected_names.add(name)
                if len(selected) >= min(max_tools, 4):
                    break

        selected_names = {_tool_config_name(tool) for tool in selected}
        omitted = len([
            tool for tool in self._tool_configs
            if _tool_config_name(tool) not in selected_names
        ])
        return selected, omitted

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
            "- Prefer one @flow with one @task and focused @step functions.",
            "- For file/project generation, use one author/write phase per substantial file; never one giant JSON file map.",
            "- Normalise stable request fields into ctx.shared_data in the first step; later steps should not read project_dir/target_url from ctx.input.",
            "- Give every @task and @step a specific prompt; avoid placeholder prompts.",
            "- Put intended tool names in decorator tools=[\"tool_name\"] as string references.",
            "- Use Python code for small deterministic shaping only.",
            "- Use ctx.call_tool(name, **kwargs) for DIRECT tool calls (deterministic).",
            "- Use ctx.call_llm('instruction <tool_name>') when LLM reasoning is needed; the angle-bracket name is required for LLM-mediated tools.",
            "- ALWAYS prefer existing builtin tools over writing custom code.",
            "- Always raise on builtin tool results with ok=False; do not silently return failed tool payloads.",
            "- Before files_write_text, reject empty/trivial content; after writing text artefacts, verify with files_read_text.",
            "- For binary/document artefacts (.docx, .pptx, .pdf, images, charts), verify with creator-tool size/count fields or files_list_dir; never files_read_text.",
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
            "- generated_step_timeout_seconds: "
            f"{self._dynamic_options.generated_step_timeout_seconds}",
            f"- allowed_shell_modes: {self._dynamic_options.allowed_shell_modes}",
            f"- dependency_install_allowed: {policy.allow_install}",
            f"- allowed_dependency_managers: {policy.allowed_managers}",
            f"- allowed_packages: {policy.allowed_packages or '(not restricted)'}",
            f"- denied_packages: {policy.denied_packages or '(none)'}",
            "- mcp_server_commands: "
            f"{list(self._dynamic_options.mcp_server_commands.keys()) or '(none)'}",
            "- mcp_server_urls: "
            f"{self._dynamic_options.mcp_server_urls or '(none)'}",
            "- mcp_server_tools: "
            f"{self._dynamic_options.mcp_server_tools or '(none)'}",
            "- mcp_server_headers: "
            f"{list(self._dynamic_options.mcp_server_headers.keys()) or '(none)'}",
            "- codegen_tool_catalog_max_tools: "
            f"{self._dynamic_options.codegen_tool_catalog_max_tools}",
            "- codegen_tool_catalog_max_chars: "
            f"{self._dynamic_options.codegen_tool_catalog_max_chars}",
        ])

    def _codegen_tool_configs(self) -> list[ToolConfig] | None:
        if self._dynamic_options is None:
            return None
        if self._dynamic_options.allow_codegen_tool_use:
            return self._tool_configs

        from flowforge.types import AgentSkill, ClaudeSkill

        prompt_only: list[ToolConfig] = []
        for tool in self._tool_configs:
            if isinstance(tool, AgentSkill):
                prompt_only.append(tool)
            elif (
                isinstance(tool, ClaudeSkill)
                and self._llm_config.provider == "anthropic"
            ):
                prompt_only.append(tool)
        return prompt_only or None

    def _tool_names(self) -> list[str]:
        from flowforge.types import MCPServer, FunctionTool, HTTPTool, ClaudeSkill, AgentSkill

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
            elif isinstance(tool, AgentSkill):
                name = tool.name
            if name:
                names.append(name)
        names.extend(self._declared_mcp_tool_names())
        seen: set[str] = set()
        unique_names: list[str] = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            unique_names.append(name)
        return unique_names

    def _declared_mcp_tool_names(self) -> list[str]:
        """Return tool names declared on dynamic MCP server options.

        Dynamic flows register these tools at runtime with
        ``mcp_register_server`` before using them via ``ctx.call_llm``. They
        are not present in the static ToolConfig list at generation time, but
        they are still valid names when they come from
        DynamicRunOptions.mcp_server_tools.
        """
        if self._dynamic_options is None:
            return []

        declared = getattr(self._dynamic_options, "mcp_server_tools", {}) or {}
        names: list[str] = []
        seen: set[str] = set()
        for tool_names in declared.values():
            for name in tool_names or []:
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    def _prompt_only_tool_names(self) -> set[str]:
        from flowforge.types import AgentSkill, ClaudeSkill

        names: set[str] = set()
        for tool in self._tool_configs:
            if isinstance(tool, ClaudeSkill):
                name = tool.name or tool.skill_id
            elif isinstance(tool, AgentSkill):
                name = tool.name
            else:
                continue
            if name:
                names.add(name)
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

_STABLE_REQUEST_KEYS: set[str] = {
    "project_dir",
    "project_root",
    "output_dir",
    "target_url",
    "reference_url",
    "project_name",
    "request",
    "required_tools",
    "expected_commands",
    "expected_output",
}

_CONTENT_NAME_HINTS: tuple[str, ...] = (
    "content",
    "body",
    "html",
    "css",
    "js",
    "json",
    "markdown",
    "source",
    "text",
)


def _check_generated_code_quality(
    code: str,
    user_query: str | Any,
) -> str | None:
    """Return a human-readable quality error for generated FlowForge code.

    These checks are intentionally conservative and targeted at code produced
    by the dynamic generator.  They reject patterns that compile successfully
    but commonly produce empty clone-coding artefacts or silent partial runs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # The normal compile path will report the syntax error.
        return None

    for func in _iter_step_functions(tree):
        order = _step_order(func)
        if order is not None and order > 1:
            stable_keys = sorted(_ctx_input_stable_key_reads(func))
            if stable_keys:
                return (
                    f"Step {func.name!r} reads stable request field(s) "
                    f"{stable_keys} from ctx.input at order {order}. "
                    "For order > 1, ctx.input is the previous step output. "
                    "Read these fields from ctx.task_input, ctx.flow_input, "
                    "or ctx.shared_data instead."
                )

        if _swallows_json_parse_to_empty(func):
            return (
                f"Step {func.name!r} catches a JSON parse failure and "
                "falls back to an empty dict/list. Generated flows must raise "
                "on parse failure so runtime repair can fix the malformed "
                "LLM output instead of writing zero files."
            )

        unsafe_llm_text = _unsafe_call_llm_text_handling(func)
        if unsafe_llm_text:
            return (
                f"Step {func.name!r} treats ctx.call_llm() output as text "
                f"without first handling parsed dict/list responses: "
                f"{unsafe_llm_text}. FlowForge auto-parses JSON-looking "
                "LLM responses, so generated code must branch on isinstance "
                "before calling string methods or json.loads."
            )

        placeholder = _placeholder_success_literal(func)
        if placeholder:
            return (
                f"Step {func.name!r} contains a placeholder failure string "
                f"{placeholder!r} outside a raise path. Generated flows must "
                "raise when real web/listing data cannot be extracted; they "
                "must not return placeholder stories or summaries as success."
            )

        if _uses_project_file_map_antipattern(func):
            return (
                f"Step {func.name!r} uses the banned project-file map pattern "
                "(one JSON object/loop for many files). Author/write one "
                "substantial file at a time and verify the entrypoint."
            )

        if _calls_tool(func, "files_write_text"):
            fallback_name = _content_fallback_assignment(func)
            if fallback_name:
                return (
                    f"Step {func.name!r} substitutes fallback content for "
                    f"{fallback_name!r}. Empty/trivial authored content must "
                    "raise before files_write_text; do not write placeholder "
                    "HTML/CSS/JS/package stubs."
                )
            if not _function_has_content_guard(func):
                return (
                    f"Step {func.name!r} calls files_write_text without a "
                    "non-empty/trivial-content guard. Check len(content) or "
                    "truthiness for the file body and raise before writing."
                )
            if not _function_has_raise(func):
                return (
                    f"Step {func.name!r} calls files_write_text without any "
                    "raise path. Check content length before writing and "
                    "raise RuntimeError when files_write_text returns ok=False."
                )
            if not _function_checks_ok(func):
                return (
                    f"Step {func.name!r} calls files_write_text without "
                    "checking result.get('ok'). Builtin tool failures must "
                    "raise instead of being returned as successful step output."
                )

    _, runtime_used = _collect_generated_tool_usage(code)
    required_tools = set(_extract_required_tools(user_query))
    if (
        "files_write_text" in runtime_used
        and "files_read_text" in required_tools
        and "files_read_text" not in runtime_used
    ):
        return (
            "Generated flow writes files but does not read any file back with "
            "files_read_text even though files_read_text is required. Add a "
            "verification step that reads the entrypoint and rejects empty "
            "or placeholder-only output."
        )

    return None


def _iter_step_functions(tree: ast.AST) -> list[ast.AsyncFunctionDef]:
    steps: list[ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if _step_order(node) is not None:
            steps.append(node)
    return steps


def _step_order(func: ast.AsyncFunctionDef) -> int | None:
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if _call_name(decorator.func) != "step":
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "order":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                return value.value
    return None


def _ctx_input_stable_key_reads(func: ast.AsyncFunctionDef) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            call = node.func
            if (
                isinstance(call, ast.Attribute)
                and call.attr == "get"
                and _is_ctx_attr(call.value, "input")
                and node.args
            ):
                key = _literal_str(node.args[0])
                if key in _STABLE_REQUEST_KEYS:
                    keys.add(key)
        elif isinstance(node, ast.Subscript) and _is_ctx_attr(node.value, "input"):
            key = _literal_str(node.slice)
            if key in _STABLE_REQUEST_KEYS:
                keys.add(key)
    return keys


def _is_ctx_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctx"
    )


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _swallows_json_parse_to_empty(func: ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        if not _nodes_call_json_loads(node.body):
            continue
        for handler in node.handlers:
            if _body_assigns_empty_collection(handler.body):
                return True
    return False


def _unsafe_call_llm_text_handling(func: ast.AsyncFunctionDef) -> str:
    llm_vars = _call_llm_assigned_names(func)
    if not llm_vars:
        return ""

    guarded_as_str = _isinstance_guarded_names(func, "str")
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"strip", "split", "splitlines"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in llm_vars
                and node.func.value.id not in guarded_as_str
            ):
                return f"{node.func.value.id}.{node.func.attr}()"
            if _dotted_name(node.func) in {"json.loads", "_json.loads"}:
                if (
                    node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in llm_vars
                    and node.args[0].id not in guarded_as_str
                ):
                    return f"json.loads({node.args[0].id})"
            if _dotted_name(node.func) in {"re.search", "re.findall", "re.sub"}:
                for arg in node.args[1:]:
                    if (
                        isinstance(arg, ast.Name)
                        and arg.id in llm_vars
                        and arg.id not in guarded_as_str
                    ):
                        return f"{_dotted_name(node.func)}(..., {arg.id})"
    return ""


_PLACEHOLDER_FAILURE_PHRASES: tuple[str, ...] = (
    "html 본문 데이터 없음",
    "html이 잘려",
    "페이지가 잘렸",
    "기사 제목을 추출할 수 없습니다",
    "html unavailable",
    "page truncated",
    "docx 파일 읽기 실패",
    ".docx 파일 읽기 실패",
    "파일 검증 중 오류",
)


def _placeholder_success_literal(func: ast.AsyncFunctionDef) -> str:
    parents = _parent_map(func)
    for node in ast.walk(func):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        text = node.value.strip()
        lowered = text.lower()
        if not any(phrase in lowered for phrase in _PLACEHOLDER_FAILURE_PHRASES):
            continue
        if _has_ancestor(node, parents, ast.Raise):
            continue
        return text[:80]
    return ""


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _has_ancestor(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    ancestor_type: type[ast.AST],
) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ancestor_type):
            return True
        current = parents.get(current)
    return False


def _call_llm_assigned_names(func: ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.Await) and _is_ctx_call(value.value, "call_llm"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_ctx_call(node: ast.AST, method: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ctx"
    )


def _isinstance_guarded_names(
    func: ast.AsyncFunctionDef,
    type_name: str,
) -> set[str]:
    guarded: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "isinstance" or len(node.args) < 2:
            continue
        subject, type_expr = node.args[0], node.args[1]
        if not isinstance(subject, ast.Name):
            continue
        if _type_expr_contains(type_expr, type_name):
            guarded.add(subject.id)
    return guarded


def _type_expr_contains(node: ast.AST, type_name: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == type_name
    if isinstance(node, ast.Tuple):
        return any(_type_expr_contains(elt, type_name) for elt in node.elts)
    return False


def _nodes_call_json_loads(nodes: list[ast.stmt]) -> bool:
    for stmt in nodes:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            if dotted in {"json.loads", "_json.loads"}:
                return True
    return False


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return ""


def _body_assigns_empty_collection(nodes: list[ast.stmt]) -> bool:
    for stmt in nodes:
        if isinstance(stmt, ast.Assign) and _is_empty_collection(stmt.value):
            return True
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if _is_empty_collection(stmt.value):
                return True
        if isinstance(stmt, ast.Return) and _is_empty_collection(stmt.value):
            return True
    return False


def _is_empty_collection(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    return False


def _uses_project_file_map_antipattern(func: ast.AsyncFunctionDef) -> bool:
    text = "\n".join(_literal_strings_in(func)).lower()
    if (
        "json object" in text
        and "file" in text
        and ("relative file path" in text or "file content" in text)
    ):
        return True
    if "mapping of relative file path to file content" in text:
        return True

    for node in ast.walk(func):
        if not isinstance(node, ast.For):
            continue
        iter_text = _dotted_name(node.iter)
        if ".items" not in iter_text:
            continue
        if any(token in iter_text for token in ("project_files", "file_map", "files")):
            if _calls_tool(node, "files_write_text"):
                return True
    return False


def _calls_tool(node: ast.AST, tool_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if _call_name(child.func) != "call_tool" or not child.args:
            continue
        first = child.args[0]
        if isinstance(first, ast.Constant) and first.value == tool_name:
            return True
    return False


def _content_fallback_assignment(func: ast.AsyncFunctionDef) -> str:
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        tested_names = {
            name for name in _names_in(node.test)
            if _looks_like_file_content_name(name)
        }
        if not tested_names:
            continue
        for stmt in node.body:
            for target_name in _assigned_names(stmt):
                if target_name in tested_names:
                    return target_name
    return ""


def _names_in(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }


def _assigned_names(stmt: ast.stmt) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        targets.extend(stmt.targets)
    elif isinstance(stmt, ast.AnnAssign):
        targets.append(stmt.target)
    elif isinstance(stmt, ast.AugAssign):
        targets.append(stmt.target)

    names: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _looks_like_file_content_name(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _CONTENT_NAME_HINTS)


def _function_has_raise(func: ast.AsyncFunctionDef) -> bool:
    return any(isinstance(node, ast.Raise) for node in ast.walk(func))


def _function_has_content_guard(func: ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        if not any(isinstance(stmt, ast.Raise) for stmt in ast.walk(node)):
            continue
        if _test_checks_content(node.test):
            return True
    return False


def _test_checks_content(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        if isinstance(node.operand, ast.Name):
            return _looks_like_file_content_name(node.operand.id)

    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _call_name(child.func) == "len":
            if child.args and _expr_mentions_content_name(child.args[0]):
                return True
    return False


def _expr_mentions_content_name(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name)
        and _looks_like_file_content_name(child.id)
        for child in ast.walk(node)
    )


def _function_checks_ok(func: ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            call = node.func
            if (
                isinstance(call, ast.Attribute)
                and call.attr == "get"
                and node.args
                and _literal_str(node.args[0]) == "ok"
            ):
                return True
        elif isinstance(node, ast.Subscript):
            if _literal_str(node.slice) == "ok":
                return True
    return False


def _format_plan_for_synthesis(plan: Any) -> str:
    lines = [
        f"flow_name: {plan.flow_name}",
        f"flow_prompt: {plan.flow_prompt}",
        f"task_name: {plan.task_name}",
        f"task_prompt: {plan.task_prompt}",
        f"top_class: {plan.top_class}",
        "steps (in execution order):",
    ]
    for step in plan.steps:
        consumes = (
            ", ".join(str(o) for o in step.consumes_previous_orders)
            if step.consumes_previous_orders else "(none — reads ctx.input only)"
        )
        branch = ""
        if step.branch is not None:
            targets = ", ".join(
                f"{value!r}->{target}"
                for value, target in step.branch.targets.items()
            )
            fb = step.branch.fallback or "(no fallback)"
            branch = (
                f"\n    branch: switch on output field '{step.branch.field}' "
                f"with targets {{{targets}}}, fallback={fb}"
            )
        lines.append(
            f"  - name: {step.name}\n"
            f"    order: {step.order}\n"
            f"    needs_llm_reasoning: {step.needs_llm_reasoning}\n"
            f"    purpose: {step.purpose}\n"
            f"    consumes_previous_orders: {consumes}{branch}"
        )
    return "\n".join(lines)


def _format_selection_for_synthesis(selection: Any) -> str:
    lines = []
    for sel in selection.selections:
        tools = ", ".join(sel.tool_names) if sel.tool_names else "(none)"
        mcp = (
            f", mcp_server={sel.mcp_server_name}" if sel.mcp_server_name else ""
        )
        lines.append(
            f"  - {sel.step_name}: mode={sel.mode}{mcp}, tools=[{tools}]\n"
            f"    rationale: {sel.rationale}"
        )
    return "\n".join(lines)


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
        "convert the data into a JSON array of editable slide objects. Prefer "
        "structured layouts such as 'cover', 'metric', 'cards', 'timeline', "
        "'chart', and 'table' over plain title+bullets when they fit the "
        "content. For high-fidelity decks, pass engine='ppt-master' and either "
        "provide slide-level 'svg'/'svg_path' values or structured layouts; "
        "FlowForge will run the vendored PPT Master SVG-to-DrawingML pipeline "
        "to produce editable native PowerPoint shapes. Use table/chart/shape "
        "fields instead of rendering a slide screenshot.",
    ])
    return "\n".join(lines)


def _extract_required_tools(user_query: str | Any) -> list[str]:
    """Extract ordered unique ``required_tools`` names from a user query."""
    raw: Any = None
    if isinstance(user_query, dict):
        raw = user_query.get("required_tools")
    elif hasattr(user_query, "model_dump"):
        try:
            dumped = user_query.model_dump()
            raw = dumped.get("required_tools") if isinstance(dumped, dict) else None
        except Exception:
            raw = None

    if not isinstance(raw, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _collect_generated_tool_usage(
    code: str,
    *,
    known_tool_refs: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Return (decorator-scoped names, runtime-used names) from source code."""
    import re

    from flowforge.execution.llm import is_html_tag_name

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()

    scoped: set[str] = set()
    runtime_used: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if _call_name(decorator.func) not in {"flow", "task", "step"}:
                    continue
                for keyword in decorator.keywords:
                    if keyword.arg == "tools":
                        scoped.update(_literal_string_items(keyword.value))
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name == "call_tool" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    runtime_used.add(first.value)
            elif call_name == "call_llm":
                for text in _literal_strings_in(node):
                    for match in re.finditer(
                        r"<([A-Za-z0-9_][A-Za-z0-9_-]*)>",
                        text,
                    ):
                        name = match.group(1)
                        if known_tool_refs is not None and name not in known_tool_refs:
                            if is_html_tag_name(name):
                                continue
                        runtime_used.add(name)

    return scoped, runtime_used


def _tool_config_name(tool: Any) -> str:
    """Return the FlowForge reference name for a ToolConfig-like object."""
    from flowforge.types import AgentSkill, ClaudeSkill, FunctionTool, HTTPTool, MCPServer

    if isinstance(tool, MCPServer):
        return tool.name
    if isinstance(tool, FunctionTool):
        return tool.name or (
            tool.func.__name__ if hasattr(tool.func, "__name__") else ""
        )
    if isinstance(tool, HTTPTool):
        return tool.name
    if isinstance(tool, ClaudeSkill):
        return tool.name or tool.skill_id
    if isinstance(tool, AgentSkill):
        return tool.name
    return ""


def _word_tokens(text: str) -> set[str]:
    import re

    return {
        match.group(0)
        for match in re.finditer(r"[a-zA-Z0-9_-]+", text.lower())
    }


def _literal_string_items(node: ast.AST) -> set[str]:
    """Collect string constants from a literal list/tuple/set expression."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    values: set[str] = set()
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.add(item.value)
    return values


def _literal_strings_in(node: ast.AST) -> list[str]:
    """Collect literal string fragments under an AST node."""
    strings: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            strings.append(child.value)
    return strings


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


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
