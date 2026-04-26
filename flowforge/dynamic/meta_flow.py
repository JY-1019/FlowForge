"""Built-in meta-flow — FlowForge uses itself to generate new flows.

This module defines ``_dynamic_generator``, an internal ``@flow`` that is
automatically injected into agents with ``@global_config(dynamic_flow=True)``.

The flow itself is defined using FlowForge's own decorators, making this
a self-referential ("dogfooding") design:

.. code-block:: text

    @flow "_dynamic_generator"
    └─ @task "generate"
        ├─ step[1] analyse_gap        — check if existing DAG covers the query
        ├─ step[2] prepare_codegen    — build a lightweight implementation brief
        ├─ step[3] generate_and_inject — generate code, compile, persist, inject

The meta-flow is **never** selected by the autonomous planner for normal
queries — it is triggered explicitly by the ``ExecutionEngine`` when:

1. ``dynamic_flow=True`` on the agent, AND
2. The planner found no matching flows, AND
3. Gap analysis confirms the query is not covered (or a planner has already
   provided a precomputed gap payload).

The meta-flow communicates with the ``CompiledAgent`` via
``ctx.shared_data["_compiled_agent"]`` which the engine injects before
calling the flow.

All code generation, compile-retry, contract validation, and persistence
are delegated to ``DynamicFlowGenerator`` public methods — the meta-flow
steps are thin orchestration wrappers that pass context between them.
"""
from __future__ import annotations

from flowforge.annotations.decorators import flow, task, step

_DYNAMIC_META_STEP_TIMEOUT_SECONDS = 300


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_generator(ctx):
    """Build a ``DynamicFlowGenerator`` from the current execution context."""
    from flowforge.dynamic.generator import DynamicFlowGenerator

    agent = ctx.shared_data["_compiled_agent"]
    return DynamicFlowGenerator(
        llm_config=ctx.global_ctx.llm_config,
        dag=agent.dag,
        docs=agent.docs,
        tool_configs=agent._global_meta.tools,
        dynamic_options=getattr(ctx.global_ctx, "dynamic_options", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step handlers
# ─────────────────────────────────────────────────────────────────────────────

async def _analyse_gap(ctx):
    """Check if any existing flow covers the user query."""
    user_query = ctx.input

    if isinstance(user_query, dict) and user_query.get("gap_analysis"):
        gap = user_query["gap_analysis"]
        raw_query = user_query.get("user_query", "")
        return {
            "user_query": str(raw_query),
            "gap_analysis": gap,
            "covered": gap.get("covered", False),
            "flow_name": gap.get("suggested_flow_name", ""),
            "flow_prompt": gap.get("suggested_flow_prompt", ""),
            "downstream_flow_route": user_query.get("downstream_flow_route", ""),
            "requirement": user_query.get("requirement"),
            "requirement_index": user_query.get("requirement_index"),
            "precomputed_gap": True,
        }

    generator = _build_generator(ctx)
    gap = await generator.analyse_gap(user_query)
    return {
        "user_query": str(user_query),
        "gap_analysis": gap,
        "covered": gap.get("covered", True),
        "flow_name": gap.get("suggested_flow_name", ""),
        "flow_prompt": gap.get("suggested_flow_prompt", ""),
        "downstream_flow_route": "",
        "precomputed_gap": False,
    }


async def _prepare_codegen(ctx):
    """Prepare a fast coding brief without scanning the project."""
    data = ctx.input
    if data.get("covered", True):
        return {
            **data,
            "codegen_context": "",
            "downstream_contract": None,
            "downstream_flow_name": None,
            "skipped": True,
        }

    generator = _build_generator(ctx)

    generation_query = (
        data["flow_prompt"] if data.get("precomputed_gap") else data["user_query"]
    )
    downstream_route = data.get("downstream_flow_route") or ""
    downstream_contract, downstream_name = generator.resolve_downstream_contract(
        downstream_route,
    )
    codegen_context = generator.build_code_generation_context(
        flow_name=data["flow_name"],
        flow_prompt=data["flow_prompt"],
        user_query=generation_query,
        downstream_contract=downstream_contract,
        downstream_flow_name=downstream_name,
    )

    return {
        **data,
        "codegen_context": codegen_context,
        "downstream_contract": downstream_contract,
        "downstream_flow_name": downstream_name,
        "skipped": False,
    }


async def _generate_and_inject(ctx):
    """Generate code, compile, persist, and inject the new flow.

    Delegates the entire generate → compile → retry → persist pipeline to
    ``DynamicFlowGenerator.generate_compile_and_persist()`` so the retry
    logic lives in a single place.
    """
    import logging

    logger = logging.getLogger(__name__)
    data = ctx.input

    if data.get("skipped"):
        return {
            **data,
            "success": False,
            "injected": False,
            "reason": "Gap analysis reported the query is already covered.",
        }

    agent = ctx.shared_data["_compiled_agent"]
    generator = _build_generator(ctx)

    generation_query = (
        data["flow_prompt"] if data.get("precomputed_gap") else data["user_query"]
    )

    try:
        flow_meta, code = await generator.generate_compile_and_persist(
            flow_name=data["flow_name"],
            flow_prompt=data["flow_prompt"],
            user_query=generation_query,
            downstream_contract=data.get("downstream_contract"),
            downstream_flow_name=data.get("downstream_flow_name"),
            downstream_flow_route=data.get("downstream_flow_route", ""),
            project_context=data.get("codegen_context"),
        )

        # Inject into the live agent.
        node_id = agent.add_flow(flow_meta.cls)

        logger.info(
            "dynamic flow '%s' injected as %s",
            data["flow_name"], node_id,
        )

        try:
            await agent.generate_docs(planning_only=True)
        except Exception as doc_error:
            logger.warning(
                "dynamic flow doc generation failed for '%s': %s",
                data["flow_name"], doc_error,
            )

        return {
            **data,
            "success": True,
            "injected": True,
            "node_id": node_id,
            "flow_meta_name": flow_meta.name,
            "dynamic_flow": flow_meta.name,
            "generated_code": code,
        }
    except Exception as e:
        logger.warning("dynamic flow generation failed: %s", e)
        return {
            **data,
            "success": False,
            "injected": False,
            "error": str(e),
            "reason": str(e),
            "dynamic_flow": data.get("flow_name", ""),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Meta-flow definition — FlowForge using its own decorators
# ─────────────────────────────────────────────────────────────────────────────

@flow(
    name="_dynamic_generator",
    prompt=(
        "Internal meta-flow: analyses whether the user query is covered by "
        "existing flows, generates new FlowForge code if needed, compiles "
        "and injects the new flow."
    ),
    max_retries=1,
)
class DynamicGeneratorFlow:
    """Built-in flow for dynamic flow generation.

    This flow is automatically added to agents with
    ``@global_config(dynamic_flow=True)`` and is triggered by the
    execution engine when no existing flow matches a user query.
    """

    @task(
        name="_generate_and_run",
        prompt="Generate a new flow from user query and inject it into the live DAG",
    )
    class GenerateAndRunTask:

        @step(
            order=1,
            prompt=(
                "Analyse the user query against the existing DAG to determine "
                "if a new flow needs to be generated."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def analyse_gap(ctx):
            return await _analyse_gap(ctx)

        @step(
            order=2,
            prompt=(
                "Prepare a lightweight implementation brief for codegen. "
                "Do not scan project files; only summarize the missing flow, "
                "available tools, downstream contract, and dynamic policy."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def prepare_codegen(ctx):
            return await _prepare_codegen(ctx)

        @step(
            order=3,
            prompt=(
                "Generate FlowForge code, compile with retry loop, persist "
                "to manifest, and inject the new flow into the live DAG."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def generate_and_inject(ctx):
            return await _generate_and_inject(ctx)
