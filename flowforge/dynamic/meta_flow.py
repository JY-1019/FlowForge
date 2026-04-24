"""Built-in meta-flow — FlowForge uses itself to generate new flows.

This module defines ``_dynamic_generator``, an internal ``@flow`` that is
automatically injected into agents with ``@global_config(dynamic_flow=True)``.

The flow itself is defined using FlowForge's own decorators, making this
a self-referential ("dogfooding") design:

.. code-block:: text

    @flow "_dynamic_generator"
    └─ @task "generate"
        ├─ step[1] analyse_gap       — check if existing DAG covers the query
        ├─ step[2] generate_code     — LLM writes FlowForge decorator code
        ├─ step[3] compile_and_inject — compile + add to live DAG
        └─ step[4] execute_new_flow  — run the newly created flow

The meta-flow is **never** selected by the autonomous planner for normal
queries — it is triggered explicitly by the ``ExecutionEngine`` when:

1. ``dynamic_flow=True`` on the agent, AND
2. The planner found no matching flows, AND
3. Gap analysis confirms the query is not covered.

The meta-flow communicates with the ``CompiledAgent`` via
``ctx.shared_data["_compiled_agent"]`` which the engine injects before
calling the flow.
"""
from __future__ import annotations

from flowforge.annotations.decorators import flow, task, step


# ─────────────────────────────────────────────────────────────────────────────
# Step handlers
# ─────────────────────────────────────────────────────────────────────────────

async def _analyse_gap(ctx):
    """Check if any existing flow covers the user query."""
    from flowforge.dynamic.generator import DynamicFlowGenerator

    agent = ctx.shared_data["_compiled_agent"]
    user_query = ctx.input

    generator = DynamicFlowGenerator(
        llm_config=ctx.global_ctx.llm_config,
        dag=agent.dag,
        docs=agent.docs,
    )

    gap = await generator.analyse_gap(user_query)
    return {
        "user_query": str(user_query),
        "gap_analysis": gap,
        "covered": gap.get("covered", True),
        "flow_name": gap.get("suggested_flow_name", ""),
        "flow_prompt": gap.get("suggested_flow_prompt", ""),
    }


async def _generate_code(ctx):
    """Generate FlowForge code for the new flow via LLM."""
    from flowforge.dynamic.generator import DynamicFlowGenerator

    data = ctx.input
    if data.get("covered", True):
        # Query is already covered — skip generation.
        return {
            **data,
            "code": None,
            "skipped": True,
        }

    agent = ctx.shared_data["_compiled_agent"]
    generator = DynamicFlowGenerator(
        llm_config=ctx.global_ctx.llm_config,
        dag=agent.dag,
        docs=agent.docs,
    )

    code = await generator.generate_flow_code(
        flow_name=data["flow_name"],
        flow_prompt=data["flow_prompt"],
        user_query=data["user_query"],
    )

    return {
        **data,
        "code": code,
        "skipped": False,
    }


async def _compile_and_inject(ctx):
    """Compile the generated code and inject the new flow into the live DAG."""
    from flowforge.dynamic.generator import DynamicFlowGenerator, _MAX_RETRIES
    from flowforge.errors import CompileError
    import logging

    logger = logging.getLogger(__name__)
    data = ctx.input

    if data.get("skipped"):
        return {**data, "injected": False}

    agent = ctx.shared_data["_compiled_agent"]
    generator = DynamicFlowGenerator(
        llm_config=ctx.global_ctx.llm_config,
        dag=agent.dag,
        docs=agent.docs,
    )

    code = data["code"]
    last_error = None

    for attempt in range(_MAX_RETRIES):
        try:
            flow_meta = generator.compile_flow_code(code)

            # Inject into the live agent.
            node_id = agent.add_flow(flow_meta.cls)

            logger.info(
                "dynamic flow '%s' injected as %s (attempt %d)",
                data["flow_name"], node_id, attempt + 1,
            )

            return {
                **data,
                "injected": True,
                "node_id": node_id,
                "flow_meta_name": flow_meta.name,
            }
        except (CompileError, Exception) as e:
            last_error = e
            logger.warning(
                "compile attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, e,
            )
            if attempt + 1 < _MAX_RETRIES:
                code = await generator._fix_code(
                    code, str(e), data["user_query"],
                )

    return {
        **data,
        "injected": False,
        "error": str(last_error),
    }


async def _execute_new_flow(ctx):
    """Execute the newly injected flow with the original user query."""
    data = ctx.input

    if not data.get("injected"):
        return {
            "success": False,
            "reason": data.get("error", "Flow was not injected"),
            "dynamic_flow": data.get("flow_name"),
            "gap_analysis": data.get("gap_analysis"),
        }

    agent = ctx.shared_data["_compiled_agent"]
    flow_name = data["flow_meta_name"]
    user_query = data["user_query"]

    # Run only the newly created flow via route.
    result = await agent.run(user_query, route=flow_name)

    return {
        "success": True,
        "dynamic_flow": flow_name,
        "result": result,
        "gap_analysis": data.get("gap_analysis"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Meta-flow definition — FlowForge using its own decorators
# ─────────────────────────────────────────────────────────────────────────────

@flow(
    name="_dynamic_generator",
    prompt=(
        "Internal meta-flow: analyses whether the user query is covered by "
        "existing flows, generates new FlowForge code if needed, compiles "
        "and injects the new flow, then executes it."
    ),
)
class DynamicGeneratorFlow:
    """Built-in flow for dynamic flow generation.

    This flow is automatically added to agents with
    ``@global_config(dynamic_flow=True)`` and is triggered by the
    execution engine when no existing flow matches a user query.
    """

    @task(
        name="_generate_and_run",
        prompt="Generate a new flow from user query and execute it",
    )
    class GenerateAndRunTask:

        @step(
            order=1,
            prompt=(
                "Analyse the user query against the existing DAG to determine "
                "if a new flow needs to be generated."
            ),
        )
        async def analyse_gap(ctx):
            return await _analyse_gap(ctx)

        @step(
            order=2,
            prompt=(
                "Generate FlowForge decorator code for the new flow using "
                "LLM. The code will define @flow, @task, and @step decorators."
            ),
        )
        async def generate_code(ctx):
            return await _generate_code(ctx)

        @step(
            order=3,
            prompt=(
                "Compile the generated Python code into a FlowMeta, validate "
                "it, and inject the new flow into the live DAG. Retry with "
                "error feedback if compilation fails."
            ),
        )
        async def compile_and_inject(ctx):
            return await _compile_and_inject(ctx)

        @step(
            order=4,
            prompt=(
                "Execute the newly injected flow with the original user query "
                "using route-based execution."
            ),
        )
        async def execute_new_flow(ctx):
            return await _execute_new_flow(ctx)
