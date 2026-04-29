"""Built-in meta-flow — FlowForge uses itself to generate new flows.

This module defines ``_dynamic_generator``, an internal ``@flow`` that is
automatically injected into agents with ``@global_config(dynamic_flow=True)``.

The flow uses FlowForge's own decorators (dogfooding):

.. code-block:: text

    @flow "_dynamic_generator"
    └─ @task "generate"
        ├─ step[1] analyse_gap        — does the existing DAG cover the query?
        ├─ step[2] plan_workflow      — design the best step decomposition
        ├─ step[3] select_capability  — pick mode + tools for each step
        ├─ step[4] mcp_provision      — record auto-provisioned MCP specs
        ├─ step[5] synthesise_inject  — translate the plan into FlowForge code

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

_DYNAMIC_META_STEP_TIMEOUT_SECONDS = 600


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


def _flow_summaries(generator) -> list[str]:
    """Return short ``"name: prompt (doc summary)"`` lines for every flow."""
    from flowforge.schema.dag import NodeType

    summaries: list[str] = []
    for node in generator._dag.get_all_nodes():
        if node.type != NodeType.FLOW:
            continue
        doc = generator._docs.get(node.id)
        summary = getattr(doc, "summary", "") if doc else ""
        prompt_text = getattr(node.meta, "prompt", "")
        entry = (
            f"- {node.id}: {prompt_text}"
            + (f" (doc: {summary})" if summary else "")
        )
        summaries.append(entry)
    return summaries


def _install_runtime_repair(
    *,
    flow_meta,
    agent,
    generator,
    global_ctx,
    initial_code: str,
    user_query,
    downstream_contract,
    downstream_flow_name,
    project_context,
    downstream_flow_route="",
) -> None:
    """Attach a ``runtime_repair`` hook to a freshly injected dynamic flow.

    The hook is called by ``FlowRunner`` on ``ExecutionError`` (before each
    retry).  It feeds the runtime error back to the LLM via
    ``generator.fix_code(...)``, recompiles the corrected source, replaces
    the broken subtree in the DAG, and refreshes
    ``global_ctx.planned_node_ids`` so the planner-restricted runner can
    enter the new sub-tasks.  Returns ``True`` when repair succeeded so
    ``FlowRunner`` skips the back-off sleep and runs the fixed flow
    immediately.
    """
    import logging

    logger = logging.getLogger(__name__)
    state = {"code": initial_code}
    flow_name = flow_meta.name
    setattr(flow_meta, "__flowforge_dynamic_source_code__", initial_code)
    setattr(
        flow_meta,
        "__flowforge_dynamic_downstream_flow_route__",
        downstream_flow_route,
    )

    async def _repair(error_msg: str) -> bool:
        try:
            new_code = await generator.fix_code(
                state["code"], error_msg, user_query,
                downstream_contract=downstream_contract,
                downstream_flow_name=downstream_flow_name,
                project_context=project_context,
            )
            new_meta = generator.compile_flow_code(new_code)
            compatibility_error = generator.check_contract_compatibility(
                new_meta,
                downstream_contract,
            )
            if compatibility_error is not None:
                raise ValueError(compatibility_error)
            tool_ref_error = generator.check_tool_ref_validity(new_code)
            if tool_ref_error is not None:
                raise ValueError(tool_ref_error)
            tool_usage_error = generator.check_required_tool_usage(
                new_code,
                user_query,
            )
            if tool_usage_error is not None:
                raise ValueError(tool_usage_error)
            quality_error = generator.check_generated_code_quality(
                new_code,
                user_query,
            )
            if quality_error is not None:
                raise ValueError(quality_error)
        except Exception as exc:
            logger.warning(
                "runtime repair failed to regenerate '%s': %s",
                flow_name, exc,
            )
            return False

        record = None
        options = getattr(global_ctx, "dynamic_options", None)
        if (
            options is not None
            and getattr(options, "persist_generated", False)
        ):
            try:
                from flowforge.dynamic.manifest import persist_flow_code

                record = persist_flow_code(
                    flow_name=new_meta.name,
                    code=new_code,
                    options=options,
                    class_name=new_meta.cls.__name__,
                    inject_before=downstream_flow_route,
                    downstream_flow_route=downstream_flow_route,
                )
            except Exception as exc:
                logger.warning(
                    "runtime repair regenerated '%s' but failed to persist "
                    "the fixed code: %s",
                    flow_name, exc,
                )

        # Copy mutable fields onto the live FlowMeta so FlowRunner's next
        # ``_run_once`` walks the regenerated task tree.  Preserves identity
        # (name, runtime_repair hook) so subsequent failures can repair too.
        for attr in (
            "cls", "prompt", "input_schema", "output_schema",
            "depends_on", "parallel", "child_flows", "tasks", "tools",
            "condition", "branches", "fallback",
        ):
            setattr(flow_meta, attr, getattr(new_meta, attr))
        setattr(flow_meta, "__flowforge_dynamic_source_code__", new_code)
        setattr(
            flow_meta,
            "__flowforge_dynamic_downstream_flow_route__",
            downstream_flow_route,
        )
        if record is not None:
            setattr(flow_meta, "__flowforge_dynamic_manifest_record__", record)
            setattr(new_meta, "__flowforge_dynamic_manifest_record__", record)
        setattr(new_meta, "__flowforge_dynamic_source_code__", new_code)
        setattr(
            new_meta,
            "__flowforge_dynamic_downstream_flow_route__",
            downstream_flow_route,
        )

        # Rebuild the DAG subtree under the same node_id so descendant
        # lookups reflect the new structure.
        try:
            agent.replace_flow(flow_meta.cls)
        except Exception as exc:
            logger.warning(
                "runtime repair failed to replace flow '%s' in DAG: %s",
                flow_name, exc,
            )
            return False

        # Refresh the planned-route restriction so the runner doesn't filter
        # out the new descendants.  Drop every old ID under this flow's
        # subtree and add every new descendant.
        planned = getattr(global_ctx, "planned_node_ids", None)
        if planned is not None:
            node_id = f"global.{flow_name}"
            stale = [pid for pid in planned if pid == node_id or pid.startswith(node_id + ".")]
            for pid in stale:
                planned.discard(pid) if isinstance(planned, set) else planned.remove(pid)
            new_node = agent.dag.get_node(node_id)
            if new_node is not None:
                planned.add(node_id) if isinstance(planned, set) else planned.append(node_id)
                for descendant in agent.dag.get_descendants(node_id):
                    if isinstance(planned, set):
                        planned.add(descendant.id)
                    elif descendant.id not in planned:
                        planned.append(descendant.id)

        state["code"] = new_code
        logger.info(
            "runtime repair regenerated dynamic flow '%s' (%d chars)",
            flow_name, len(new_code),
        )
        return True

    flow_meta.runtime_repair = _repair


# ─────────────────────────────────────────────────────────────────────────────
# Step handlers
# ─────────────────────────────────────────────────────────────────────────────

async def _analyse_gap(ctx):
    """Step 1 — check if any existing flow covers the user query."""
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


async def _plan_workflow(ctx):
    """Step 2 — design the workflow as a list of single-responsibility steps."""
    data = ctx.input
    if data.get("covered", True):
        return {**data, "skipped": True}

    from flowforge.dynamic.plan import plan_workflow

    generator = _build_generator(ctx)
    generation_query = (
        data["flow_prompt"] if data.get("precomputed_gap") else data["user_query"]
    )
    plan = await plan_workflow(
        user_query=generation_query,
        suggested_flow_name=data.get("flow_name", "dynamic_flow"),
        suggested_flow_prompt=data.get("flow_prompt", str(generation_query)),
        flow_summaries=_flow_summaries(generator),
        llm_config=ctx.global_ctx.llm_config,
    )
    # The planner may refine the suggested flow name — keep its choice.
    return {
        **data,
        "flow_name": plan.flow_name,
        "flow_prompt": plan.flow_prompt,
        "plan": plan.model_dump(),
        "skipped": False,
    }


async def _select_capability(ctx):
    """Step 3 — pick implementation mode + tool names per step."""
    data = ctx.input
    if data.get("skipped"):
        return data

    from flowforge.dynamic.capability import select_capabilities
    from flowforge.dynamic.plan import WorkflowPlan

    plan = WorkflowPlan.model_validate(data["plan"])
    agent = ctx.shared_data["_compiled_agent"]
    selection = await select_capabilities(
        plan=plan,
        tool_configs=agent._global_meta.tools,
        dynamic_options=getattr(ctx.global_ctx, "dynamic_options", None),
        llm_config=ctx.global_ctx.llm_config,
    )
    return {
        **data,
        "capability_selection": selection.model_dump(),
    }


async def _provision_mcp(ctx):
    """Step 4 — persist ``_artifact/mcp/<server>.json`` for any MCP-mode steps."""
    data = ctx.input
    if data.get("skipped"):
        return data

    from flowforge.dynamic.capability import CapabilitySelection
    from flowforge.dynamic.mcp_provision import provision_mcp_servers
    from flowforge.dynamic.plan import WorkflowPlan

    options = getattr(ctx.global_ctx, "dynamic_options", None)
    if options is None:
        return {**data, "mcp_provisioned": []}

    plan = WorkflowPlan.model_validate(data["plan"])
    selection = CapabilitySelection.model_validate(data["capability_selection"])
    records = provision_mcp_servers(
        selection=selection, plan=plan, options=options,
    )
    return {
        **data,
        "mcp_provisioned": [record.model_dump() for record in records],
    }


async def _synthesise_and_inject(ctx):
    """Step 5 — synthesise FlowForge code from plan + capability and inject."""
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

    from flowforge.dynamic.capability import CapabilitySelection
    from flowforge.dynamic.plan import WorkflowPlan

    agent = ctx.shared_data["_compiled_agent"]
    generator = _build_generator(ctx)
    plan = WorkflowPlan.model_validate(data["plan"])
    selection = CapabilitySelection.model_validate(data["capability_selection"])

    generation_query = (
        data["flow_prompt"] if data.get("precomputed_gap") else data["user_query"]
    )
    downstream_route = data.get("downstream_flow_route") or ""
    downstream_contract, downstream_name = generator.resolve_downstream_contract(
        downstream_route,
    )
    project_context = generator.build_code_generation_context(
        flow_name=plan.flow_name,
        flow_prompt=plan.flow_prompt,
        user_query=generation_query,
        downstream_contract=downstream_contract,
        downstream_flow_name=downstream_name,
    )

    try:
        flow_meta, code = await generator.generate_compile_and_persist_from_plan(
            plan=plan,
            selection=selection,
            user_query=generation_query,
            downstream_contract=downstream_contract,
            downstream_flow_name=downstream_name,
            downstream_flow_route=downstream_route,
            project_context=project_context,
        )

        node_id = agent.add_flow(flow_meta.cls)

        _install_runtime_repair(
            flow_meta=flow_meta,
            agent=agent,
            generator=generator,
            global_ctx=ctx.global_ctx,
            initial_code=code,
            user_query=generation_query,
            downstream_contract=downstream_contract,
            downstream_flow_name=downstream_name,
            downstream_flow_route=downstream_route,
            project_context=project_context,
        )
        logger.info(
            "dynamic flow '%s' injected as %s",
            plan.flow_name, node_id,
        )

        await agent.generate_docs(planning_only=True)

        return {
            **data,
            "success": True,
            "injected": True,
            "node_id": node_id,
            "flow_meta_name": flow_meta.name,
            "dynamic_flow": flow_meta.name,
            "generated_code": code,
        }
    except Exception as exc:
        logger.warning("dynamic flow generation failed: %s", exc)
        return {
            **data,
            "success": False,
            "injected": False,
            "error": str(exc),
            "reason": str(exc),
            "dynamic_flow": plan.flow_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Meta-flow definition — FlowForge using its own decorators
# ─────────────────────────────────────────────────────────────────────────────

@flow(
    name="_dynamic_generator",
    prompt=(
        "Internal meta-flow: analyses whether the user query is covered by "
        "existing flows, plans the best workflow, selects capabilities per "
        "step, provisions MCP servers when needed, then synthesises and "
        "injects FlowForge code for the missing flow."
    ),
    max_retries=1,
)
class DynamicGeneratorFlow:
    """Built-in flow for dynamic flow generation."""

    @task(
        name="_generate_and_run",
        prompt=(
            "Plan the workflow, decide capabilities, provision MCP servers, "
            "and synthesise the new flow before injecting it into the DAG."
        ),
    )
    class GenerateAndRunTask:

        @step(
            order=1,
            prompt=(
                "Analyse the user query against the existing DAG to determine "
                "whether a new flow needs to be generated."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def analyse_gap(ctx):
            return await _analyse_gap(ctx)

        @step(
            order=2,
            prompt=(
                "Design the BEST sequential workflow for the missing flow as "
                "a list of single-responsibility steps with explicit branches."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def plan_workflow(ctx):
            return await _plan_workflow(ctx)

        @step(
            order=3,
            prompt=(
                "For every planned step, pick the optimal implementation mode "
                "(llm_only / builtin_tool / claude_skill / agent_skill / mcp) "
                "and the exact tool names from the registered catalog."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def select_capability(ctx):
            return await _select_capability(ctx)

        @step(
            order=4,
            prompt=(
                "Persist a JSON record under _artifact/mcp/<server>.json for "
                "every MCP server the plan selected, so the spec is visible "
                "in the dynamic manifest."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def mcp_provision(ctx):
            return await _provision_mcp(ctx)

        @step(
            order=5,
            prompt=(
                "Synthesise FlowForge code from the plan + capability "
                "decision, compile with retry, persist to manifest, and "
                "inject the new flow into the live DAG."
            ),
            timeout_seconds=_DYNAMIC_META_STEP_TIMEOUT_SECONDS,
        )
        async def synthesise_and_inject(ctx):
            return await _synthesise_and_inject(ctx)
