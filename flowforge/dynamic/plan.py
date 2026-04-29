"""Phase 1 — Workflow planning.

The dynamic generator's first LLM call no longer produces Python code.  It
produces a structured ``WorkflowPlan`` that describes:

* which @flow / @task / @step skeleton best fits the user request,
* what each step is responsible for (single-responsibility per step),
* which steps need LLM reasoning vs. pure deterministic logic,
* whether a step should dispatch to one of several next steps via a
  Branch condition.

The plan is returned as a Pydantic model so later phases (capability
selection, MCP provisioning, code synthesis) can consume a normalised,
type-checked structure instead of free-form text.

Design notes
------------
* The plan does **not** name tools — that decision is delegated to
  :mod:`flowforge.dynamic.capability` where the registered catalog is
  enforced as a whitelist.  Keeping tool selection out of Phase 1 keeps
  the planning prompt small and lets Phase 2 specialise on capability
  reasoning.
* ``max_tokens`` is set generously so the planner can emit a complete plan
  for moderately complex agents without truncation.
"""
from __future__ import annotations

import logging
import textwrap
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from flowforge.types import LLMConfig

logger = logging.getLogger(__name__)


# Generous response budget — agentic runs are slow anyway, and a partial
# plan is worse than a slow one.
_PLAN_MAX_TOKENS = 8000


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PlannedBranch(BaseModel):
    """Branch dispatch declared on a step's ``order``.

    When the planner says step *N* should pick the next step based on a
    field in its output, the synthesis phase emits ``@step(condition=...,
    branches={...}, fallback=...)`` against named target steps that already
    exist in the plan.
    """

    field: str = Field(
        ...,
        description=(
            "Name of the dict field in the deciding step's output that "
            "selects the branch target."
        ),
    )
    targets: dict[str, str] = Field(
        ...,
        description=(
            "Map from output field value to the name of the next step to "
            "dispatch to.  Target names must match a `PlannedStep.name` "
            "later in the plan."
        ),
    )
    fallback: str = Field(
        "",
        description=(
            "Default target step name used when no branch matches.  Empty "
            "string disables the fallback."
        ),
    )


class PlannedStep(BaseModel):
    """One workflow phase.

    The planner is encouraged to emit one step per *single responsibility*
    rather than one mega-step per flow.  Capability selection happens in
    Phase 2 — Phase 1 only records intent.
    """

    name: str = Field(
        ...,
        description="snake_case function name unique inside the plan.",
    )
    order: int = Field(
        ...,
        ge=1,
        description="1-based execution order. Parallel siblings share an order.",
    )
    purpose: str = Field(
        ...,
        min_length=10,
        description=(
            "1-2 sentence description of WHAT this step accomplishes.  Used "
            "verbatim in the @step prompt and as input to capability "
            "selection."
        ),
    )
    needs_llm_reasoning: bool = Field(
        ...,
        description=(
            "True when the step requires generative LLM work (analysis, "
            "summarisation, content authoring, structured extraction).  "
            "False when the step is a deterministic call into a tool or "
            "pure Python shaping of upstream results."
        ),
    )
    consumes_previous_orders: list[int] = Field(
        default_factory=list,
        description=(
            "Orders of earlier steps whose outputs this step consumes via "
            "`ctx.previous_results.get(<order>)`.  Empty for the first "
            "step or for steps that read only `ctx.input`."
        ),
    )
    branch: PlannedBranch | None = Field(
        None,
        description=(
            "When set, this step routes execution to one of several next "
            "steps based on a field of its own output.  Only set on steps "
            "whose output drives a fork; non-routing steps leave it null."
        ),
    )

    @field_validator("name")
    @classmethod
    def _name_is_snake_case(cls, value: str) -> str:
        if not value or not value.replace("_", "").isalnum():
            raise ValueError(
                f"step name must be snake_case alphanumerics, got {value!r}"
            )
        return value


class WorkflowPlan(BaseModel):
    """The complete plan returned by Phase 1."""

    flow_name: str = Field(
        ...,
        description="snake_case top-level @flow name.",
    )
    flow_prompt: str = Field(
        ...,
        min_length=10,
        description="System-level instruction for the top-level @flow.",
    )
    task_name: str = Field(
        ...,
        description="snake_case @task name that wraps the planned steps.",
    )
    task_prompt: str = Field(
        ...,
        min_length=10,
        description="System-level instruction for the leaf @task.",
    )
    top_class: str = Field(
        ...,
        description="PascalCase Python class name for the top-level @flow.",
    )
    steps: list[PlannedStep] = Field(
        ...,
        min_length=1,
        description="Ordered phases that make up the workflow.",
    )

    @field_validator("steps")
    @classmethod
    def _orders_are_dense_from_one(
        cls, value: list[PlannedStep]
    ) -> list[PlannedStep]:
        if not value:
            raise ValueError("workflow plan must contain at least one step")
        names = [step.name for step in value]
        if len(set(names)) != len(names):
            raise ValueError(
                f"duplicate step names in plan: {names}"
            )
        orders = sorted({step.order for step in value})
        if orders[0] != 1:
            raise ValueError(
                f"step orders must start at 1, got {orders}"
            )
        for prev, curr in zip(orders, orders[1:]):
            if curr - prev > 1:
                raise ValueError(
                    f"step orders must be dense (no gaps), got {orders}"
                )
        return value

    def step_by_name(self, name: str) -> PlannedStep | None:
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def steps_by_order(self) -> dict[int, list[PlannedStep]]:
        groups: dict[int, list[PlannedStep]] = {}
        for step in self.steps:
            groups.setdefault(step.order, []).append(step)
        return groups


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


_PLAN_SYSTEM = textwrap.dedent("""\
    You are FlowForge's workflow planner.  Given a user request, you design
    the BEST sequential workflow as a list of single-responsibility steps.

    PRINCIPLES
    1. Decompose the request into the smallest set of distinct phases that
       still fully solves it — typically 2–6 steps.  Avoid 1 mega-step that
       does everything.  File/project generation workflows may need more
       steps because each substantial file must be authored and verified
       separately.
    2. Each step has ONE focused responsibility (e.g. "fetch source data",
       "design the layout structure", "write the index.html file",
       "verify the output exists and is non-trivial").
    3. Use ``order`` to express sequencing.  Use the SAME order for siblings
       that can run in parallel (rare — only for genuinely independent work).
    4. Set ``needs_llm_reasoning`` honestly:
         * True  → step authors content, analyses, summarises, plans, or
                   makes a judgement call.
         * False → step is a deterministic tool/library call, a file write
                   driven by upstream content, or pure data shaping.
       (Capability selection in the next phase uses this flag.)
    5. List ``consumes_previous_orders`` for any step that reads earlier
       output via ``ctx.previous_results``.  Step 1 normally has [].
    6. Use ``branch`` only when the next step depends on a value computed
       by this step.  ``branch.targets`` maps that field's value to the
       NAME of another step in this plan that becomes the dispatch target.
       Most plans have NO branch.
    7. Always include a final verification or summarisation step when the
       workflow produces files, external state, or generated content — the
       verifier reads back a key artefact and asserts it is non-trivial.
    8. ``flow_name`` and ``task_name`` are snake_case.  ``top_class`` is
       PascalCase ending in ``Flow`` (e.g. ``CloneSiteFlow``).
    9. Do NOT name specific tools, MCP servers, or skills — that is decided
       in the next phase.  Describe phases in terms of *what* must happen,
       not *how*.
    10. The plan must be self-contained: a downstream code generator must
        be able to translate every step into FlowForge code without asking
        for clarification.

    FILE / FRONTEND PROJECT WORKFLOWS
    - NEVER plan a single step that asks the LLM to return every project
      file as one JSON object or "file map".  That pattern truncates easily
      and produces empty placeholder files.
    - For frontend or clone-coding requests, plan separate authoring and
      write phases for the important files: package.json, index.html,
      CSS, JavaScript/TypeScript, and any config files.  A write phase must
      use the file-writing tool and a later verification phase must read
      back at least index.html (and a built output when applicable).
    - Include install and build phases when the request asks for a runnable
      npm/Vite/React project.
    - The final verification phase must fail loudly when the entrypoint is
      empty, trivial, or placeholder-only; do not let "0 files written" count
      as success.

    Respond by calling the ``submit_workflow_plan`` tool.  Do NOT respond
    with prose.
""")


# Anthropic-native tool schema: this is what `call_with_tool()` enforces.
_PLAN_TOOL_SCHEMA: dict[str, Any] = {
    "name": "submit_workflow_plan",
    "description": (
        "Submit the planned FlowForge workflow as a structured object."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "flow_name": {
                "type": "string",
                "description": "snake_case top-level @flow name.",
            },
            "flow_prompt": {
                "type": "string",
                "description": "Instruction for the top-level @flow.",
            },
            "task_name": {
                "type": "string",
                "description": "snake_case leaf @task name.",
            },
            "task_prompt": {
                "type": "string",
                "description": "Instruction for the leaf @task.",
            },
            "top_class": {
                "type": "string",
                "description": "PascalCase Python class name (ends in 'Flow').",
            },
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "order": {"type": "integer", "minimum": 1},
                        "purpose": {"type": "string"},
                        "needs_llm_reasoning": {"type": "boolean"},
                        "consumes_previous_orders": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                        "branch": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "targets": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                },
                                "fallback": {"type": "string"},
                            },
                            "required": ["field", "targets"],
                        },
                    },
                    "required": [
                        "name", "order", "purpose", "needs_llm_reasoning",
                    ],
                },
            },
        },
        "required": [
            "flow_name", "flow_prompt", "task_name", "task_prompt",
            "top_class", "steps",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _format_dag_summary(
    flow_summaries: list[str], max_chars: int = 2400
) -> str:
    if not flow_summaries:
        return "(no existing flows)"
    text = "\n".join(flow_summaries)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[…flow list truncated]"
    return text


async def plan_workflow(
    *,
    user_query: str | Any,
    suggested_flow_name: str,
    suggested_flow_prompt: str,
    flow_summaries: list[str],
    llm_config: LLMConfig,
) -> WorkflowPlan:
    """Run Phase 1 — produce a ``WorkflowPlan`` for the missing flow.

    Parameters
    ----------
    user_query:
        Original user request (or planner-supplied flow brief).
    suggested_flow_name, suggested_flow_prompt:
        Names produced by gap analysis; the planner may refine them.
    flow_summaries:
        One-line descriptions of existing flows so the planner can avoid
        re-implementing capabilities that already exist.
    llm_config:
        Provider/model/credentials.
    """
    from flowforge.llm.caller import call_with_tool

    user_prompt = (
        f"User request:\n{user_query}\n\n"
        f"Gap-analysis suggested flow: {suggested_flow_name}\n"
        f"Gap-analysis suggested purpose: {suggested_flow_prompt}\n\n"
        f"Existing flows (do not re-implement these):\n"
        f"{_format_dag_summary(flow_summaries)}\n\n"
        f"Plan the BEST workflow that solves this request as a sequence of "
        f"single-responsibility steps.  Submit the plan via the tool."
    )

    raw = await call_with_tool(
        prompt=user_prompt,
        tool_schema=_PLAN_TOOL_SCHEMA,
        llm_config=llm_config,
        system_prompt=_PLAN_SYSTEM,
        max_tokens=_PLAN_MAX_TOKENS,
    )
    plan = WorkflowPlan.model_validate(raw)
    logger.info(
        "workflow plan: flow=%s steps=%d",
        plan.flow_name, len(plan.steps),
    )
    return plan


__all__ = [
    "PlannedBranch",
    "PlannedStep",
    "WorkflowPlan",
    "plan_workflow",
]
