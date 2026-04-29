"""FlowForge dynamic flow generation.

When ``@global_config(dynamic_flow=True)`` is set, the framework can
generate new flows at runtime when no existing flow matches a user query.

Public API
----------
``DynamicFlowGenerator``
    Orchestrates the full lifecycle: gap analysis → code generation →
    compile → inject → execute.  Used internally by the ``ExecutionEngine``
    and also available for programmatic use.
"""
from flowforge.dynamic.capability import (
    CapabilitySelection,
    StepCapability,
    heuristic_capability_selection,
    select_capabilities,
)
from flowforge.dynamic.generator import DynamicFlowGenerator, detect_output_artifacts
from flowforge.dynamic.manifest import (
    DynamicManifest,
    GeneratedFlowRecord,
    GeneratedToolRecord,
    load_manifest,
    save_manifest,
)
from flowforge.dynamic.mcp_provision import (
    McpProvisionRecord,
    provision_mcp_servers,
)
from flowforge.dynamic.plan import (
    PlannedBranch,
    PlannedStep,
    WorkflowPlan,
    heuristic_workflow_plan,
    plan_workflow,
)

__all__ = [
    "CapabilitySelection",
    "DynamicFlowGenerator",
    "DynamicManifest",
    "GeneratedFlowRecord",
    "GeneratedToolRecord",
    "McpProvisionRecord",
    "PlannedBranch",
    "PlannedStep",
    "StepCapability",
    "WorkflowPlan",
    "detect_output_artifacts",
    "heuristic_capability_selection",
    "heuristic_workflow_plan",
    "load_manifest",
    "plan_workflow",
    "provision_mcp_servers",
    "save_manifest",
    "select_capabilities",
]
