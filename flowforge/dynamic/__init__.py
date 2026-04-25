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
from flowforge.dynamic.generator import DynamicFlowGenerator, detect_output_artifacts
from flowforge.dynamic.manifest import (
    DynamicManifest,
    GeneratedFlowRecord,
    GeneratedToolRecord,
    load_manifest,
    save_manifest,
)

__all__ = [
    "DynamicFlowGenerator",
    "detect_output_artifacts",
    "DynamicManifest",
    "GeneratedFlowRecord",
    "GeneratedToolRecord",
    "load_manifest",
    "save_manifest",
]
