"""OTel span instrumentation: flow/task/step produce a nested span tree.

Skipped automatically when ``opentelemetry-sdk`` is not installed (the
``otel`` extra is optional).
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from flowforge import FlowForge, flow, global_config, step, task


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


@global_config(prompt="otel test agent")
class _Agent:
    @flow(name="f", prompt="flow", input_schema=_In, output_schema=_Out)
    class F:
        @task(name="t", prompt="task", input_schema=_In, output_schema=_Out)
        class T:
            @step(order=1, prompt="s1", input_schema=_In, output_schema=_In)
            async def s1(ctx):
                return {"value": ctx.input.value + 1}

            @step(order=2, prompt="s2", input_schema=_In, output_schema=_Out)
            async def s2(ctx):
                return {"value": ctx.input.value * 2}


# set_tracer_provider only takes effect once per process, so the exporter is
# created once at module scope and cleared before each test.  The proxy tracer
# obtained in observability.py resolves to whatever provider is current.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
try:
    trace.set_tracer_provider(_PROVIDER)
except Exception:
    pass


@pytest.fixture()
def exporter() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


async def test_span_hierarchy(exporter: InMemorySpanExporter) -> None:
    engine = FlowForge.compile(_Agent)
    result = await engine.run(_In(value=10))
    assert result.value == 22  # (10 + 1) * 2

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    names = {s.name for s in spans}

    # All four hierarchy levels are present.
    assert "flowforge.run" in names
    assert "global.f" in names
    assert "global.f.t" in names
    assert "global.f.t.s1[1]" in names
    assert "global.f.t.s2[2]" in names

    def parent_name(span):
        if span.parent is None:
            return None
        return by_id[span.parent.span_id].name

    span_by_name = {s.name: s for s in spans}

    # Parent/child nesting: step -> task -> flow -> run.
    assert parent_name(span_by_name["global.f.t.s1[1]"]) == "global.f.t"
    assert parent_name(span_by_name["global.f.t.s2[2]"]) == "global.f.t"
    assert parent_name(span_by_name["global.f.t"]) == "global.f"
    assert parent_name(span_by_name["global.f"]) == "flowforge.run"

    # node_type attributes are tagged correctly.
    assert span_by_name["global.f"].attributes["flowforge.node_type"] == "flow"
    assert span_by_name["global.f.t"].attributes["flowforge.node_type"] == "task"
    assert span_by_name["global.f.t.s1[1]"].attributes["flowforge.node_type"] == "step"
    assert span_by_name["global.f.t.s1[1]"].attributes["flowforge.order"] == 1


async def test_tool_span_emitted(exporter: InMemorySpanExporter) -> None:
    """execute_raw wraps each tool call in a `tool <name>` span."""
    from flowforge.execution.tool_executor import ToolExecutor
    from flowforge.types import FunctionTool

    tool = FunctionTool(func=lambda value: {"echo": value}, name="echo_tool")
    executor = ToolExecutor([tool])
    try:
        result = await executor.execute_raw("echo_tool", {"value": 7})
    finally:
        await executor.close()
    assert result == {"echo": 7}

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "tool echo_tool" in spans
    span = spans["tool echo_tool"]
    assert span.attributes["flowforge.tool.name"] == "echo_tool"
    assert span.attributes["flowforge.tool.type"] == "function"


def test_llm_span_records_usage(exporter: InMemorySpanExporter) -> None:
    """llm_span records GenAI attributes and accumulates token usage."""
    from flowforge.observability import llm_span, record_llm_usage

    with llm_span("anthropic", "claude-sonnet-4-6"):
        record_llm_usage(10, 5)
        record_llm_usage(3, 2)  # second tool-use round accumulates

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "llm.chat claude-sonnet-4-6" in spans
    span = spans["llm.chat claude-sonnet-4-6"]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.system"] == "anthropic"
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert span.attributes["gen_ai.usage.input_tokens"] == 13
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
