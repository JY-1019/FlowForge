"""OpenTelemetry span instrumentation for the FlowForge execution hierarchy.

Every ``@flow`` / ``@task`` / ``@step`` execution is wrapped in an OTel span,
so a single agent run produces a span tree mirroring the DAG::

    flowforge.run
    └─ global.credit_analysis            (flow)
       └─ global.credit_analysis.assess  (task)
          ├─ ...assess.collect[1]        (step)
          ├─ ...assess.ratios[2]         (step)
          └─ ...

Design
------
* **API-only dependency.** This module imports ``opentelemetry-api`` only.
  Applications wire up an SDK + exporter (OTLP, console, …); when no SDK is
  configured the OTel API returns no-op spans, so instrumentation is free.
* **Graceful absence.** If ``opentelemetry`` is not installed at all (the
  ``otel`` extra was not selected), :func:`node_span` becomes a no-op context
  manager and execution is unaffected.
* **Opt-out.** Set ``FLOWFORGE_OTEL=0`` (or ``false``/``off``) to disable
  span creation even when OpenTelemetry is installed.

Parent/child nesting works automatically: :func:`node_span` uses
``start_as_current_span``, which attaches the span to the OTel context
(``contextvars``).  Child node executions awaited within that context — and
those scheduled concurrently via ``asyncio.gather`` — inherit it as parent.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

__all__ = [
    "node_span",
    "llm_span",
    "tool_span",
    "record_llm_usage",
    "otel_enabled",
]


def _is_disabled() -> bool:
    return os.getenv("FLOWFORGE_OTEL", "").strip().lower() in {"0", "false", "off", "no"}


_OTEL_ENABLED = False
_tracer = None

if not _is_disabled():
    try:  # pragma: no cover - depends on optional dependency
        from opentelemetry import trace as _otel_trace

        try:
            from importlib.metadata import version as _pkg_version

            _ff_version = _pkg_version("flowforge")
        except Exception:  # noqa: BLE001
            _ff_version = ""

        _tracer = _otel_trace.get_tracer("flowforge", _ff_version)
        _OTEL_ENABLED = True
    except Exception:  # noqa: BLE001 - opentelemetry not installed
        _OTEL_ENABLED = False


def otel_enabled() -> bool:
    """Return True when OTel spans are actually being emitted."""
    return _OTEL_ENABLED


@contextmanager
def node_span(
    node_id: str,
    node_type: str,
    *,
    name: str,
    order: int | None = None,
    is_branch: bool = False,
    pass_criteria: bool = False,
) -> Iterator[object | None]:
    """Open an OTel span for one DAG node execution.

    Parameters
    ----------
    node_id:
        Full DAG path (e.g. ``"global.credit_analysis.assess.collect[1]"``);
        used as the span name so traces are navigable by node path.
    node_type:
        ``"run"`` | ``"flow"`` | ``"task"`` | ``"step"``.
    name, order, is_branch, pass_criteria:
        Node metadata recorded as ``flowforge.*`` span attributes.

    Yields the active span (or ``None`` when instrumentation is disabled).
    The context manager records exceptions and sets span status to ERROR
    automatically when the wrapped block raises.
    """
    if not _OTEL_ENABLED or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(node_id) as span:
        try:
            span.set_attribute("flowforge.node_id", node_id)
            span.set_attribute("flowforge.node_type", node_type)
            span.set_attribute("flowforge.node_name", name)
            if order is not None:
                span.set_attribute("flowforge.order", order)
            span.set_attribute("flowforge.is_branch", is_branch)
            if node_type == "step":
                span.set_attribute("flowforge.pass_criteria", pass_criteria)
        except Exception:  # noqa: BLE001 - never let instrumentation break a run
            pass
        yield span


# Accumulates token usage for the LLM call currently in scope.  Provider loops
# may return from several points and run multiple tool-use rounds; each round
# adds to the running totals on the active span via ``record_llm_usage``.
_llm_usage: ContextVar[tuple[int, int] | None] = ContextVar("_llm_usage", default=None)


@contextmanager
def llm_span(provider: str, model: str) -> Iterator[object | None]:
    """Open a child span for one LLM call (``call_llm_api``).

    Records GenAI semantic-convention attributes (``gen_ai.system``,
    ``gen_ai.request.model``, ``gen_ai.operation.name``).  Token usage is
    accumulated through :func:`record_llm_usage` for the duration of the span.

    Yields the active span (or ``None`` when instrumentation is disabled).
    """
    if not _OTEL_ENABLED or _tracer is None:
        yield None
        return

    span_name = f"llm.chat {model}" if model else "llm.chat"
    token = _llm_usage.set((0, 0))
    try:
        with _tracer.start_as_current_span(span_name) as span:
            try:
                span.set_attribute("gen_ai.operation.name", "chat")
                if provider:
                    span.set_attribute("gen_ai.system", provider)
                if model:
                    span.set_attribute("gen_ai.request.model", model)
            except Exception:  # noqa: BLE001
                pass
            yield span
    finally:
        _llm_usage.reset(token)


def record_llm_usage(input_tokens: int, output_tokens: int) -> None:
    """Add token counts to the LLM span currently in scope.

    Safe to call repeatedly (e.g. once per tool-use round); the running totals
    are written to the active span as ``gen_ai.usage.{input,output}_tokens``.
    No-op when instrumentation is disabled or no :func:`llm_span` is open.
    """
    if not _OTEL_ENABLED:
        return
    current = _llm_usage.get()
    if current is None:
        return
    total_in = current[0] + int(input_tokens or 0)
    total_out = current[1] + int(output_tokens or 0)
    _llm_usage.set((total_in, total_out))
    try:  # pragma: no cover - depends on optional dependency
        span = _otel_trace.get_current_span()
        span.set_attribute("gen_ai.usage.input_tokens", total_in)
        span.set_attribute("gen_ai.usage.output_tokens", total_out)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def tool_span(tool_name: str, tool_type: str) -> Iterator[object | None]:
    """Open a child span for one tool invocation (``execute_raw``).

    Records ``flowforge.tool.name`` and ``flowforge.tool.type`` attributes.
    Yields the active span (or ``None`` when instrumentation is disabled).
    """
    if not _OTEL_ENABLED or _tracer is None:
        yield None
        return

    span_name = f"tool {tool_name}" if tool_name else "tool"
    with _tracer.start_as_current_span(span_name) as span:
        try:
            if tool_name:
                span.set_attribute("flowforge.tool.name", tool_name)
            if tool_type:
                span.set_attribute("flowforge.tool.type", tool_type)
        except Exception:  # noqa: BLE001
            pass
        yield span
