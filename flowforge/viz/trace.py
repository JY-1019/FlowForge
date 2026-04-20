"""Real-time execution trace for FlowForge."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event."""

    node_id: str
    event_type: str  # "start" | "end" | "error"
    timestamp: float = field(default_factory=time.time)
    data: Any = None
    error: str | None = None


class ExecutionTracer:
    """Collects and displays execution trace events."""

    def __init__(self, verbose: bool = True) -> None:
        self._events: list[TraceEvent] = []
        self._verbose = verbose

    def record(self, event: TraceEvent) -> None:
        self._events.append(event)
        if self._verbose:
            self._print_event(event)

    def start(self, node_id: str, data: Any = None) -> None:
        self.record(TraceEvent(node_id=node_id, event_type="start", data=data))

    def end(self, node_id: str, data: Any = None) -> None:
        self.record(TraceEvent(node_id=node_id, event_type="end", data=data))

    def error(self, node_id: str, error: str) -> None:
        self.record(TraceEvent(node_id=node_id, event_type="error", error=error))

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def _print_event(self, event: TraceEvent) -> None:
        icon = {"start": "▶", "end": "✓", "error": "✗"}.get(event.event_type, "•")
        print(f"[{icon}] {event.node_id} ({event.event_type})")
        if event.error:
            print(f"    ERROR: {event.error}")
