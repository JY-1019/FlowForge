"""Session memory for FlowForge — persists across multiple ``run()`` calls.

Two levels of memory:

1. **Session memory** (``SessionMemory``)
   Stores summaries of previous runs on the ``ExecutionEngine``.  Each run
   automatically records a compact entry (input summary, route, output
   summary).  The LLM receives this as context in subsequent runs so it
   can reference earlier results.

2. **Step conversation history** (``StepHistory``)
   Within a single run, each step's LLM interaction (prompt → response
   summary) is recorded in the ``TaskContext``.  Subsequent steps receive
   a condensed version so the LLM knows what happened earlier in the
   pipeline without replaying full conversations.

Token budget
------------
- Session memory: keeps the last ``max_entries`` (default 10) run summaries.
  Each entry is capped at ``max_entry_chars`` (default 300) characters.
- Step history: keeps the last ``max_entries`` (default 20) LLM interactions.
  Each entry is capped at ``max_step_chars`` (default 200) characters.
  Injected as a compact block in the system prompt.

Both can be cleared by the user at any time via ``engine.memory.clear()``
or ``ctx.clear_step_history()``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMemoryEntry:
    """A single run's summary stored in session memory."""

    run_index: int
    input_summary: str
    route: str | None
    output_summary: str
    planning_mode: str = "deterministic"

    def to_text(self) -> str:
        """Render as a compact one-liner for LLM context."""
        parts = [f"[Run {self.run_index}]"]
        if self.route:
            parts.append(f"route={self.route}")
        parts.append(f"input: {self.input_summary}")
        parts.append(f"output: {self.output_summary}")
        return "  ".join(parts)


class SessionMemory:
    """Stores summaries of previous runs for cross-run context.

    Attached to ``ExecutionEngine`` and survives across multiple ``run()``
    calls.  The memory is injected into the LLM system prompt as a compact
    "previous runs" block.

    Parameters
    ----------
    max_entries:
        Maximum number of run summaries to keep.  Oldest entries are
        dropped when the limit is exceeded.
    max_entry_chars:
        Maximum characters per entry summary (input + output).
    """

    def __init__(
        self,
        max_entries: int = 10,
        max_entry_chars: int = 300,
    ) -> None:
        self.max_entries = max_entries
        self.max_entry_chars = max_entry_chars
        self._entries: list[RunMemoryEntry] = []
        self._run_counter: int = 0

    @property
    def entries(self) -> list[RunMemoryEntry]:
        """All stored run summaries (oldest first)."""
        return list(self._entries)

    def record_run(
        self,
        input_data: Any,
        output_data: Any,
        route: str | list[str] | None = None,
        planning_mode: str = "deterministic",
    ) -> None:
        """Record a completed run's summary."""
        self._run_counter += 1

        input_summary = self._summarize(input_data, self.max_entry_chars // 2)
        output_summary = self._summarize(output_data, self.max_entry_chars // 2)

        route_str = None
        if route is not None:
            route_str = ", ".join(route) if isinstance(route, list) else route

        entry = RunMemoryEntry(
            run_index=self._run_counter,
            input_summary=input_summary,
            route=route_str,
            output_summary=output_summary,
            planning_mode=planning_mode,
        )
        self._entries.append(entry)

        # Evict oldest if over limit.
        if self.max_entries <= 0:
            self._entries.clear()
        elif len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def to_prompt_block(self) -> str:
        """Render all entries as a prompt block for the LLM.

        Returns an empty string if no entries exist.
        """
        if not self._entries:
            return ""

        lines = ["## Previous Runs (session memory)"]
        for entry in self._entries:
            lines.append(entry.to_text())
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored run summaries."""
        self._entries.clear()
        self._run_counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return len(self._entries) > 0

    @staticmethod
    def _summarize(data: Any, max_chars: int) -> str:
        """Create a compact text summary of data."""
        if data is None:
            return "(none)"
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = json.dumps(data, ensure_ascii=False, default=str)
        elif hasattr(data, "model_dump"):
            text = json.dumps(data.model_dump(), ensure_ascii=False, default=str)
        else:
            text = str(data)

        if len(text) > max_chars:
            if max_chars <= 0:
                return ""
            if max_chars <= 3:
                return "." * max_chars
            text = text[:max_chars - 3] + "..."
        return text


@dataclass
class StepHistoryEntry:
    """A single step's LLM interaction summary."""

    step_name: str
    order: int
    prompt_summary: str
    response_summary: str

    def to_text(self) -> str:
        return f"[step {self.order}: {self.step_name}] {self.response_summary}"


class StepHistory:
    """Tracks step-level LLM interactions within a single task run.

    Stored on ``TaskContext`` and accessible from subsequent steps.
    Each step's LLM prompt and response are summarized and added here.
    The next step receives this as context in its system prompt.

    Parameters
    ----------
    max_entries:
        Maximum number of step summaries to keep. Oldest entries are dropped
        when the limit is exceeded.
    max_step_chars:
        Maximum characters per step response summary.
    """

    def __init__(self, max_entries: int = 20, max_step_chars: int = 200) -> None:
        self.max_entries = max_entries
        self.max_step_chars = max_step_chars
        self._entries: list[StepHistoryEntry] = []

    @property
    def entries(self) -> list[StepHistoryEntry]:
        return list(self._entries)

    def record_step(
        self,
        step_name: str,
        order: int,
        prompt: str,
        response: Any,
    ) -> None:
        """Record a step's LLM interaction."""
        prompt_summary = prompt[:100] + "..." if len(prompt) > 100 else prompt
        resp_text = self._to_text(response)
        if len(resp_text) > self.max_step_chars:
            resp_text = resp_text[:self.max_step_chars - 3] + "..."

        self._entries.append(StepHistoryEntry(
            step_name=step_name,
            order=order,
            prompt_summary=prompt_summary,
            response_summary=resp_text,
        ))
        if self.max_entries <= 0:
            self._entries.clear()
        elif len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def to_prompt_block(self) -> str:
        """Render as a compact prompt block for subsequent steps."""
        if not self._entries:
            return ""

        lines = ["## Previous Steps (this task)"]
        for entry in self._entries:
            lines.append(entry.to_text())
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all step history."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return len(self._entries) > 0

    @staticmethod
    def _to_text(data: Any) -> str:
        if data is None:
            return "(none)"
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, default=str)
        if hasattr(data, "model_dump"):
            return json.dumps(data.model_dump(), ensure_ascii=False, default=str)
        return str(data)
