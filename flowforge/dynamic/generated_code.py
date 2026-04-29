"""Utilities for cleaning LLM-generated Python code before compile."""
from __future__ import annotations


def _normalise_generated_flow_code(
    text: str,
    flow_name: str,
    flow_prompt: str,
) -> str:
    """Return cleaned generated flow code with the top-level @flow populated."""
    code = _strip_markdown_fences(text)
    return _fill_bare_top_level_flow_decorator(code, flow_name, flow_prompt)


def _strip_markdown_fences(text: str) -> str:
    """Remove outer Markdown code fences if the LLM wrapped the code.

    Generated Python often contains prompt strings that mention Markdown
    fences such as `````json````.  The extractor therefore only treats a
    fence as closing the outer code block when it starts at column 0, which
    matches normal LLM Markdown output and avoids cutting Python strings in
    half.
    """
    text = text.strip()
    lines = text.splitlines(keepends=True)

    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if not line.startswith("```") or stripped not in {
            "```", "```python", "```py",
        }:
            continue

        closing_idx = None
        for candidate_idx in range(len(lines) - 1, idx, -1):
            candidate = lines[candidate_idx]
            if candidate.startswith("```") and candidate.strip() == "```":
                closing_idx = candidate_idx
                break
        if closing_idx is not None:
            return "".join(lines[idx + 1:closing_idx]).strip()
        break

    starts = [
        idx for idx in (
            text.find("from flowforge import"),
            text.find("@flow("),
            text.find("@flow\n"),
        )
        if idx >= 0
    ]
    if starts:
        text = text[min(starts):]

    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _fill_bare_top_level_flow_decorator(
    code: str,
    flow_name: str,
    flow_prompt: str,
) -> str:
    """Replace the first top-level empty ``@flow`` with required kwargs.

    Some repair attempts return syntactically valid code with ``@flow`` or
    ``@flow()`` on its own line.  That fails only when decorators execute,
    costing a retry even though the requested flow name and prompt are
    already known.
    """
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if line == line.lstrip() and line.strip() in {"@flow", "@flow()"}:
            lines[idx] = (
                f"@flow(name={flow_name!r}, prompt={flow_prompt!r})"
            )
            return "\n".join(lines).strip()
        if line.startswith("class "):
            break
    return code
