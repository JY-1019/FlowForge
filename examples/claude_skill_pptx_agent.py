"""Claude pptx Skill example for FlowForge.

This example shows how to expose Anthropic's built-in ``pptx`` Agent Skill
through FlowForge's annotation ``tools=[...]`` API and select it from a step
with the same angle-bracket syntax used by regular tools.

References:
- Anthropic public pptx skill:
  https://github.com/anthropics/skills/tree/main/skills/pptx
- Anthropic Skills API guide:
  https://platform.claude.com/docs/en/build-with-claude/skills-guide

Prerequisites:

    export ANTHROPIC_API_KEY="sk-ant-..."

Run:

    python examples/claude_skill_pptx_agent.py

Notes:
- The ``pptx`` Skill runs in Anthropic's code execution container.
- When Claude creates a deck, the API may return file IDs for generated
  files. Downloading those files requires Anthropic's Files API.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel, Field

from flowforge import ClaudeSkill, FlowForge, flow, global_config, step, task
from flowforge.types import LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "claude_skill_pptx"


class DeckRequest(BaseModel):
    """Input for the demo deck generation flow."""

    topic: str = Field(description="Presentation topic")
    audience: str = Field(default="executive stakeholders")
    slide_count: int = Field(default=3, ge=3, le=8)


def _llm_config_from_env() -> LLMConfig:
    model = os.getenv("FLOWFORGE_MODEL", "claude-sonnet-4-6").strip()
    return LLMConfig.for_claude(model=model, max_tokens=8192, temperature=0.2)


@global_config(
    prompt=(
        "You are a presentation design agent. "
        "When a step asks for slides, use the registered Claude pptx Skill. "
        "Prefer polished, visual decks over plain bullet lists."
    ),
    llm_config=_llm_config_from_env(),
    tools=[
        ClaudeSkill(
            name="pptx",
            type="anthropic",
            skill_id="pptx",
            version="latest",
            description="Anthropic built-in Skill for creating and editing PPTX decks.",
        )
    ],
)
class ClaudePptxAgent:
    @flow(
        name="presentation_builder",
        prompt="Create a PowerPoint deck using Anthropic's pptx Skill.",
        input_schema=DeckRequest,
        max_retries=0,
    )
    class PresentationBuilderFlow:
        @task(name="create_deck", prompt="Generate a complete PPTX presentation")
        class CreateDeckTask:
            @step(
                order=1,
                prompt=(
                    "Create a polished PowerPoint deck. Use the pptx Skill "
                    "whenever a presentation file is created or edited. "
                    "Do not stop after reading the Skill instructions; create "
                    "and export the actual PPTX file."
                ),
                timeout_seconds=600,
            )
            async def generate_presentation(ctx):
                request = ctx.input
                return await ctx.call_llm(
                    f"""
                    Create a {request.slide_count}-slide PowerPoint deck about:
                    {request.topic}

                    Audience: {request.audience}

                    Requirements:
                    - Use the Claude pptx Skill to create the deck file.
                    - Actually create and export the .pptx file; do not stop after planning.
                    - Copy the final .pptx to $OUTPUT_DIR so the API returns a file_id.
                    - Make the deck visually polished, not a plain bullet list.
                    - Include a title slide, content slide(s), and a closing slide.
                    - Use speaker notes with concise talking points.
                    - After creating the file, summarize what was created.
                    - Include any generated filename or file_id shown in the response.

                    <pptx>
                    """
                )


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required to run this example.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    engine = FlowForge.compile(ClaudePptxAgent)
    request = DeckRequest(
        topic="How AI agent workflows change enterprise software delivery",
        audience="CTO and platform engineering leadership",
        slide_count=3,
    )

    result = await engine.run(request)

    response_path = ARTIFACT_DIR / "pptx_skill_response.txt"
    response_path.write_text(str(result), encoding="utf-8")

    print("=" * 72)
    print("Claude pptx Skill FlowForge example")
    print("=" * 72)
    print(result)
    print()
    print(f"Response saved to: {response_path}")
    print(
        "If the response includes a file_id, download the generated PPTX "
        "with Anthropic's Files API."
    )


if __name__ == "__main__":
    asyncio.run(main())
