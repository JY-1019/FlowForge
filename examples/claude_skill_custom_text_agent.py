"""Custom Claude Skill example for FlowForge.

This example proves that FlowForge can pass a Claude Skill from annotation
``tools=[...]`` into Anthropic's native Skills API.  It creates a tiny custom
Skill, registers it as ``ClaudeSkill(type="custom", ...)``, calls it with the
same ``<tool_name>`` syntax as other FlowForge tools, and verifies that the
plain-text response contains a marker that only the Skill instructions require.

Unlike document Skills such as ``pptx``, this Skill does not create a file and
does not require a later Files API download.  The proof appears directly in the
Python process output.

References:
- Anthropic Skills API guide:
  https://platform.claude.com/docs/en/build-with-claude/skills-guide

Prerequisites:

    export ANTHROPIC_API_KEY="sk-ant-..."

Run:

    python examples/claude_skill_custom_text_agent.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import anthropic
from anthropic.lib import files_from_dir
from pydantic import BaseModel, Field

from flowforge import ClaudeSkill, FlowForge, flow, global_config, step, task
from flowforge.errors import ExecutionError
from flowforge.types import LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "claude_skill_custom_text"
SKILL_SOURCE_DIR = ARTIFACT_DIR / "flowforge-proof"
SKILL_ID_PATH = ARTIFACT_DIR / "skill_id.txt"
PROOF_MARKER = "FLOWFORGE_CUSTOM_SKILL_USED"


class ProofRequest(BaseModel):
    """Input request for the custom Claude Skill proof."""

    topic: str = Field(description="Short topic to summarize with the Skill")


def _llm_config_from_env() -> LLMConfig:
    model = os.getenv("FLOWFORGE_MODEL", "claude-haiku-4-5").strip()
    max_tokens = int(os.getenv("FLOWFORGE_MAX_TOKENS", "1024"))
    return LLMConfig.for_claude(model=model, max_tokens=max_tokens, temperature=0.0)


def _is_rate_limit_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        text = f"{type(current).__name__}: {current}"
        if "RateLimitError" in text or "rate_limit_error" in text:
            return True
        current = current.__cause__
    return False


def _write_skill_source() -> None:
    SKILL_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SKILL_SOURCE_DIR / "SKILL.md").write_text(
        f"""---
name: flowforge-proof
description: Plain-text verification Skill for proving FlowForge can invoke a custom Claude Skill.
---

# FlowForge Proof Skill

Use this Skill whenever the user asks to prove that FlowForge invoked a custom
Claude Skill.

Return plain text only. Do not create files.

Every response using this Skill must include these exact lines:

- marker: {PROOF_MARKER}
- skill_name: flowforge-proof

Then add:

- input_summary: one short sentence about the user's topic
- proof: one short sentence saying this marker came from the Skill instructions
""",
        encoding="utf-8",
    )


def _ensure_custom_skill_id() -> str:
    from_env = os.getenv("FLOWFORGE_CUSTOM_SKILL_ID", "").strip()
    if from_env:
        return from_env

    if SKILL_ID_PATH.exists():
        cached = SKILL_ID_PATH.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    _write_skill_source()
    client = anthropic.Anthropic()
    skill = client.beta.skills.create(
        display_title="FlowForge Proof",
        files=files_from_dir(SKILL_SOURCE_DIR),
        betas=["skills-2025-10-02"],
    )
    SKILL_ID_PATH.write_text(skill.id, encoding="utf-8")
    return skill.id


def build_agent(skill_id: str):
    """Build the FlowForge agent after the custom Skill ID is known."""

    @global_config(
        prompt=(
            "You are a verification assistant. When the user asks to prove "
            "custom Claude Skill usage, use the registered FlowForge proof Skill."
        ),
        llm_config=_llm_config_from_env(),
        tools=[
            ClaudeSkill(
                name="flowforge_proof",
                type="custom",
                skill_id=skill_id,
                version="latest",
                description="Custom proof Skill that returns a fixed marker.",
            )
        ],
    )
    class CustomClaudeSkillAgent:
        @flow(
            name="custom_skill_proof",
            prompt="Prove that a custom Claude Skill is available to this FlowForge run.",
            input_schema=ProofRequest,
            max_retries=0,
        )
        class CustomSkillProofFlow:
            @task(name="prove", prompt="Use the custom Skill to produce proof text")
            class ProveTask:
                @step(
                    order=1,
                    prompt="Call Claude with the custom Skill attached.",
                    timeout_seconds=180,
                )
                async def prove_custom_skill(ctx):
                    return await ctx.call_llm(
                        f"""
                        Use the FlowForge proof Skill to answer this request:

                        Topic: {ctx.input.topic}

                        The final answer must include the marker required by
                        the Skill instructions.

                        <flowforge_proof>
                        """
                    )

    return CustomClaudeSkillAgent


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required to run this example.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        skill_id = _ensure_custom_skill_id()
        engine = FlowForge.compile(build_agent(skill_id))
        result = await engine.run(
            ProofRequest(topic="FlowForge annotation tools can carry Claude Skills")
        )
    except ExecutionError as exc:
        if _is_rate_limit_error(exc):
            raise SystemExit(
                "Anthropic rate limit hit while running the custom Skill. "
                "Wait for the minute window to reset, then retry."
            ) from exc
        raise

    result_text = str(result)
    output_path = ARTIFACT_DIR / "custom_skill_response.md"
    output_path.write_text(result_text, encoding="utf-8")

    if PROOF_MARKER not in result_text:
        raise SystemExit(
            "The request completed, but the expected Skill marker was missing. "
            f"Expected marker: {PROOF_MARKER}"
        )

    print("=" * 72)
    print("FlowForge custom Claude Skill proof")
    print("=" * 72)
    print(f"custom skill_id: {skill_id}")
    print(result_text)
    print()
    print(f"Response saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
