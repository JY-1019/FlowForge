"""Zero-flow dynamic docx report agent.

This example fetches the top 3 stories from https://news.hada.io and writes
them into a Korean-language ``.docx`` report — entirely through the dynamic
generator.  The example file intentionally defines NO flows, NO helper
functions, and NO custom tools.  The agent class is empty; everything is
synthesised at runtime from the user request and the built-in tool catalog.

Run:

    python examples/dynamic_docx_report_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from flowforge import DynamicRunOptions, FlowForge, global_config
from flowforge.types import LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.getenv(
        "FLOWFORGE_DOCX_PROJECT_ROOT",
        str(ROOT_DIR / "_artifacts" / "dynamic_docx_report"),
    )
).expanduser()
TARGET_ROOT = Path(
    os.getenv(
        "FLOWFORGE_DOCX_ROOT",
        str(PROJECT_ROOT / "output"),
    )
).expanduser()
OUTPUT_PATH = TARGET_ROOT / "hada_top3.docx"
SOURCE_URL = "https://news.hada.io"


@global_config(
    prompt=(
        "You are an empty FlowForge agent. There are no static flows, "
        "Skills, or preregistered tools. In autonomous mode, generate a "
        "dynamic flow that fulfils the user's request using only the "
        "built-in tools available (web fetch, files, docx, etc.)."
    ),
    llm_config=LLMConfig.for_claude(
        model=os.getenv("FLOWFORGE_MODEL", "claude-sonnet-4-6"),
        temperature=0.2,
        max_tokens=int(os.getenv("FLOWFORGE_MAX_TOKENS", "8192")),
        verify_ssl=False,
    ),
    tools=[],
    dynamic_flow=True,
    include_builtin_tools=True,
)
class DynamicDocxReportAgent:
    """No static flows. The report-authoring flow is generated at runtime."""


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required because this example calls the "
            "Anthropic Messages API."
        )

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    options = DynamicRunOptions(
        project_root=str(PROJECT_ROOT),
        generated_dir="generated",
        auto_load_generated=os.getenv("FLOWFORGE_DOCX_AUTOLOAD", "1") == "1",
        persist_generated=os.getenv("FLOWFORGE_DOCX_PERSIST", "1") == "1",
        include_builtin_tools=True,
        allow_codegen_tool_use=False,
        shell_output_max_chars=int(os.getenv("FLOWFORGE_DOCX_FETCH_CHARS", "50000")),
    )

    engine = FlowForge.compile(DynamicDocxReportAgent, dynamic_options=options)
    await engine.generate_docs(planning_only=True)

    request = {
        "request": (
            f"Fetch the top 3 stories currently listed on {SOURCE_URL} and "
            "write a Korean-language .docx report summarising them. For "
            "each story include: title, source/article URL, and a 3-5 "
            "sentence summary in Korean. Save the .docx file to the path "
            "given by `output_path`. If the fetched page is truncated or "
            "real story titles/URLs cannot be extracted, raise an error "
            "instead of writing placeholder stories."
        ),
        "source_url": SOURCE_URL,
        "output_path": str(OUTPUT_PATH),
        "expected_output": ["output_path", "stories", "notes"],
    }

    print("=" * 64)
    print("Dynamic docx report agent - zero static flows, zero tools")
    print("=" * 64)
    print(f"  Source: {SOURCE_URL}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  persist_generated:    {options.persist_generated}")
    print(f"  auto_load_generated:  {options.auto_load_generated}")
    print()

    result, _trace = await engine.run_traced(
        request, planning_mode="autonomous", dynamic_options=options,
    )

    print("\nResult:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\nOutput docx: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
