"""Zero-flow dynamic clone-coding agent.

This example starts with an empty FlowForge agent and lets dynamic generation
create the missing clone-coding flow at runtime. The generated flow is expected
to use:

- ``<clone-coding>``: a local Agent Skill created by this example
- ``web_fetch_url``: inspect the public target page when available
- ``files_write_text``: write the generated frontend project
- ``shell_install_dependency``: run ``npm install`` inside the project
- ``shell_project_exec``: run ``npm run build`` after installation

All generated app files are sandboxed under ``~/test`` by
``DynamicRunOptions.project_root``.

Run:

    python examples/dynamic_clone_coding_agent.py https://www.naver.com/
    python examples/dynamic_clone_coding_agent.py https://www.daangn.com/ daangn_clone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from flowforge import DynamicRunOptions, FlowForge, global_config
from flowforge.schema.dag import NodeType
from flowforge.types import AgentSkill, DependencyPolicy, LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_clone_coding"
TARGET_ROOT = Path(os.getenv("FLOWFORGE_CLONE_ROOT", "~/test")).expanduser()
GENERATED_DIR = ".flowforge/generated"


def _llm_config_from_env() -> LLMConfig:
    provider = os.getenv("FLOWFORGE_PROVIDER", "").strip().lower()
    model = os.getenv("FLOWFORGE_MODEL", "").strip()
    max_tokens = int(os.getenv("FLOWFORGE_MAX_TOKENS", "8192"))

    kwargs = {"temperature": 0.2, "max_tokens": max_tokens}
    if provider == "openai":
        return LLMConfig.for_openai(model=model or "gpt-4o", **kwargs)
    if provider == "google":
        return LLMConfig.for_gemini(model=model or "gemini-2.0-flash", **kwargs)
    return LLMConfig.for_claude(model=model or "claude-sonnet-4-6", **kwargs)


def _dynamic_options() -> DynamicRunOptions:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    return DynamicRunOptions(
        project_root=str(TARGET_ROOT),
        generated_dir=GENERATED_DIR,
        auto_load_generated=False,
        persist_generated=True,
        include_builtin_tools=True,
        allow_codegen_tool_use=True,
        allowed_shell_modes=[
            "readonly",
            "workspace_write",
            "project_exec",
            "install_dependency",
        ],
        shell_timeout_seconds=300,
        shell_output_max_chars=8000,
        project_context_max_chars=6000,
        dependency_policy=DependencyPolicy(
            allow_install=True,
            allowed_managers=["pip", "uv", "npm", "pnpm", "yarn"],
        ),
    )


def _write_clone_skill() -> Path:
    skill_dir = ARTIFACT_DIR / "skills" / "clone-coding"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: clone-coding
description: Build frontend clone projects from public web page references.
---

# Clone Coding

Use this skill when the user asks for a clone, redesign, or reference-based
frontend implementation from a public URL.

## Workflow

1. Inspect the target URL with `web_fetch_url` when the page is public.
2. Extract visible layout, navigation, content groups, colors, typography, and
   interaction cues. Do not copy private assets or claim pixel perfection.
3. Create a small Vite React project unless the user asks for another stack.
4. Use relative paths under the configured project root, for example
   `naver_clone/package.json` and `naver_clone/src/App.jsx`.
5. Write files with `files_write_text`. Parent directories are created by the
   tool, so shell redirection and chained shell commands are unnecessary.
6. Run `shell_install_dependency` with `command="npm install"` in the project
   directory after writing `package.json`.
7. Run `shell_project_exec` with `command="npm run build"` in the project
   directory after installation.
8. Return a compact JSON-compatible dict with `project_dir`, `target_url`,
   `files`, `install`, `build`, and `notes`.

## Frontend Quality Bar

- Make the first screen look like the referenced product, not a generic demo.
- Build responsive desktop and mobile layouts.
- Prefer concrete UI structure over placeholder text.
- Keep generated code self-contained and easy to run with `npm run dev`.
""",
        encoding="utf-8",
    )
    return skill_dir


def _build_agent(skill_dir: Path, llm_config: LLMConfig) -> type:
    @global_config(
        prompt=(
            "You are an empty FlowForge agent. No static flow exists for app "
            "generation. In autonomous mode, create the missing dynamic flow "
            "for clone-coding requests. Use the local <clone-coding> Agent "
            "Skill and the builtin tools for web fetch, file writes, npm "
            "dependency installation, and build verification. All generated "
            "frontend project files must stay under the configured project "
            "root, which is ~/test by default."
        ),
        llm_config=llm_config,
        tools=[
            AgentSkill(
                path=str(skill_dir),
                name="clone-coding",
                description=(
                    "Reference-based frontend clone-coding workflow that "
                    "writes a Vite app, installs npm dependencies, and builds."
                ),
            )
        ],
        dynamic_flow=True,
        include_builtin_tools=True,
    )
    class DynamicCloneCodingAgent:
        """No static flows. The clone-coding flow is generated at runtime."""

    return DynamicCloneCodingAgent


def _default_project_name(target_url: str) -> str:
    parsed = urlparse(target_url)
    host = parsed.netloc or parsed.path or "site"
    host = host.lower().removeprefix("www.")
    stem = re.sub(r"[^a-z0-9]+", "_", host).strip("_") or "site"
    return f"{stem}_clone"


def _build_request(target_url: str, project_name: str) -> str:
    return f"""
Create a clone-coding frontend project for this public reference URL:
{target_url}

Project constraints:
- The FlowForge agent has no static clone-coding flow. Generate the missing
  flow dynamically and then execute it.
- Use <clone-coding> for implementation guidance.
- Use web_fetch_url to inspect the target page when possible.
- Write the app under this relative directory: {project_name}
- The absolute sandbox root is {TARGET_ROOT}; do not write outside it.
- Use files_write_text for package.json, index.html, src/main.jsx,
  src/App.jsx, and src/styles.css or equivalent files.
- Use shell_install_dependency with command "npm install" and cwd
  "{project_name}" after package.json is written.
- Use shell_project_exec with command "npm run build" and cwd
  "{project_name}" after installation.
- Return a JSON-compatible dict with project_dir, target_url, files, install,
  build, and notes.
"""


def _user_flow_names(engine: Any) -> list[str]:
    return [
        node.name
        for node in engine.dag.get_children("global")
        if node.type == NodeType.FLOW and not node.name.startswith("_")
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dynamic clone-coding flow that writes a frontend "
            "project under ~/test and runs npm install."
        )
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=os.getenv("FLOWFORGE_CLONE_URL", "https://www.naver.com/"),
        help="Public reference URL to clone.",
    )
    parser.add_argument(
        "project_name",
        nargs="?",
        help="Directory name under ~/test. Defaults to a name from the URL.",
    )
    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    target_url = args.url
    project_name = args.project_name or _default_project_name(target_url)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    skill_dir = _write_clone_skill()
    options = _dynamic_options()
    agent = _build_agent(skill_dir, _llm_config_from_env())
    engine = FlowForge.compile(agent, dynamic_options=options)
    await engine.generate_docs(planning_only=True)

    print("=" * 64)
    print("Dynamic clone-coding agent - zero static flows")
    print("=" * 64)
    print(f"  Target URL: {target_url}")
    print(f"  Project root: {TARGET_ROOT}")
    print(f"  Project directory: {project_name}")
    print(f"  User flows before run: {_user_flow_names(engine) or '(none)'}")
    print("  Tools: AgentSkill <clone-coding> + builtin web/files/shell tools")
    print()

    result, trace = await engine.run_traced(
        _build_request(target_url, project_name),
        planning_mode="autonomous",
        dynamic_options=options,
    )

    mermaid_path = ARTIFACT_DIR / "dynamic_clone_coding_run.md"
    mermaid_path.write_text(engine.compare_mermaid(trace), encoding="utf-8")

    dynamic_info = engine.last_dynamic_generation or {}
    if "generated" in dynamic_info:
        for idx, gen in enumerate(dynamic_info["generated"], 1):
            if gen.get("generated_code"):
                name = gen.get("dynamic_flow", f"flow_{idx}")
                path = ARTIFACT_DIR / f"{name}.py"
                path.write_text(gen["generated_code"], encoding="utf-8")
                print(f"  Generated flow {idx}: {name}")
    elif dynamic_info.get("generated_code"):
        name = dynamic_info.get("dynamic_flow", "dynamic_flow")
        path = ARTIFACT_DIR / f"{name}.py"
        path.write_text(dynamic_info["generated_code"], encoding="utf-8")
        print(f"  Generated flow: {name}")

    print(f"\n  User flows after run: {_user_flow_names(engine) or '(none)'}")
    print(f"  Mermaid: {mermaid_path}")
    print(f"  App directory: {TARGET_ROOT / project_name}")
    print("\nResult:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
