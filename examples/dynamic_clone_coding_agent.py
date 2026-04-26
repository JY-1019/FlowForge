"""Zero-flow dynamic clone-coding agent.

This example starts with an empty FlowForge agent and lets dynamic generation
create the missing clone-coding flow at runtime. The generated flow is expected
to use:

- ``<clone-coding>``: a local Agent Skill created by this example
- ``<frontend-design>``: Anthropic's frontend-design Skill guidance for codegen
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
from flowforge.types import (
    AgentSkill,
    DependencyPolicy,
    FunctionTool,
    LLMConfig,
)


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_clone_coding"
TARGET_ROOT = Path(os.getenv("FLOWFORGE_CLONE_ROOT", "~/test")).expanduser()
DEFAULT_GENERATED_DIR = ".flowforge/generated/clone-coding"


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


def _dynamic_options(generated_namespace: str) -> DynamicRunOptions:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    persist_generated = os.getenv("FLOWFORGE_CLONE_PERSIST", "0") == "1"
    return DynamicRunOptions(
        project_root=str(TARGET_ROOT),
        generated_dir=f"{DEFAULT_GENERATED_DIR}/{generated_namespace}",
        auto_load_generated=False,
        persist_generated=persist_generated,
        include_builtin_tools=True,
        allow_codegen_tool_use=False,
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


def _write_frontend_design_skill() -> Path:
    """Create a local frontend-design Skill based on Anthropic's public Skill."""
    skill_dir = ARTIFACT_DIR / "skills" / "frontend-design"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: frontend-design
description: Guide production-grade frontend code generation with distinctive visual design quality.
---

# Frontend Design

Use this skill when generating frontend code, web pages, dashboards, React
components, HTML/CSS layouts, or any UI that should feel polished and specific
instead of generic.

## Code Generation Guidance

1. Start from the product context, audience, and reference URL.
2. Pick a clear visual direction that fits the reference brand or product.
3. Build real working code, not a static mock description.
4. Use cohesive design tokens for color, spacing, radius, shadows, and motion.
5. Avoid generic AI-looking defaults: predictable centered layouts, overused
   purple gradients, and interchangeable card grids.
6. Make the first viewport recognizable and useful.
7. Keep responsive behavior, accessibility, and build correctness intact.
8. Prefer concrete component structure and maintainable CSS over one-off hacks.

## Clone-Coding Notes

- Preserve the reference page's information architecture and visual hierarchy.
- Use original text and placeholder content only where appropriate; do not copy
  private assets.
- Translate brand cues into fresh code: layout rhythm, spacing, dominant color,
  interaction shape, and density.
- Generated files must build successfully with the selected frontend stack.
""",
        encoding="utf-8",
    )
    return skill_dir


def _safe_project_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return safe or "clone_site"


def _safe_project_path(project_name: str) -> Path:
    project_dir = (TARGET_ROOT / _safe_project_name(project_name)).resolve()
    try:
        project_dir.relative_to(TARGET_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"project_name must stay inside {TARGET_ROOT}") from exc
    return project_dir


def _clone_profile(target_url: str, html_excerpt: str = "") -> dict[str, Any]:
    text = f"{target_url} {html_excerpt}".lower()
    if "naver" in text:
        return {
            "brand": "NAVER",
            "accent": "#03c75a",
            "accent_dark": "#019f49",
            "tagline": "Search, news, shopping, and daily shortcuts in one calm portal.",
            "search_placeholder": "Search NAVER",
            "quick_links": [
                "Mail", "Cafe", "Blog", "Shopping", "News", "Maps", "Pay", "More",
            ],
            "cards": [
                ("Newsstand", "Headlines, press picks, and live issue snapshots."),
                ("Shopping", "Trending products with clean price and badge rows."),
                ("Weather", "Local forecast, fine dust, and commute-ready details."),
            ],
        }
    if "daangn" in text or "danggeun" in text or "karrot" in text:
        return {
            "brand": "Daangn",
            "accent": "#ff6f0f",
            "accent_dark": "#d95c00",
            "tagline": "A friendly neighborhood marketplace for nearby finds.",
            "search_placeholder": "Search nearby items",
            "quick_links": [
                "Used Deals", "Jobs", "Real Estate", "Cars", "Community", "Local Ads",
            ],
            "cards": [
                ("Popular Nearby", "Warm product cards with price, region, and time."),
                ("Neighborhood Life", "Short community posts and helpful local tips."),
                ("Trust Signals", "Manner temperature, safe meetups, and verified areas."),
            ],
        }
    return {
        "brand": "Reference Clone",
        "accent": "#2563eb",
        "accent_dark": "#1d4ed8",
        "tagline": "A responsive frontend clone based on the supplied public URL.",
        "search_placeholder": "Search this site",
        "quick_links": ["Home", "Explore", "Updates", "Products", "Support"],
        "cards": [
            ("Hero Area", "Primary message, action, and visual hierarchy."),
            ("Content Grid", "Reusable sections inspired by the reference page."),
            ("Footer", "Navigation links and lightweight trust details."),
        ],
    }


def clone_coding_scaffold(
    target_url: str,
    project_name: str,
    html_excerpt: str = "",
) -> dict[str, Any]:
    """Write a small Vite React clone project under ``~/test``.

    The generated dynamic flow should call this tool directly, then run
    ``shell_install_dependency`` and ``shell_project_exec`` as separate steps.
    """
    project_name = _safe_project_name(project_name)
    project_dir = _safe_project_path(project_name)
    profile = _clone_profile(target_url, html_excerpt)

    package_json = {
        "name": project_name,
        "private": True,
        "version": "0.1.0",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "preview": "vite preview",
        },
        "dependencies": {
            "@vitejs/plugin-react": "4.3.4",
            "vite": "5.4.21",
            "react": "18.3.1",
            "react-dom": "18.3.1",
            "lucide-react": "0.468.0",
        },
        "devDependencies": {},
    }

    app_jsx = _render_app_jsx(profile, target_url)
    styles_css = _render_styles_css(profile)
    files = {
        "package.json": json.dumps(package_json, ensure_ascii=False, indent=2) + "\n",
        "index.html": (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "  <head>\n"
            "    <meta charset=\"UTF-8\" />\n"
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
            f"    <title>{profile['brand']} Clone</title>\n"
            "  </head>\n"
            "  <body>\n"
            "    <div id=\"root\"></div>\n"
            "    <script type=\"module\" src=\"/src/main.jsx\"></script>\n"
            "  </body>\n"
            "</html>\n"
        ),
        "src/main.jsx": (
            "import React from 'react';\n"
            "import { createRoot } from 'react-dom/client';\n"
            "import App from './App.jsx';\n"
            "import './styles.css';\n\n"
            "createRoot(document.getElementById('root')).render(\n"
            "  <React.StrictMode>\n"
            "    <App />\n"
            "  </React.StrictMode>,\n"
            ");\n"
        ),
        "src/App.jsx": app_jsx,
        "src/styles.css": styles_css,
    }

    written: list[str] = []
    for rel_path, content in files.items():
        destination = (project_dir / rel_path).resolve()
        try:
            destination.relative_to(project_dir)
        except ValueError as exc:
            raise ValueError(f"refusing to write outside {project_dir}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(f"{project_name}/{rel_path}")

    return {
        "ok": True,
        "project_dir": project_name,
        "target_url": target_url,
        "files": written,
        "notes": (
            "Project files are ready. Next run npm install with "
            "shell_install_dependency, then npm run build with shell_project_exec."
        ),
    }


def _render_app_jsx(profile: dict[str, Any], target_url: str) -> str:
    quick_links = json.dumps(profile["quick_links"], ensure_ascii=False)
    cards = json.dumps(profile["cards"], ensure_ascii=False)
    return f"""import {{
  Bell,
  Grid3X3,
  MapPin,
  Menu,
  Search,
  ShoppingBag,
  Sparkles,
}} from 'lucide-react';

const quickLinks = {quick_links};
const cards = {cards};

export default function App() {{
  return (
    <main className="page-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">{profile['brand'][0]}</div>
          <strong>{profile['brand']}</strong>
        </div>
        <nav className="nav-links">
          <a>Services</a>
          <a>Content</a>
          <a>Creators</a>
        </nav>
        <button className="icon-button" aria-label="menu">
          <Menu size={{20}} />
        </button>
      </header>

      <section className="hero">
        <p className="eyebrow">Reference: {target_url}</p>
        <h1>{profile['brand']}</h1>
        <p className="tagline">{profile['tagline']}</p>
        <form className="search-box">
          <Search size={{24}} />
          <input placeholder="{profile['search_placeholder']}" />
          <button type="button">Search</button>
        </form>
      </section>

      <section className="quick-grid" aria-label="shortcuts">
        {{quickLinks.map((item, index) => (
          <button className="quick-link" key={{item}}>
            <span className="quick-icon">
              {{index % 3 === 0 ? <Grid3X3 size={{22}} /> : index % 3 === 1 ? <ShoppingBag size={{22}} /> : <MapPin size={{22}} />}}
            </span>
            <span>{{item}}</span>
          </button>
        ))}}
      </section>

      <section className="content-grid">
        {{cards.map(([title, body], index) => (
          <article className="content-card" key={{title}}>
            <div className="card-header">
              <span>{{index === 0 ? <Sparkles size={{20}} /> : <Bell size={{20}} />}}</span>
              <strong>{{title}}</strong>
            </div>
            <p>{{body}}</p>
            <div className="mock-lines">
              <span />
              <span />
              <span />
            </div>
          </article>
        ))}}
      </section>

      <footer className="footer">
        <span>Responsive clone scaffold</span>
        <span>Built with FlowForge dynamic tools</span>
      </footer>
    </main>
  );
}}
"""


def _render_styles_css(profile: dict[str, Any]) -> str:
    return f""":root {{
  --accent: {profile['accent']};
  --accent-dark: {profile['accent_dark']};
  --ink: #17202a;
  --muted: #6b7280;
  --line: #e5e7eb;
  --surface: #ffffff;
  --soft: #f6f8fb;
}}

* {{
  box-sizing: border-box;
}}

body {{
  margin: 0;
  background: var(--soft);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

button,
input {{
  font: inherit;
}}

.page-shell {{
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}}

.topbar {{
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 0 clamp(20px, 5vw, 72px);
  background: rgba(255, 255, 255, 0.92);
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  backdrop-filter: blur(16px);
  z-index: 10;
}}

.brand-lockup,
.nav-links,
.footer {{
  display: flex;
  align-items: center;
  gap: 16px;
}}

.brand-mark {{
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  color: white;
  background: var(--accent);
  border-radius: 8px;
  font-weight: 900;
}}

.nav-links a {{
  color: var(--muted);
  font-size: 14px;
}}

.icon-button {{
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
}}

.hero {{
  width: min(900px, calc(100% - 40px));
  margin: 54px auto 28px;
  text-align: center;
}}

.eyebrow {{
  color: var(--accent-dark);
  font-weight: 700;
  font-size: 13px;
}}

h1 {{
  margin: 8px 0 12px;
  color: var(--accent);
  font-size: clamp(48px, 8vw, 88px);
  letter-spacing: 0;
}}

.tagline {{
  margin: 0 auto 28px;
  max-width: 640px;
  color: var(--muted);
  line-height: 1.6;
}}

.search-box {{
  height: 64px;
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 14px;
  padding: 8px 8px 8px 22px;
  background: white;
  border: 2px solid var(--accent);
  border-radius: 8px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}}

.search-box input {{
  width: 100%;
  border: 0;
  outline: 0;
  font-size: 18px;
}}

.search-box button {{
  height: 48px;
  padding: 0 24px;
  border: 0;
  border-radius: 6px;
  color: white;
  background: var(--accent);
  font-weight: 800;
}}

.quick-grid,
.content-grid {{
  width: min(1120px, calc(100% - 40px));
  margin: 0 auto;
  display: grid;
  gap: 14px;
}}

.quick-grid {{
  grid-template-columns: repeat(8, minmax(0, 1fr));
}}

.quick-link,
.content-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
}}

.quick-link {{
  min-height: 98px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--ink);
}}

.quick-icon {{
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  color: var(--accent-dark);
  background: color-mix(in srgb, var(--accent) 10%, white);
  border-radius: 8px;
}}

.content-grid {{
  grid-template-columns: 1.2fr 1fr 1fr;
  margin-top: 18px;
  margin-bottom: 48px;
}}

.content-card {{
  min-height: 220px;
  padding: 22px;
}}

.card-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--accent-dark);
}}

.content-card p {{
  color: var(--muted);
  line-height: 1.55;
}}

.mock-lines {{
  display: grid;
  gap: 10px;
  margin-top: 22px;
}}

.mock-lines span {{
  height: 12px;
  border-radius: 999px;
  background: #eef2f7;
}}

.mock-lines span:nth-child(2) {{
  width: 72%;
}}

.mock-lines span:nth-child(3) {{
  width: 48%;
}}

.footer {{
  justify-content: space-between;
  margin-top: auto;
  padding: 22px clamp(20px, 5vw, 72px);
  color: var(--muted);
  border-top: 1px solid var(--line);
  background: white;
}}

@media (max-width: 840px) {{
  .nav-links {{
    display: none;
  }}

  .hero {{
    margin-top: 36px;
  }}

  .quick-grid {{
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }}

  .content-grid {{
    grid-template-columns: 1fr;
  }}
}}

@media (max-width: 520px) {{
  .search-box {{
    grid-template-columns: auto 1fr;
    height: auto;
  }}

  .search-box button {{
    grid-column: 1 / -1;
    width: 100%;
  }}

  .quick-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .footer {{
    align-items: flex-start;
    flex-direction: column;
  }}
}}
"""


def _build_agent(
    clone_skill_dir: Path,
    frontend_skill_dir: Path,
    llm_config: LLMConfig,
) -> type:
    tools = [
        FunctionTool(
            func=clone_coding_scaffold,
            name="clone_coding_scaffold",
            description=(
                "Create a Vite React clone-coding scaffold under the given "
                "project_name inside ~/test. Parameters: target_url, "
                "project_name, optional html_excerpt. After this direct "
                "tool call, run shell_install_dependency with npm install "
                "and shell_project_exec with npm run build."
            ),
        ),
        AgentSkill(
            path=str(frontend_skill_dir),
            name="frontend-design",
            description=(
                "Anthropic frontend-design Skill guidance for distinctive, "
                "production-grade frontend interface code generation."
            ),
        ),
        AgentSkill(
            path=str(clone_skill_dir),
            name="clone-coding",
            description=(
                "Reference-based frontend clone-coding workflow that "
                "writes a Vite app, installs npm dependencies, and builds."
            ),
        ),
    ]

    @global_config(
        prompt=(
            "You are an empty FlowForge agent. No static flow exists for app "
            "generation. In autonomous mode, create the missing dynamic flow "
            "for clone-coding requests. During dynamic code generation, use "
            "the <frontend-design> Skill guidance and the local "
            "<clone-coding> Agent Skill for reference-based clone workflow "
            "guidance. Prefer the direct clone_coding_scaffold tool "
            "to write project files, then call shell_install_dependency for "
            "npm install and shell_project_exec for npm run build. Do not use "
            "one large runtime ctx.call_llm call for file generation, install, "
            "or build work. All generated frontend project files must stay "
            "under the configured project root, which is ~/test by default."
        ),
        llm_config=llm_config,
        tools=tools,
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


def _build_request(target_url: str, project_name: str) -> dict[str, Any]:
    instructions = (
        "Create a clone-coding frontend project for the public reference URL. "
        "The FlowForge agent has no static clone-coding flow, so generate the "
        "missing flow dynamically and then execute it. Use <frontend-design> "
        "during code generation, use <clone-coding> for clone workflow "
        "guidance, prefer ctx.call_tool('clone_coding_scaffold', ...) "
        "for writing files, then call shell_install_dependency with "
        "'npm install' and shell_project_exec with 'npm run build'. Do not "
        "use a runtime ctx.call_llm call to write files, install dependencies, "
        "or run the build when direct tools are available."
    )
    return {
        "request": instructions,
        "target_url": target_url,
        "project_name": project_name,
        "project_root": str(TARGET_ROOT),
        "required_tools": [
            "frontend-design",
            "clone_coding_scaffold",
            "web_fetch_url",
            "shell_install_dependency",
            "shell_project_exec",
        ],
        "expected_commands": {
            "install": "npm install",
            "build": "npm run build",
        },
        "expected_output": [
            "project_dir",
            "target_url",
            "files",
            "install",
            "build",
            "notes",
        ],
    }


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
    project_name = _safe_project_name(args.project_name or _default_project_name(target_url))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    clone_skill_dir = _write_clone_skill()
    frontend_skill_dir = _write_frontend_design_skill()
    options = _dynamic_options(project_name)
    llm_config = _llm_config_from_env()
    agent = _build_agent(clone_skill_dir, frontend_skill_dir, llm_config)
    engine = FlowForge.compile(agent, dynamic_options=options)
    await engine.generate_docs(planning_only=True)

    print("=" * 64)
    print("Dynamic clone-coding agent - zero static flows")
    print("=" * 64)
    print(f"  Target URL: {target_url}")
    print(f"  Project root: {TARGET_ROOT}")
    print(f"  Project directory: {project_name}")
    print(f"  User flows before run: {_user_flow_names(engine) or '(none)'}")
    print(
        "  Tools: AgentSkill <frontend-design> + AgentSkill <clone-coding> "
        "+ builtin web/files/shell tools"
    )
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
