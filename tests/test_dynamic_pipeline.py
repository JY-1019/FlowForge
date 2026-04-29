"""Tests for the 4-phase dynamic generator pipeline.

Phases under test:
1. ``flowforge.dynamic.plan.plan_workflow``       — LLM produces a WorkflowPlan
2. ``flowforge.dynamic.capability.select_capabilities`` — per-step mode/tool
3. ``flowforge.dynamic.mcp_provision.provision_mcp_servers`` — _artifact + manifest
4. ``DynamicFlowGenerator.generate_flow_code_from_plan`` — code synthesis
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from flowforge import FlowForge, flow, global_config, step, task
from flowforge.dynamic import (
    CapabilitySelection,
    PlannedBranch,
    PlannedStep,
    WorkflowPlan,
    plan_workflow,
    provision_mcp_servers,
    select_capabilities,
)
from flowforge.dynamic.capability import StepCapability
from flowforge.dynamic.generator import DynamicFlowGenerator
from flowforge.dynamic.manifest import load_manifest
from flowforge.types import (
    AgentSkill,
    DynamicRunOptions,
    FunctionTool,
    LLMConfig,
)


@flow(name="alpha", prompt="alpha")
class _Alpha:
    @task(name="t", prompt="t")
    class _T:
        @step(order=1, prompt="run")
        async def run(ctx):
            return {"ok": True}


@global_config(prompt="agent")
class _Agent:
    Alpha = _Alpha


# ---------------------------------------------------------------------------
# Phase 1 — workflow planning
# ---------------------------------------------------------------------------


class TestPlanWorkflow:

    def test_plan_validates_dense_orders(self):
        with pytest.raises(Exception):
            WorkflowPlan(
                flow_name="x",
                flow_prompt="produce x",
                task_name="t",
                task_prompt="task t",
                top_class="XFlow",
                steps=[
                    PlannedStep(
                        name="a", order=1, purpose="step a long enough",
                        needs_llm_reasoning=False,
                    ),
                    PlannedStep(
                        name="b", order=3, purpose="step b long enough",
                        needs_llm_reasoning=False,
                    ),
                ],
            )

    def test_plan_rejects_duplicate_step_names(self):
        with pytest.raises(Exception):
            WorkflowPlan(
                flow_name="x",
                flow_prompt="produce x",
                task_name="t",
                task_prompt="task t",
                top_class="XFlow",
                steps=[
                    PlannedStep(
                        name="a", order=1, purpose="step a long enough",
                        needs_llm_reasoning=False,
                    ),
                    PlannedStep(
                        name="a", order=2, purpose="step a long enough",
                        needs_llm_reasoning=False,
                    ),
                ],
            )

    @pytest.mark.asyncio
    async def test_plan_workflow_calls_llm_and_validates(self):
        fake_response = {
            "flow_name": "fetch_papers",
            "flow_prompt": "fetch trending papers and summarise",
            "task_name": "main",
            "task_prompt": "fetch and analyse papers",
            "top_class": "FetchPapersFlow",
            "steps": [
                {
                    "name": "fetch",
                    "order": 1,
                    "purpose": "Fetch raw paper list from upstream API",
                    "needs_llm_reasoning": False,
                    "consumes_previous_orders": [],
                },
                {
                    "name": "summarise",
                    "order": 2,
                    "purpose": "Summarise the fetched papers into 5 bullet points",
                    "needs_llm_reasoning": True,
                    "consumes_previous_orders": [1],
                },
            ],
        }

        with patch(
            "flowforge.llm.caller.call_with_tool",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = fake_response
            plan = await plan_workflow(
                user_query="latest hf papers",
                suggested_flow_name="fetch_papers",
                suggested_flow_prompt="fetch trending papers",
                flow_summaries=["- alpha: alpha"],
                llm_config=LLMConfig(model="test"),
            )

        assert plan.flow_name == "fetch_papers"
        assert len(plan.steps) == 2
        assert plan.steps[0].needs_llm_reasoning is False
        assert plan.steps[1].needs_llm_reasoning is True
        # Generous max_tokens — synthesis budget should be passed through.
        kwargs = mock_call.call_args.kwargs
        assert kwargs["max_tokens"] >= 4096

    @pytest.mark.asyncio
    async def test_plan_workflow_uses_heuristic_when_llm_fails(self):
        with patch(
            "flowforge.llm.caller.call_with_tool",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.side_effect = TimeoutError("network timeout")
            plan = await plan_workflow(
                user_query="fetch latest arxiv papers before report pipeline",
                suggested_flow_name="arxiv_paper_fetcher",
                suggested_flow_prompt="Fetch arXiv API data for downstream use.",
                flow_summaries=["- paper_report_pipeline: consumes papers"],
                llm_config=LLMConfig(model="test"),
            )

        assert plan.flow_name == "arxiv_paper_fetcher"
        assert [step.name for step in plan.steps] == [
            "normalise_request",
            "fetch_source_data",
            "extract_records",
            "validate_payload",
        ]
        assert plan.steps[1].needs_llm_reasoning is False


# ---------------------------------------------------------------------------
# Phase 2 — capability selection
# ---------------------------------------------------------------------------


def _make_plan() -> WorkflowPlan:
    return WorkflowPlan(
        flow_name="clone_site",
        flow_prompt="clone a public site",
        task_name="main",
        task_prompt="clone the page",
        top_class="CloneSiteFlow",
        steps=[
            PlannedStep(
                name="fetch",
                order=1,
                purpose="Fetch raw HTML from the target URL",
                needs_llm_reasoning=False,
            ),
            PlannedStep(
                name="design",
                order=2,
                purpose="Design the layout structure for the clone",
                needs_llm_reasoning=True,
                consumes_previous_orders=[1],
            ),
        ],
    )


class TestSelectCapabilities:

    @pytest.mark.asyncio
    async def test_valid_selection_passes(self, tmp_path):
        plan = _make_plan()
        skill_dir = tmp_path / "skills" / "frontend-design"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: frontend-design\ndescription: design.\n---\n# d\n",
            encoding="utf-8",
        )
        tools = [
            FunctionTool(
                func=lambda url: {"ok": True, "url": url},
                name="web_fetch_url",
                description="HTTP GET",
            ),
            AgentSkill(
                path=str(skill_dir),
                name="frontend-design",
                description="Frontend design guidance",
            ),
        ]
        options = DynamicRunOptions(project_root=str(tmp_path))

        good = {
            "selections": [
                {
                    "step_name": "fetch",
                    "mode": "builtin_tool",
                    "tool_names": ["web_fetch_url"],
                    "rationale": "deterministic HTTP",
                },
                {
                    "step_name": "design",
                    "mode": "agent_skill",
                    "tool_names": ["frontend-design"],
                    "rationale": "design guidance",
                },
            ]
        }

        with patch(
            "flowforge.llm.caller.call_with_tool",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = good
            sel = await select_capabilities(
                plan=plan,
                tool_configs=tools,
                dynamic_options=options,
                llm_config=LLMConfig(model="test"),
            )

        by_name = sel.by_step_name()
        assert by_name["fetch"].mode == "builtin_tool"
        assert by_name["design"].mode == "agent_skill"

    @pytest.mark.asyncio
    async def test_rejects_unknown_tool_then_retries(self, tmp_path):
        plan = _make_plan()
        tool = FunctionTool(
            func=lambda url: {"ok": True, "url": url},
            name="web_fetch_url",
        )
        options = DynamicRunOptions(project_root=str(tmp_path))

        bad = {
            "selections": [
                {
                    "step_name": "fetch",
                    "mode": "builtin_tool",
                    "tool_names": ["hallucinated_tool"],
                    "rationale": "nope",
                },
                {
                    "step_name": "design",
                    "mode": "llm_only",
                    "tool_names": [],
                    "rationale": "pure reasoning",
                },
            ]
        }
        good = {
            "selections": [
                {
                    "step_name": "fetch",
                    "mode": "builtin_tool",
                    "tool_names": ["web_fetch_url"],
                    "rationale": "fixed",
                },
                {
                    "step_name": "design",
                    "mode": "llm_only",
                    "tool_names": [],
                    "rationale": "pure reasoning",
                },
            ]
        }

        with patch(
            "flowforge.llm.caller.call_with_tool",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.side_effect = [bad, good]
            sel = await select_capabilities(
                plan=plan,
                tool_configs=[tool],
                dynamic_options=options,
                llm_config=LLMConfig(model="test"),
            )

        assert mock_call.await_count == 2
        assert sel.by_step_name()["fetch"].tool_names == ["web_fetch_url"]

    @pytest.mark.asyncio
    async def test_select_capabilities_uses_heuristic_when_llm_fails(self, tmp_path):
        plan = _make_plan()
        tool = FunctionTool(
            func=lambda url: {"ok": True, "url": url},
            name="web_fetch_url",
        )
        options = DynamicRunOptions(project_root=str(tmp_path))

        with patch(
            "flowforge.llm.caller.call_with_tool",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.side_effect = TimeoutError("network timeout")
            sel = await select_capabilities(
                plan=plan,
                tool_configs=[tool],
                dynamic_options=options,
                llm_config=LLMConfig(model="test"),
            )

        by_name = sel.by_step_name()
        assert by_name["fetch"].mode == "builtin_tool"
        assert by_name["fetch"].tool_names == ["web_fetch_url"]
        assert by_name["design"].mode == "llm_only"


# ---------------------------------------------------------------------------
# Phase 3 — MCP provisioning + _artifact persistence
# ---------------------------------------------------------------------------


class TestProvisionMcp:

    def test_no_mcp_steps_returns_empty(self, tmp_path):
        plan = _make_plan()
        options = DynamicRunOptions(project_root=str(tmp_path))
        selection = CapabilitySelection(
            selections=[
                StepCapability(
                    step_name="fetch", mode="builtin_tool",
                    tool_names=["web_fetch_url"], rationale="reason ok",
                ),
                StepCapability(
                    step_name="design", mode="llm_only",
                    tool_names=[], rationale="reason ok",
                ),
            ]
        )
        records = provision_mcp_servers(
            selection=selection, plan=plan, options=options,
        )
        assert records == []

    def test_writes_artifact_and_manifest(self, tmp_path):
        plan = _make_plan()
        options = DynamicRunOptions(
            project_root=str(tmp_path),
            persist_generated=True,
            mcp_server_commands={"playwright": ["npx", "playwright-mcp"]},
            mcp_server_urls={"playwright": "http://localhost:9000"},
            mcp_server_tools={"playwright": ["pw_navigate", "pw_screenshot"]},
        )
        selection = CapabilitySelection(
            selections=[
                StepCapability(
                    step_name="fetch", mode="mcp",
                    mcp_server_name="playwright",
                    tool_names=["pw_navigate"],
                    rationale="needs browser",
                ),
                StepCapability(
                    step_name="design", mode="llm_only",
                    tool_names=[], rationale="pure reasoning",
                ),
            ]
        )
        records = provision_mcp_servers(
            selection=selection, plan=plan, options=options,
        )
        assert len(records) == 1
        assert records[0].server_name == "playwright"
        assert records[0].selected_tools == ["pw_navigate"]
        assert records[0].used_by_steps == ["fetch"]

        artifact_path = (
            Path(options.project_root)
            / options.generated_dir
            / "_artifact"
            / "mcp"
            / "playwright.json"
        )
        assert artifact_path.is_file()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["server_name"] == "playwright"
        assert payload["created_by_flow"] == "clone_site"

        manifest = load_manifest(options)
        assert any(
            getattr(entry, "server_name", "") == "playwright"
            or entry.get("server_name") == "playwright"  # type: ignore[union-attr]
            for entry in manifest.mcp_servers
        )


# ---------------------------------------------------------------------------
# Phase 4 — code synthesis from plan + capability
# ---------------------------------------------------------------------------


class TestSynthesisFromPlan:

    @pytest.mark.asyncio
    async def test_generate_flow_code_from_plan_uses_synthesis_prompt(self):
        plan = _make_plan()
        selection = CapabilitySelection(
            selections=[
                StepCapability(
                    step_name="fetch", mode="builtin_tool",
                    tool_names=["web_fetch_url"], rationale="reason ok",
                ),
                StepCapability(
                    step_name="design", mode="llm_only",
                    tool_names=[], rationale="pure reasoning",
                ),
            ]
        )
        tool = FunctionTool(
            func=lambda url: {"ok": True, "url": url},
            name="web_fetch_url",
        )
        gen = DynamicFlowGenerator(
            llm_config=LLMConfig(model="test", max_tokens=4096),
            dag=FlowForge.compile(_Agent).dag,
            tool_configs=[tool],
        )

        with patch(
            "flowforge.execution.llm.call_llm_api",
            new_callable=AsyncMock,
        ) as mock_api:
            mock_api.return_value = "from flowforge import flow, task, step\n"
            await gen.generate_flow_code_from_plan(
                plan=plan, selection=selection,
            )

        kwargs = mock_api.call_args.kwargs
        # Generous token budget for synthesis (≥12000).
        assert kwargs["llm_config"].max_tokens >= 12000
        system = kwargs["system_prompt"]
        assert "code synthesiser" in system
        user = kwargs["user_prompt"]
        assert "Workflow plan" in user
        assert "Capability decision" in user
        assert "fetch" in user and "design" in user

    @pytest.mark.asyncio
    async def test_generate_and_compile_from_plan_runs_validators(self):
        plan = _make_plan()
        selection = CapabilitySelection(
            selections=[
                StepCapability(
                    step_name="fetch", mode="builtin_tool",
                    tool_names=["web_fetch_url"], rationale="reason ok",
                ),
                StepCapability(
                    step_name="design", mode="llm_only",
                    tool_names=[], rationale="pure reasoning",
                ),
            ]
        )
        tool = FunctionTool(
            func=lambda url: {"ok": True, "url": url},
            name="web_fetch_url",
        )
        gen = DynamicFlowGenerator(
            llm_config=LLMConfig(model="test"),
            dag=FlowForge.compile(_Agent).dag,
            tool_configs=[tool],
        )

        good_code = '''
from flowforge import flow, task, step

@flow(name="clone_site", prompt="clone a public site",
      tools=["web_fetch_url"])
class CloneSiteFlow:
    @task(name="main", prompt="clone the page",
          tools=["web_fetch_url"])
    class MainTask:
        @step(order=1, prompt="Fetch the URL via web_fetch_url",
              tools=["web_fetch_url"])
        async def fetch(ctx):
            return await ctx.call_tool(
                "web_fetch_url", url="https://example.com",
            )

        @step(order=2, prompt="Design the layout based on fetched HTML")
        async def design(ctx):
            data = ctx.previous_results.get(1)
            return await ctx.call_llm(
                f"Design a layout structure given this html: {data}"
            )
'''

        with patch(
            "flowforge.execution.llm.call_llm_api",
            new_callable=AsyncMock,
        ) as mock_api:
            mock_api.return_value = good_code
            meta, code = await gen.generate_and_compile_from_plan(
                plan=plan, selection=selection,
                user_query="clone https://example.com",
            )

        assert meta.name == "clone_site"
        assert "web_fetch_url" in code


class TestGeneratedCodeQuality:

    def _generator(self) -> DynamicFlowGenerator:
        return DynamicFlowGenerator(
            llm_config=LLMConfig(model="test"),
            dag=FlowForge.compile(_Agent).dag,
        )

    def test_rejects_late_ctx_input_project_dir_reads(self):
        code = '''
from flowforge import flow, task, step

@flow(name="clone_site", prompt="clone")
class CloneSiteFlow:
    @task(name="main", prompt="main")
    class Main:
        @step(order=1, prompt="produce content")
        async def produce(ctx):
            return {"html": "<html>" + ("x" * 300) + "</html>"}

        @step(order=2, prompt="write html", tools=["files_write_text"])
        async def write_html(ctx):
            project_dir = ctx.input.get("project_dir")
            content = ctx.previous_results.get(1).get("html", "")
            if len(content) < 200:
                raise RuntimeError("trivial html")
            result = await ctx.call_tool(
                "files_write_text",
                path=f"{project_dir}/index.html",
                content=content,
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "write failed"))
            return {"ok": True}
'''
        err = self._generator().check_generated_code_quality(code, {})

        assert err is not None
        assert "ctx.input" in err
        assert "ctx.task_input" in err

    def test_rejects_file_map_prompt_pattern(self):
        code = '''
from flowforge import flow, task, step

@flow(name="clone_site", prompt="clone")
class CloneSiteFlow:
    @task(name="main", prompt="main")
    class Main:
        @step(order=1, prompt="generate files")
        async def generate(ctx):
            return await ctx.call_llm(
                "Return a JSON object where each key is a relative file path "
                "and each value is the complete file content."
            )
'''
        err = self._generator().check_generated_code_quality(code, {})

        assert err is not None
        assert "file map" in err

    def test_rejects_fallback_stub_file_content(self):
        code = '''
from flowforge import flow, task, step

@flow(name="clone_site", prompt="clone")
class CloneSiteFlow:
    @task(name="main", prompt="main")
    class Main:
        @step(order=1, prompt="write html", tools=["files_write_text"])
        async def write_html(ctx):
            html_content = ""
            if not html_content:
                html_content = "<html><body><script src='main.js'></script></body></html>"
            result = await ctx.call_tool(
                "files_write_text",
                path="site/index.html",
                content=html_content,
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "write failed"))
            return {"ok": True}
'''
        err = self._generator().check_generated_code_quality(code, {})

        assert err is not None
        assert "fallback content" in err

    def test_accepts_guarded_write_and_readback(self):
        code = '''
from flowforge import flow, task, step

@flow(name="clone_site", prompt="clone")
class CloneSiteFlow:
    @task(name="main", prompt="main")
    class Main:
        @step(order=1, prompt="normalise request")
        async def init(ctx):
            raw = ctx.input
            params = raw if isinstance(raw, dict) else {}
            ctx.shared_data["project_dir"] = params.get("project_dir", "site")
            ctx.shared_data["html"] = "<html><body>" + ("x" * 300) + "</body></html>"
            return {"ok": True}

        @step(order=2, prompt="write html", tools=["files_write_text"])
        async def write_html(ctx):
            project_dir = ctx.shared_data.get("project_dir", "site")
            html_content = ctx.shared_data.get("html", "")
            if len(html_content) < 200:
                raise RuntimeError("trivial html")
            result = await ctx.call_tool(
                "files_write_text",
                path=f"{project_dir}/index.html",
                content=html_content,
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "write failed"))
            return {"ok": True, "path": f"{project_dir}/index.html"}

        @step(order=3, prompt="verify html", tools=["files_read_text"])
        async def verify(ctx):
            project_dir = ctx.shared_data.get("project_dir", "site")
            result = await ctx.call_tool(
                "files_read_text",
                path=f"{project_dir}/index.html",
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "read failed"))
            content = result.get("content", "")
            if len(content) < 200 or "<body></body>" in content:
                raise RuntimeError("html appears trivial")
            return {"ok": True}
'''
        err = self._generator().check_generated_code_quality(
            code,
            {"required_tools": ["files_write_text", "files_read_text"]},
        )

        assert err is None
