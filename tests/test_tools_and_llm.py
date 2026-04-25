"""Tests for hierarchical tool inheritance and StepContext.call_llm().

Covers:
- Tool parameter on @flow, @task, @step decorators
- Hierarchical tool merging: global → flow → task → step
- Prompt templating: {var} replacement
- Tool reference parsing: <tool_name> extraction
- call_llm integration (mocked LLM)
"""
import pytest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel

from flowforge import global_config, flow, task, step, FlowForge, BranchCondition
from flowforge.execution.context import GlobalContext, FlowContext, TaskContext, StepContext
from flowforge.execution.runner import StepRunner
from flowforge.execution.llm import render_prompt, parse_tool_refs
from flowforge.tools.registry import ToolRegistry
from flowforge.types import (
    LLMConfig,
    MCPServer,
    ClaudeSkill,
    AgentSkill,
    FunctionTool,
    HTTPTool,
    DynamicRunOptions,
    DependencyPolicy,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryInput(BaseModel):
    query: str
    language: str = "ko"


class QueryResult(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    def test_basic_substitution_pydantic(self):
        inp = QueryInput(query="hello world", language="en")
        result = render_prompt("Search for {query} in {language}", inp)
        assert result == "Search for hello world in en"

    def test_basic_substitution_dict(self):
        inp = {"query": "hello", "count": 5}
        result = render_prompt("Find {query} with {count} results", inp)
        assert result == "Find hello with 5 results"

    def test_missing_field_left_as_is(self):
        inp = {"query": "test"}
        result = render_prompt("{query} and {unknown}", inp)
        assert result == "test and {unknown}"

    def test_none_input(self):
        result = render_prompt("no substitution {here}", None)
        assert result == "no substitution {here}"

    def test_empty_template(self):
        inp = {"x": 1}
        result = render_prompt("", inp)
        assert result == ""

    def test_triple_quoted_dedent(self):
        """Triple-quoted strings are auto-dedented."""
        inp = QueryInput(query="AI trends", language="ko")
        result = render_prompt("""
            Search for {query} in {language}.

            Return the top results.
        """, inp)
        assert result == "Search for AI trends in ko.\n\nReturn the top results."

    def test_tool_refs_allow_hyphenated_agent_skill_names(self):
        prompt, tools = parse_tool_refs("Roll with <roll-dice> and <web_search>.")

        assert prompt == "Roll with and ."
        assert tools == ["roll-dice", "web_search"]

    def test_triple_quoted_with_tools(self):
        """Triple-quoted strings with <tool> refs and {vars} work together."""
        inp = {"url": "https://example.com", "goal": "summarize"}
        result = render_prompt("""
            Navigate to {url}.
            Goal: {goal}

            Use <browser_navigate> to open the page.
        """, inp)
        assert "https://example.com" in result
        assert "summarize" in result
        # Indentation is gone
        assert not result.startswith(" ")
        assert not result.startswith("\n")

    def test_single_line_unchanged(self):
        """Single-line strings pass through without issue."""
        inp = {"name": "Alice"}
        result = render_prompt("Hello {name}!", inp)
        assert result == "Hello Alice!"


# ---------------------------------------------------------------------------
# parse_tool_refs
# ---------------------------------------------------------------------------

class TestParseToolRefs:
    def test_single_tool(self):
        cleaned, tools = parse_tool_refs("Use <web_search> to find results")
        assert tools == ["web_search"]
        assert "<web_search>" not in cleaned
        assert "web_search" not in cleaned

    def test_multiple_tools(self):
        cleaned, tools = parse_tool_refs("Use <search> and <translate> tools")
        assert tools == ["search", "translate"]
        assert "<search>" not in cleaned
        assert "<translate>" not in cleaned

    def test_duplicate_tool_deduplicated(self):
        cleaned, tools = parse_tool_refs("Call <api> then <api> again")
        assert tools == ["api"]

    def test_no_tools(self):
        cleaned, tools = parse_tool_refs("Plain prompt with no tool refs")
        assert tools == []
        assert cleaned == "Plain prompt with no tool refs"

    def test_tool_at_beginning(self):
        cleaned, tools = parse_tool_refs("<search> for results")
        assert tools == ["search"]
        assert cleaned.startswith("for results") or cleaned == "for results"

    def test_tool_at_end(self):
        cleaned, tools = parse_tool_refs("find results using <search>")
        assert tools == ["search"]


# ---------------------------------------------------------------------------
# Tool declaration on decorators
# ---------------------------------------------------------------------------

class TestToolDeclaration:
    def test_step_with_tools(self):
        mcp = MCPServer(url="http://test", name="test_mcp")

        @step(order=1, prompt="test", tools=[mcp])
        async def my_step(ctx):
            pass

        meta = my_step.__flowforge_step_meta__
        assert len(meta.tools) == 1
        assert meta.tools[0].name == "test_mcp"

    def test_step_with_claude_skill(self):
        skill = ClaudeSkill(name="pptx")

        @step(order=1, prompt="create presentation", tools=[skill])
        async def my_step(ctx):
            pass

        meta = my_step.__flowforge_step_meta__
        assert len(meta.tools) == 1
        assert meta.tools[0].name == "pptx"
        assert meta.tools[0].skill_id == "pptx"

    def test_task_with_tools(self):
        http = HTTPTool(url="http://api", name="api_tool")

        @task(name="t", prompt="test", tools=[http])
        class MyTask:
            @step(order=1, prompt="s")
            async def s1(ctx):
                pass

        meta = MyTask.__flowforge_task_meta__
        assert len(meta.tools) == 1
        assert meta.tools[0].name == "api_tool"

    def test_flow_with_tools(self):
        mcp = MCPServer(url="http://test", name="flow_mcp")

        @flow(name="f", prompt="test", tools=[mcp])
        class MyFlow:
            @task(name="t", prompt="t")
            class MyTask:
                @step(order=1, prompt="s")
                async def s1(ctx):
                    pass

        meta = MyFlow.__flowforge_flow_meta__
        assert len(meta.tools) == 1
        assert meta.tools[0].name == "flow_mcp"

    def test_no_tools_default_empty(self):
        @step(order=1, prompt="test")
        async def my_step(ctx):
            pass

        meta = my_step.__flowforge_step_meta__
        assert meta.tools == []


# ---------------------------------------------------------------------------
# Hierarchical tool merging in StepContext
# ---------------------------------------------------------------------------

class TestToolMerging:
    def test_merged_tools_all_levels(self):
        """Tools from global, flow, task, and step are all available."""
        global_mcp = MCPServer(url="http://global", name="global_tool")
        flow_mcp = MCPServer(url="http://flow", name="flow_tool")
        task_http = HTTPTool(url="http://task", name="task_tool")
        step_mcp = MCPServer(url="http://step", name="step_tool")

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
            global_tools=[global_mcp],
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx,
            flow_name="f",
            flow_prompt="test",
            flow_tools=[flow_mcp],
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx,
            task_name="t",
            task_prompt="test",
            task_tools=[task_http],
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="test",
            step_tools=[step_mcp],
        )

        merged = step_ctx.merged_tools
        names = [t.name for t in merged]
        assert names == ["global_tool", "flow_tool", "task_tool", "step_tool"]

    def test_merged_tools_empty_when_none(self):
        """No tools at any level gives an empty list."""
        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx,
            flow_name="f",
            flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx,
            task_name="t",
            task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="test",
        )

        assert step_ctx.merged_tools == []

    def test_flow_tools_available_in_child_step(self):
        """Tools on @flow are accessible from a step inside that flow."""
        flow_tool = MCPServer(url="http://flow", name="shared_tool")

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx,
            flow_name="f",
            flow_prompt="test",
            flow_tools=[flow_tool],
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx,
            task_name="t",
            task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="test",
        )

        merged = step_ctx.merged_tools
        assert len(merged) == 1
        assert merged[0].name == "shared_tool"

    def test_step_results_alias_exposes_task_results(self):
        """StepContext.step_results aliases the enclosing task accumulator."""
        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx,
            flow_name="f",
            flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx,
            task_name="t",
            task_prompt="test",
        )
        task_ctx.step_results[1] = {"ok": True}

        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="test",
        )

        assert step_ctx.step_results is task_ctx.step_results
        assert step_ctx.step_results[1] == {"ok": True}


# ---------------------------------------------------------------------------
# Tool reference resolution
# ---------------------------------------------------------------------------

class TestToolResolution:
    def test_resolve_tool_by_name(self):
        tool_a = MCPServer(url="http://a", name="search")
        tool_b = HTTPTool(url="http://b", name="translate")

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
            global_tools=[tool_a, tool_b],
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx, step_prompt="test",
        )

        resolved = step_ctx._resolve_tool_configs(["search"])
        assert len(resolved) == 1
        assert resolved[0].name == "search"

    def test_resolve_claude_skill_by_name(self):
        skill = ClaudeSkill(name="pptx")

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
            global_tools=[skill],
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx, step_prompt="test",
        )

        resolved = step_ctx._resolve_tool_configs(["pptx"])
        assert resolved == [skill]

    def test_resolve_unknown_tool_skipped(self):
        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx, step_prompt="test",
        )

        resolved = step_ctx._resolve_tool_configs(["nonexistent"])
        assert resolved == []


# ---------------------------------------------------------------------------
# Full compile with tools at every level
# ---------------------------------------------------------------------------

class TestCompileWithTools:
    def test_compile_with_tools_at_all_levels(self):
        """Agent with tools at global, flow, task, step all compiles."""
        global_tool = MCPServer(url="http://g", name="global_search")
        flow_tool = MCPServer(url="http://f", name="flow_db")
        task_tool = HTTPTool(url="http://t", name="task_api")
        step_tool = MCPServer(url="http://s", name="step_cache")

        @global_config(prompt="agent", tools=[global_tool])
        class ToolAgent:
            @flow(name="pipeline", prompt="pipeline", tools=[flow_tool])
            class Pipeline:
                @task(name="process", prompt="process", tools=[task_tool])
                class Process:
                    @step(order=1, prompt="execute", tools=[step_tool])
                    async def execute(ctx):
                        return {"done": True}

        engine = FlowForge.compile(ToolAgent)
        assert engine.dag is not None

    @pytest.mark.asyncio
    async def test_run_with_tools_passed_to_context(self):
        """At runtime, tools flow through the context hierarchy."""
        captured_tools = {}

        flow_tool = MCPServer(url="http://flow", name="flow_t")
        task_tool = MCPServer(url="http://task", name="task_t")

        @global_config(prompt="agent")
        class ToolRunAgent:
            @flow(name="f", prompt="f", tools=[flow_tool])
            class F:
                @task(name="t", prompt="t", tools=[task_tool])
                class T:
                    @step(order=1, prompt="s")
                    async def s1(ctx):
                        captured_tools["merged"] = [
                            t.name for t in ctx.merged_tools
                        ]
                        return "ok"

        engine = FlowForge.compile(ToolRunAgent)
        await engine.run("test")

        assert "flow_t" in captured_tools["merged"]
        assert "task_t" in captured_tools["merged"]


# ---------------------------------------------------------------------------
# call_llm (mocked)
# ---------------------------------------------------------------------------

class TestCallLLM:
    @pytest.mark.asyncio
    async def test_call_llm_renders_prompt_and_calls_api(self):
        """call_llm templates {var}, strips <tool>, and calls the LLM."""
        search_tool = MCPServer(url="http://s", name="web_search", description="Search the web")

        global_ctx = GlobalContext(
            llm_config=LLMConfig(provider="anthropic", model="test-model"),
            global_prompt="system",
            tool_registry=ToolRegistry(),
            global_tools=[search_tool],
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="You are a search assistant",
            step_input=QueryInput(query="AI trends", language="ko"),
        )

        with patch("flowforge.execution.llm.call_llm_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "AI trends are growing"

            result = await step_ctx.call_llm(
                "Search for {query} in {language} using <web_search>"
            )

            assert result == "AI trends are growing"
            mock_call.assert_called_once()

            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["system_prompt"] == "You are a search assistant"
            assert "AI trends" in call_kwargs["user_prompt"]
            assert "ko" in call_kwargs["user_prompt"]
            assert "<web_search>" not in call_kwargs["user_prompt"]
            assert len(call_kwargs["tool_configs"]) == 1
            assert call_kwargs["tool_configs"][0].name == "web_search"

    @pytest.mark.asyncio
    async def test_call_llm_passes_claude_skill_from_angle_brackets(self):
        """ClaudeSkill uses the same <name> syntax as regular tools."""
        pptx_skill = ClaudeSkill(name="pptx")

        global_ctx = GlobalContext(
            llm_config=LLMConfig(provider="anthropic", model="test-model"),
            global_prompt="system",
            tool_registry=ToolRegistry(),
            global_tools=[pptx_skill],
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="You create decks",
            step_input={"topic": "solar energy"},
        )

        with patch("flowforge.execution.llm.call_llm_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "created"

            result = await step_ctx.call_llm("Create a deck about {topic}. <pptx>")

            assert result == "created"
            call_kwargs = mock_call.call_args.kwargs
            assert "<pptx>" not in call_kwargs["user_prompt"]
            assert call_kwargs["tool_configs"] == [pptx_skill]

    @pytest.mark.asyncio
    async def test_call_llm_passes_agent_skill_from_hyphenated_angle_brackets(self, tmp_path):
        """AgentSkill supports the standard hyphenated <skill-name> syntax."""
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: code-review
description: Review code.
---

Review carefully.
""",
            encoding="utf-8",
        )
        agent_skill = AgentSkill(path=str(skill_dir))

        global_ctx = GlobalContext(
            llm_config=LLMConfig(provider="openai", model="test-model"),
            global_prompt="system",
            tool_registry=ToolRegistry(),
            global_tools=[agent_skill],
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="You review code",
            step_input={"diff": "patch"},
        )

        with patch("flowforge.execution.llm.call_llm_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "reviewed"

            result = await step_ctx.call_llm("Review this {diff}. <code-review>")

            assert result == "reviewed"
            call_kwargs = mock_call.call_args.kwargs
            assert "<code-review>" not in call_kwargs["user_prompt"]
            assert call_kwargs["tool_configs"] == [agent_skill]

    @pytest.mark.asyncio
    async def test_call_llm_no_tools(self):
        """call_llm works without any tool references."""
        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="sys",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="helper",
            step_input={"name": "Alice"},
        )

        with patch("flowforge.execution.llm.call_llm_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Hello Alice"

            result = await step_ctx.call_llm("Greet {name}")

            assert result == "Hello Alice"
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["user_prompt"] == "Greet Alice"
            assert call_kwargs["tool_configs"] == []

    @pytest.mark.asyncio
    async def test_call_llm_code_step_no_call(self):
        """A pure code step does not call call_llm — just returns computed data."""
        @step(order=1, prompt="filter results")
        async def filter_step(ctx):
            # Pure code step — no ctx.call_llm()
            data = ctx.input
            return {"filtered": [x for x in data["items"] if x > 0]}

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(
            global_ctx=global_ctx, flow_name="f", flow_prompt="test",
        )
        task_ctx = TaskContext(
            flow_ctx=flow_ctx, task_name="t", task_prompt="test",
        )
        step_ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt="filter results",
            step_input={"items": [-1, 2, -3, 4]},
        )

        result = await filter_step(step_ctx)
        assert result == {"filtered": [2, 4]}


# ---------------------------------------------------------------------------
# End-to-end: AI step vs Code step in same task
# ---------------------------------------------------------------------------

class TestAIVsCodeStep:
    @pytest.mark.asyncio
    async def test_mixed_steps(self):
        """Task with a code step followed by an AI step (mocked)."""
        @global_config(prompt="agent")
        class MixedAgent:
            @flow(name="f", prompt="f")
            class F:
                @task(name="t", prompt="t")
                class T:
                    @step(order=1, prompt="validate input")
                    async def validate(ctx):
                        # Code step — no AI
                        data = ctx.input
                        if isinstance(data, dict):
                            return {"validated": True, "query": data.get("q", "")}
                        return {"validated": True, "query": str(data)}

                    @step(order=2, prompt="answer the query")
                    async def answer(ctx):
                        # Would call ctx.call_llm() in production;
                        # here we simulate it.
                        return {"answer": f"Response to: {ctx.input['query']}"}

        engine = FlowForge.compile(MixedAgent)
        result = await engine.run({"q": "what is AI?"})
        assert result["answer"] == "Response to: what is AI?"


# ═══════════════════════════════════════════════════════════════════════
# Structured output: output_schema forces tool_use for LLM responses
# ═══════════════════════════════════════════════════════════════════════

class StructuredResult(BaseModel):
    title: str
    score: float = 0.0


@global_config(prompt="structured output agent", llm_config=LLMConfig(model="test"))
class StructuredOutputAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="analyze this", output_schema=StructuredResult)
            async def analyze(ctx):
                result = await ctx.call_llm("do the analysis on {value}")
                return result


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_output_schema_passed_to_call_llm(self):
        """When step has output_schema, call_llm passes it to call_llm_api."""
        with patch("flowforge.execution.llm.call_llm_api", new_callable=AsyncMock) as mock_api:
            # Simulate the LLM returning a dict (as it would via tool_use).
            mock_api.return_value = {"title": "Test Result", "score": 0.95}

            engine = FlowForge.compile(StructuredOutputAgent)
            result = await engine.run({"value": "test data"})

            # Verify call_llm_api was called with output_schema.
            mock_api.assert_called_once()
            call_kwargs = mock_api.call_args
            assert call_kwargs.kwargs["output_schema"] is StructuredResult

            # Verify result was validated into the schema.
            assert result.title == "Test Result"
            assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_output_schema_none_when_not_set(self):
        """When step has no output_schema, call_llm passes output_schema=None."""
        @global_config(prompt="no schema agent", llm_config=LLMConfig(model="test"))
        class NoSchemaAgent:
            @flow(name="f", prompt="f")
            class F:
                @task(name="t", prompt="t")
                class T:
                    @step(order=1, prompt="do work")
                    async def work(ctx):
                        result = await ctx.call_llm("hello")
                        return result

        with patch("flowforge.execution.llm.call_llm_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "plain text response"

            engine = FlowForge.compile(NoSchemaAgent)
            await engine.run(None)

            mock_api.assert_called_once()
            assert mock_api.call_args.kwargs["output_schema"] is None


# ═══════════════════════════════════════════════════════════════════════
# Tool-use loop: ToolExecutor dispatches tool calls
# ═══════════════════════════════════════════════════════════════════════

class TestToolExecutor:
    """Unit tests for ToolExecutor dispatch logic."""

    def test_has_tool(self):
        from flowforge.execution.tool_executor import ToolExecutor

        mcp = MCPServer(url="http://localhost/mcp", name="browser_navigate")
        func = FunctionTool(func=lambda x: x, name="my_func")
        executor = ToolExecutor([mcp, func])

        assert executor.has_tool("browser_navigate")
        assert executor.has_tool("my_func")
        assert not executor.has_tool("unknown")

    @pytest.mark.asyncio
    async def test_execute_function_tool_async(self):
        from flowforge.execution.tool_executor import ToolExecutor

        async def greet(name: str = "world") -> str:
            return f"Hello, {name}!"

        ft = FunctionTool(func=greet, name="greet")
        executor = ToolExecutor([ft])
        result = await executor.execute("greet", {"name": "Alice"})
        assert result == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_execute_function_tool_sync(self):
        from flowforge.execution.tool_executor import ToolExecutor

        def add(a: int = 0, b: int = 0) -> dict:
            return {"sum": a + b}

        ft = FunctionTool(func=add, name="add")
        executor = ToolExecutor([ft])
        result = await executor.execute("add", {"a": 3, "b": 4})
        assert '"sum": 7' in result

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        from flowforge.execution.tool_executor import ToolExecutor

        executor = ToolExecutor([])
        result = await executor.execute("nonexistent", {})
        assert "unknown tool" in result.lower()


class TestToolUseLoopIntegration:
    """Integration tests verifying the tool-use loop in _call_anthropic."""

    @pytest.mark.asyncio
    async def test_build_executor_with_no_tools(self):
        """_build_executor returns (None, {}) when no tools provided."""
        from flowforge.execution.llm import _build_executor

        executor, schemas = await _build_executor(None)
        assert executor is None
        assert schemas == {}

        executor, schemas = await _build_executor([])
        assert executor is None
        assert schemas == {}

    @pytest.mark.asyncio
    async def test_build_executor_with_function_tools(self):
        """_build_executor creates executor for FunctionTool (no MCP fetch)."""
        from flowforge.execution.llm import _build_executor

        ft = FunctionTool(func=lambda: "ok", name="my_tool")
        executor, schemas = await _build_executor([ft])
        assert executor is not None
        assert executor.has_tool("my_tool")
        assert schemas == {}  # No MCP → no schemas fetched

    def test_claude_skill_request_helpers(self):
        from flowforge.execution.llm import (
            _add_anthropic_skill_kwargs,
            _claude_skill_to_container_skill,
            _format_anthropic_text_response,
            _split_claude_skill_configs,
        )

        skill = ClaudeSkill(name="pptx")
        regular = FunctionTool(func=lambda: "ok", name="local_tool")

        regular_tools, skills = _split_claude_skill_configs([regular, skill])
        assert regular_tools == [regular]
        assert skills == [skill]
        assert _claude_skill_to_container_skill(skill) == {
            "type": "anthropic",
            "skill_id": "pptx",
            "version": "latest",
        }

        kwargs = {"tools": [{"name": "local_tool", "input_schema": {}}]}
        _add_anthropic_skill_kwargs(kwargs, skills)

        assert kwargs["betas"] == [
            "code-execution-2025-08-25",
            "skills-2025-10-02",
        ]
        assert kwargs["container"]["skills"] == [
            {"type": "anthropic", "skill_id": "pptx", "version": "latest"}
        ]
        assert {"type": "code_execution_20250825", "name": "code_execution"} in kwargs["tools"]

        class TextBlock:
            text = "Deck created."

        class ToolResultBlock:
            def model_dump(self, exclude_unset=True):
                return {
                    "type": "bash_code_execution_tool_result",
                    "content": {
                        "content": [
                            {
                                "type": "bash_code_execution_output",
                                "file_id": "file_123",
                            }
                        ]
                    },
                }

        formatted = _format_anthropic_text_response([TextBlock(), ToolResultBlock()])
        assert "Deck created." in formatted
        assert "file_id: file_123" in formatted

    @pytest.mark.asyncio
    async def test_claude_skill_rejected_for_non_anthropic_provider(self):
        from flowforge.execution.llm import call_llm_api

        with pytest.raises(ValueError, match="ClaudeSkill"):
            await call_llm_api(
                system_prompt="system",
                user_prompt="make a deck",
                llm_config=LLMConfig(provider="openai", model="test"),
                tool_configs=[ClaudeSkill(name="pptx")],
            )


# ---------------------------------------------------------------------------
# FunctionTool schema auto-inference
# ---------------------------------------------------------------------------

class TestFunctionToolSchemaInference:
    """Test that FunctionToolAdapter infers input_schema from type hints."""

    def test_basic_types(self):
        def search(query: str, limit: int = 10, verbose: bool = False) -> str:
            return ""

        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(search)
        schema = adapter.schema
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["limit"]["default"] == 10
        assert schema["properties"]["verbose"]["type"] == "boolean"
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_float_type(self):
        def compute(ratio: float) -> float:
            return ratio

        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(compute)
        assert adapter.schema["properties"]["ratio"]["type"] == "number"
        assert "ratio" in adapter.schema["required"]

    def test_list_type(self):
        def process(items: list[str]) -> str:
            return ""

        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(process)
        prop = adapter.schema["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_optional_type(self):
        def fetch(url: str, timeout: int | None = None) -> str:
            return ""

        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(fetch)
        assert adapter.schema["properties"]["url"]["type"] == "string"
        assert "url" in adapter.schema["required"]
        # timeout has a default → not required
        assert "timeout" not in adapter.schema["required"]

    def test_ctx_param_skipped(self):
        async def my_step(ctx, query: str) -> str:
            return ""

        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(my_step)
        assert "ctx" not in adapter.schema["properties"]
        assert "query" in adapter.schema["properties"]

    def test_explicit_schema_overrides_inference(self):
        def my_func(query: str) -> str:
            return ""

        custom = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(my_func, schema=custom)
        assert adapter.schema == custom

    def test_no_hints_produces_empty_schema(self):
        def bare_func():
            return ""

        from flowforge.tools.function_tool import FunctionToolAdapter
        adapter = FunctionToolAdapter(bare_func)
        assert adapter.schema["properties"] == {}
        assert adapter.schema["required"] == []

    def test_llm_tool_config_uses_inferred_schema(self):
        """_tool_config_to_anthropic uses inferred schema for FunctionTool."""
        from flowforge.execution.llm import _tool_config_to_anthropic

        def search(query: str, limit: int = 5) -> str:
            """Search the web."""
            return ""

        ft = FunctionTool(func=search, name="search")
        result = _tool_config_to_anthropic(ft)
        assert result["name"] == "search"
        assert result["input_schema"]["properties"]["query"]["type"] == "string"
        assert result["input_schema"]["properties"]["limit"]["type"] == "integer"
        assert "query" in result["input_schema"]["required"]


class TestBuiltinShellTools:
    def test_builtin_pack_includes_codegen_support_tools(self, tmp_path):
        from flowforge.tools.builtin import create_builtin_tool_pack

        options = DynamicRunOptions(project_root=str(tmp_path))
        names = [tool.name for tool in create_builtin_tool_pack(options)]

        assert "python_import_check" in names
        assert "mcp_start_server" in names

    def test_readonly_shell_tool_runs_allowlisted_command(self, tmp_path):
        from flowforge.tools.builtin import create_builtin_tool_pack

        options = DynamicRunOptions(project_root=str(tmp_path))
        tool = next(
            item for item in create_builtin_tool_pack(options)
            if item.name == "shell_readonly"
        )

        result = tool.func("pwd")
        assert result["ok"] is True
        assert result["cwd"] == str(tmp_path)

    def test_readonly_shell_tool_truncates_large_output(self, tmp_path):
        from flowforge.tools.builtin import create_builtin_tool_pack

        (tmp_path / "large.txt").write_text("x" * 2000)
        options = DynamicRunOptions(
            project_root=str(tmp_path),
            shell_output_max_chars=500,
        )
        tool = next(
            item for item in create_builtin_tool_pack(options)
            if item.name == "shell_readonly"
        )

        result = tool.func("cat large.txt")
        assert result["ok"] is True
        assert result["truncated"] is True
        assert len(result["stdout"]) < 600

    def test_python_import_check_reports_available_module(self, tmp_path):
        from flowforge.tools.builtin import create_builtin_tool_pack

        options = DynamicRunOptions(project_root=str(tmp_path))
        tool = next(
            item for item in create_builtin_tool_pack(options)
            if item.name == "python_import_check"
        )

        result = tool.func("json")
        assert result["ok"] is True
        assert result["results"][0]["module"] == "json"
        assert result["results"][0]["available"] is True

    def test_mcp_start_server_requires_declared_command(self, tmp_path):
        from flowforge.tools.builtin import create_builtin_tool_pack

        options = DynamicRunOptions(project_root=str(tmp_path))
        tool = next(
            item for item in create_builtin_tool_pack(options)
            if item.name == "mcp_start_server"
        )

        result = tool.func("browser")
        assert result["ok"] is False
        assert "not declared" in result["stderr"]

    def test_install_tool_respects_dependency_policy(self, tmp_path):
        from flowforge.tools.builtin import create_builtin_tool_pack

        options = DynamicRunOptions(
            project_root=str(tmp_path),
            allowed_shell_modes=["install_dependency"],
            dependency_policy=DependencyPolicy(allow_install=False),
        )
        tool = next(
            item for item in create_builtin_tool_pack(options)
            if item.name == "shell_install_dependency"
        )

        result = tool.func("pip install requests")
        assert result["ok"] is False
        assert "disabled" in result["stderr"]


# ---------------------------------------------------------------------------
# Builtin tools available in user-defined flows via include_builtin_tools
# ---------------------------------------------------------------------------

@task(name="user_bt_task", prompt="use builtin tools")
class _UserBuiltinTask:
    @step(order=1, prompt="read a file")
    async def use_builtin(ctx):
        return {"done": True}


@flow(name="user_bt_flow", prompt="user flow with builtins")
class _UserBuiltinFlow:
    _UserBuiltinTask = _UserBuiltinTask


@global_config(
    prompt="agent with builtin tools but no dynamic flow",
    include_builtin_tools=True,
)
class _UserBuiltinAgent:
    _UserBuiltinFlow = _UserBuiltinFlow


class TestIncludeBuiltinTools:
    """include_builtin_tools=True injects tools without dynamic_flow."""

    def test_builtin_tools_injected_without_dynamic_flow(self):
        engine = FlowForge.compile(_UserBuiltinAgent)
        tool_names = [t.name for t in engine._global_meta.tools
                      if isinstance(t, FunctionTool)]
        # Core builtin tools should be present
        assert "web_fetch_url" in tool_names
        assert "files_read_text" in tool_names
        assert "files_write_text" in tool_names
        assert "json_select_fields" in tool_names
        assert "csv_read" in tool_names
        assert "pptx_create" in tool_names
        assert "chart_create" in tool_names

    def test_builtin_tools_not_injected_by_default(self):
        @global_config(prompt="plain agent")
        class PlainAgent:
            _UserBuiltinFlow = _UserBuiltinFlow

        engine = FlowForge.compile(PlainAgent)
        tool_names = [t.name for t in engine._global_meta.tools
                      if isinstance(t, FunctionTool)]
        assert "web_fetch_url" not in tool_names

    def test_image_create_and_claude_skill_in_builtins(self):
        engine = FlowForge.compile(_UserBuiltinAgent)
        tool_names = [t.name for t in engine._global_meta.tools
                      if isinstance(t, FunctionTool)]
        assert "image_create" in tool_names
        assert "claude_skill" in tool_names


class TestClaudeSkillTool:
    """Tests for the claude_skill builtin tool."""

    def test_claude_cli_not_found(self):
        from flowforge.tools.builtin import _make_claude_skill_tool
        from unittest.mock import patch as _patch

        tool = _make_claude_skill_tool(DynamicRunOptions())

        with _patch("shutil.which", return_value=None):
            result = tool(skill_name="commit", prompt="test")
            assert result["ok"] is False
            assert "not found" in result["error"]
            assert result["skill"] == "commit"

    def test_claude_skill_success(self):
        from flowforge.tools.builtin import _make_claude_skill_tool
        from unittest.mock import patch as _patch, MagicMock
        import subprocess as sp

        tool = _make_claude_skill_tool(DynamicRunOptions())

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Commit created successfully"
        mock_result.stderr = ""

        with _patch("shutil.which", return_value="/usr/local/bin/claude"), \
             _patch("subprocess.run", return_value=mock_result) as mock_run:
            result = tool(skill_name="commit", prompt="fix bug")
            assert result["ok"] is True
            assert "Commit created" in result["result"]
            assert result["skill"] == "commit"
            # Verify the CLI was called with correct args
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "/usr/local/bin/claude" in cmd
            assert "--prompt" in cmd


class TestAgentSkillTool:
    """Tests for local Agent Skills loaded from SKILL.md."""

    def test_agent_skill_loads_skill_md_prompt(self, tmp_path):
        from flowforge.execution.llm import _inject_agent_skill_prompts

        skill_dir = tmp_path / "roll-dice"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: roll-dice
description: Roll dice when the user asks for random dice results.
---

Use this marker in every answer: AGENT_SKILL_MARKER.
""",
            encoding="utf-8",
        )

        skill = AgentSkill(path=str(skill_dir))
        prompt = _inject_agent_skill_prompts("system", [skill])

        assert skill.name == "roll-dice"
        assert "Activated Agent Skills" in prompt
        assert "name=\"roll-dice\"" in prompt
        assert "Roll dice when the user asks" in prompt
        assert "AGENT_SKILL_MARKER" in prompt

    @pytest.mark.asyncio
    async def test_agent_skill_injected_for_openai_provider(self, tmp_path):
        from flowforge.execution.llm import call_llm_api

        skill_dir = tmp_path / "proof-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: proof-skill
description: Prove local Agent Skill prompt injection.
---

Always include PROOF_SKILL_USED.
""",
            encoding="utf-8",
        )

        captured: dict[str, str] = {}

        async def fake_openai(system_prompt, user_prompt, config, tools, **kwargs):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["tools"] = str(tools)
            return "ok"

        with patch("flowforge.execution.llm._call_openai", side_effect=fake_openai):
            result = await call_llm_api(
                system_prompt="system",
                user_prompt="use the skill",
                llm_config=LLMConfig(provider="openai", model="test"),
                tool_configs=[AgentSkill(path=str(skill_dir))],
            )

        assert result == "ok"
        assert "PROOF_SKILL_USED" in captured["system_prompt"]
        assert captured["user_prompt"] == "use the skill"
        assert captured["tools"] == "[]"


class TestCallSkillContext:
    """Tests for StepContext.call_skill()."""

    @pytest.mark.asyncio
    async def test_call_skill_cli_not_found(self):
        """call_skill returns error when claude CLI is not in PATH."""
        from unittest.mock import patch as _patch

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(global_ctx=global_ctx, flow_name="f", flow_prompt="f")
        task_ctx = TaskContext(flow_ctx=flow_ctx, task_name="t", task_prompt="t")
        step_ctx = StepContext(task_ctx=task_ctx, step_prompt="s")

        with _patch("shutil.which", return_value=None):
            result = await step_ctx.call_skill("commit", "test commit")
            assert result["ok"] is False
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_call_skill_success(self):
        """call_skill returns result on successful CLI invocation."""
        from unittest.mock import patch as _patch, AsyncMock as _AM

        global_ctx = GlobalContext(
            llm_config=LLMConfig(),
            global_prompt="test",
            tool_registry=ToolRegistry(),
        )
        flow_ctx = FlowContext(global_ctx=global_ctx, flow_name="f", flow_prompt="f")
        task_ctx = TaskContext(flow_ctx=flow_ctx, task_name="t", task_prompt="t")
        step_ctx = StepContext(task_ctx=task_ctx, step_prompt="s")

        # Mock asyncio.create_subprocess_exec
        mock_proc = _AM()
        mock_proc.communicate = _AM(return_value=(
            b"Review complete: LGTM",
            b"",
        ))
        mock_proc.returncode = 0

        with _patch("shutil.which", return_value="/usr/bin/claude"), \
             _patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await step_ctx.call_skill("review-pr", "PR #42")
            assert result["ok"] is True
            assert "LGTM" in result["result"]
            assert result["skill"] == "review-pr"
