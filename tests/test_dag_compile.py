"""Tests for DAG compilation.

Covers:
- Basic agent compilation and node presence
- Node type correctness (GLOBAL, FLOW, TASK, STEP)
- Branch dispatcher nodes (FLOW, TASK, STEP with is_branch=True)
- Cycle detection
- Topological ordering
- The complex research-agent example from CLAUDE.md §13 (adapted to new API)
"""
import pytest
from pydantic import BaseModel

from flowforge import global_config, flow, task, step, FlowForge, LLMConfig, BranchCondition
from flowforge.schema.dag import NodeType
from flowforge.errors import CycleDetectedError, CompileError


# ---------------------------------------------------------------------------
# Minimal agent for basic compile tests
# ---------------------------------------------------------------------------

@task(name="fetch_task", prompt="fetch data")
class FetchTask:
    @step(order=1, prompt="prepare request")
    async def prepare(ctx): ...

    @step(order=2, prompt="execute request")
    async def execute(ctx): ...


@flow(name="data_flow", prompt="data pipeline")
class DataFlow:
    FetchTask = FetchTask


@global_config(prompt="test agent")
class SimpleAgent:
    DataFlow = DataFlow


# ---------------------------------------------------------------------------
# Basic compile tests
# ---------------------------------------------------------------------------

def test_compile_returns_compiled_agent():
    engine = FlowForge.compile(SimpleAgent)
    assert engine is not None


def test_dag_has_global_node():
    engine = FlowForge.compile(SimpleAgent)
    global_node = engine.dag.get_node("global")
    assert global_node is not None
    assert global_node.type == NodeType.GLOBAL


def test_dag_has_flow_node():
    engine = FlowForge.compile(SimpleAgent)
    flow_node = engine.dag.get_node("global.data_flow")
    assert flow_node is not None
    assert flow_node.type == NodeType.FLOW


def test_dag_has_task_node():
    engine = FlowForge.compile(SimpleAgent)
    task_node = engine.dag.get_node("global.data_flow.fetch_task")
    assert task_node is not None
    assert task_node.type == NodeType.TASK


def test_dag_has_step_nodes():
    engine = FlowForge.compile(SimpleAgent)
    all_steps = engine.dag.nodes_by_type(NodeType.STEP)
    assert len(all_steps) == 2


def test_compile_unknown_class_raises():
    class NotAnAgent:
        pass

    with pytest.raises(CompileError):
        FlowForge.compile(NotAnAgent)


def test_dag_parent_child_edges():
    engine = FlowForge.compile(SimpleAgent)
    dag = engine.dag

    flow_node = dag.get_node("global.data_flow")
    parents = dag.get_parents(flow_node.id)
    assert any(p.id == "global" for p in parents)


def test_mermaid_output():
    engine = FlowForge.compile(SimpleAgent)
    mermaid = engine.mermaid()
    assert "graph TD" in mermaid
    assert "global" in mermaid


# ---------------------------------------------------------------------------
# Complex research-agent from CLAUDE.md §13 — adapted to new API
#
# The @branch decorator is replaced with @step(condition=..., branches=...).
# ---------------------------------------------------------------------------

class UserQuery(BaseModel):
    query: str
    language: str = "ko"

class AnalyzedQuery(BaseModel):
    intent: str
    keywords: list[str]
    source_preference: str

class SearchResult(BaseModel):
    results: list[dict]
    source: str

class FormattedAnswer(BaseModel):
    answer: str
    citations: list[str]


# Branch handlers used by the branching @step below.
async def web_search_handler(ctx): ...
async def db_search_handler(ctx): ...
async def api_search_handler(ctx): ...


@global_config(
    prompt="다국어 리서치 어시스턴트.",
    llm_config=LLMConfig(model="claude-sonnet-4-6", temperature=0.3),
)
class ResearchAgent:

    @flow(
        name="research",
        prompt="유저 질문을 분석 → 최적 소스 검색 → 답변 생성",
        input_schema=UserQuery,
        output_schema=FormattedAnswer,
    )
    class ResearchFlow:

        @flow(
            name="search",
            prompt="분석된 쿼리 기반으로 적절한 소스에서 검색 수행",
            input_schema=AnalyzedQuery,
            output_schema=SearchResult,
        )
        class SearchSubFlow:
            @task(name="execute_search", prompt="소스별 검색 실행")
            class ExecuteSearchTask:
                @step(order=1, prompt="검색 쿼리 최적화")
                async def optimize_query(ctx): ...

                # This @step acts as a branch dispatcher — replaces the old
                # standalone @branch decorator.
                @step(
                    order=2,
                    prompt="쿼리 분석에 따라 검색 소스 선택",
                    condition=BranchCondition(
                        field="source_preference",
                        enum=["web", "db", "api"],
                    ),
                    branches={
                        "web": web_search_handler,
                        "db":  db_search_handler,
                        "api": api_search_handler,
                    },
                    fallback=web_search_handler,
                )
                async def route_source(ctx): ...

                @step(order=3, prompt="검색 결과 정제 및 중복 제거")
                async def deduplicate(ctx): ...

        @task(name="analyze_and_format", prompt="쿼리 분석 및 최종 답변 포맷팅")
        class AnalyzeAndFormatTask:
            @task(name="analyze", prompt="유저 쿼리의 의도와 키워드를 분석")
            class AnalyzeTask:
                @step(order=1, prompt="쿼리 의도 분류")
                async def classify_intent(ctx): ...

            @task(name="format", prompt="검색 결과를 최종 답변으로 포맷팅")
            class FormatTask:
                @step(order=1, prompt="답변 초안 생성")
                async def draft_answer(ctx): ...

                @step(order=2, prompt="출처 인용 추가")
                async def add_citations(ctx): ...


def test_complex_agent_compiles():
    engine = FlowForge.compile(ResearchAgent)
    dag = engine.dag
    assert dag.get_node("global") is not None
    assert dag.get_node("global.research") is not None
    assert dag.get_node("global.research.search") is not None
    assert dag.get_node("global.research.search.execute_search") is not None
    assert dag.get_node("global.research.analyze_and_format") is not None


def test_complex_agent_has_branching_step():
    """The route_source step is a STEP node with is_branch=True (no BRANCH type)."""
    engine = FlowForge.compile(ResearchAgent)
    dag    = engine.dag

    # All nodes are either GLOBAL, FLOW, TASK, or STEP — no BRANCH type.
    all_types = {n.type for n in dag.get_all_nodes()}
    assert NodeType.STEP in all_types
    assert "branch" not in {t.value for t in all_types}

    # The branching step must be present and have is_branch=True.
    all_steps    = dag.nodes_by_type(NodeType.STEP)
    branch_steps = [n for n in all_steps if n.meta.is_branch]
    assert len(branch_steps) == 1
    assert branch_steps[0].name == "route_source"


def test_complex_agent_branch_nodes_helper():
    """FlowForgeDAG.branch_nodes() returns all nodes with is_branch=True."""
    engine       = FlowForge.compile(ResearchAgent)
    branch_nodes = engine.dag.branch_nodes()
    assert len(branch_nodes) == 1
    assert branch_nodes[0].name == "route_source"


def test_complex_agent_no_cycles():
    engine = FlowForge.compile(ResearchAgent)
    cycles = engine.dag.detect_cycles()
    assert cycles == []


def test_topological_order():
    engine = FlowForge.compile(ResearchAgent)
    order  = engine.dag.topological_order()
    # The root GLOBAL node must come first.
    assert order[0].id == "global"


# ---------------------------------------------------------------------------
# Task-level branch dispatching in the DAG
# ---------------------------------------------------------------------------

@task(name="fast_process", prompt="fast processing")
class FastProcessTask:
    @step(order=1, prompt="fast step")
    async def fast(ctx): ...


@task(name="slow_process", prompt="slow processing")
class SlowProcessTask:
    @step(order=1, prompt="slow step")
    async def slow(ctx): ...


@task(
    name="dispatch_process",
    prompt="route to fast or slow processing",
    condition=BranchCondition(field="mode", enum=["fast", "slow"]),
    branches={"fast": FastProcessTask, "slow": SlowProcessTask},
    fallback=FastProcessTask,
)
class DispatchProcessTask: ...


@flow(name="process_flow", prompt="process flow")
class ProcessFlow:
    DispatchProcessTask = DispatchProcessTask


@global_config(prompt="task branch agent")
class TaskBranchAgent:
    ProcessFlow = ProcessFlow


def test_task_branch_compiles():
    engine = FlowForge.compile(TaskBranchAgent)
    dag    = engine.dag

    # The dispatcher task is in the DAG.
    dispatch_node = dag.get_node("global.process_flow.dispatch_process")
    assert dispatch_node is not None
    assert dispatch_node.meta.is_branch is True

    # Branch target tasks are added as children of the dispatcher.
    fast_node = dag.get_node("global.process_flow.dispatch_process.fast_process")
    slow_node = dag.get_node("global.process_flow.dispatch_process.slow_process")
    assert fast_node is not None
    assert slow_node is not None
