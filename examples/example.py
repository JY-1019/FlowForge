"""FlowForge × Anthropic Claude — End-to-End Planning & Visualization Demo
=========================================================================

이 예제가 보여주는 것
---------------------
1. 복잡한 다중 Flow / Task / Step DAG 정의
2. Claude로 모든 노드의 doc 자동 생성  (각 노드의 역할·능력 요약)
3. Claude AI Planner — 쿼리별로 최적의 서브트리를 doc 기반으로 선택
4. 엔진 실행 + RunTrace 수집
5. "전체 DAG vs 실제 실행 경로" 비교 Mermaid 출력 & 파일 저장

실행 방법
---------
    export ANTHROPIC_API_KEY="your-key-here"
    python examples/example_gemini.py

출력 파일
---------
    claude_dag_structure.md  — 전체 DAG 구조 Mermaid
    claude_run_compare.md    — 전체 DAG + 실행 경로 비교

DAG 구조 개요
-------------
ResearchAssistant (global)
├── quick_lookup      (flow)  — 간단한 사실 질문에 빠르게 답변
│   └── LookupTask            — 키워드 추출 → 직접 답변
│       ├── step[1] extract_keywords
│       └── step[2] find_answer
│
├── deep_research     (flow)  — 복잡한 주제 심층 다출처 분석
│   ├── collection    (flow)  — 여러 소스에서 데이터 수집
│   │   ├── WebSearchTask     — 웹 검색·크롤링
│   │   │   ├── step[1] optimize_query
│   │   │   └── step[2] fetch_results
│   │   └── AcademicSearchTask— 논문·학술자료 검색
│   │       ├── step[1] search_papers
│   │       └── step[2] extract_abstracts
│   ├── synthesis     (flow)  — 교차 분석·인사이트 도출
│   │   └── SynthesisTask
│   │       ├── step[1] cross_reference
│   │       └── step[2] generate_insights
│   └── ReportTask            — 종합 리포트 생성
│       ├── step[1] structure_report
│       └── step[2] finalize
│
└── fact_verification (flow)  — 특정 주장·사실의 진위 검증
    └── VerificationTask
        ├── step[1] parse_claim
        ├── step[2] search_evidence
        └── step[3] verdict
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# project root를 sys.path에 추가 (직접 실행 시)
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowforge import global_config, flow, task, step, FlowForge
from flowforge.types import LLMConfig
from flowforge.planner.llm_planner import LLMPlanner

# ---------------------------------------------------------------------------
# 0. Anthropic LLMConfig
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _API_KEY:
    print("[ERROR] ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
    print("        export ANTHROPIC_API_KEY='your-key'  후 재실행해 주세요.")
    sys.exit(1)

CLAUDE = LLMConfig.for_claude(
    model="claude-sonnet-4-6",
    api_key=_API_KEY,
    temperature=0.2,
)

# ---------------------------------------------------------------------------
# 1. Agent 정의  ── 반드시 모듈 레벨에 정의해야 함
# ---------------------------------------------------------------------------

@global_config(
    prompt=(
        "You are a versatile research assistant. "
        "Choose the most efficient path: "
        "quick_lookup for simple facts, "
        "deep_research for complex multi-source analysis, "
        "fact_verification for checking the truth of specific claims."
    ),
    llm_config=CLAUDE,
)
class ResearchAssistant:

    # ── Flow 1: 간단한 사실 질문 ─────────────────────────────────────────
    @flow(
        name="quick_lookup",
        prompt=(
            "Handle simple factual questions with a fast keyword-based lookup. "
            "Best for: definitions, short facts, direct Q&A where one source is enough."
        ),
    )
    class QuickLookupFlow:

        @task(
            name="lookup",
            prompt="Extract core keywords and return a direct, concise answer.",
        )
        class LookupTask:

            @step(order=1, prompt="Extract the key concepts and search terms from the query.")
            async def extract_keywords(ctx):
                return f"keywords extracted: {ctx.input}"

            @step(order=2, prompt="Find the most relevant direct answer for the keywords.")
            async def find_answer(ctx):
                return f"answer: {ctx.input}"

    # ── Flow 2: 심층 다출처 연구 ─────────────────────────────────────────
    @flow(
        name="deep_research",
        prompt=(
            "Conduct comprehensive multi-source research on complex topics. "
            "Best for: trend analysis, academic topics, anything needing synthesis "
            "from web sources AND academic literature."
        ),
    )
    class DeepResearchFlow:

        @flow(
            name="collection",
            prompt="Gather raw data from web sources and academic databases.",
        )
        class CollectionFlow:

            @task(
                name="web_search",
                prompt="Search and crawl web sources, extract relevant text content.",
            )
            class WebSearchTask:

                @step(order=1, prompt="Reformulate the query for maximum web search coverage.")
                async def optimize_query(ctx):
                    return f"optimized: {ctx.input}"

                @step(order=2, prompt="Fetch and parse the top-ranked search results.")
                async def fetch_results(ctx):
                    return f"web results: {ctx.input}"

            @task(
                name="academic_search",
                prompt="Search academic papers, journals, and citations on the topic.",
            )
            class AcademicSearchTask:

                @step(order=1, prompt="Query academic databases (arXiv, Semantic Scholar).")
                async def search_papers(ctx):
                    return f"papers: {ctx.input}"

                @step(order=2, prompt="Extract and normalize paper abstracts and key findings.")
                async def extract_abstracts(ctx):
                    return f"abstracts: {ctx.input}"

        @flow(
            name="synthesis",
            prompt="Cross-reference collected sources and distill key insights.",
        )
        class SynthesisFlow:

            @task(
                name="synthesize",
                prompt=(
                    "Cross-reference web and academic findings, resolve contradictions, "
                    "and generate consolidated insights."
                ),
            )
            class SynthesisTask:

                @step(order=1, prompt="Cross-reference findings across all collected sources.")
                async def cross_reference(ctx):
                    return f"cross-referenced: {ctx.input}"

                @step(order=2, prompt="Distill the most important insights and conclusions.")
                async def generate_insights(ctx):
                    return f"insights: {ctx.input}"

        @task(
            name="report",
            prompt="Compile all findings into a structured, comprehensive research report.",
        )
        class ReportTask:

            @step(order=1, prompt="Organize findings into a logical report structure.")
            async def structure_report(ctx):
                return f"structured: {ctx.input}"

            @step(order=2, prompt="Write the final polished report with citations.")
            async def finalize(ctx):
                return f"final report: {ctx.input}"

    # ── Flow 3: 사실 검증 ────────────────────────────────────────────────
    @flow(
        name="fact_verification",
        prompt=(
            "Verify the truth or accuracy of a specific claim or statement. "
            "Best for: fact-checking, myth-busting, verifying statistics or quotes."
        ),
    )
    class FactVerificationFlow:

        @task(
            name="verify",
            prompt=(
                "Parse the claim, gather supporting or refuting evidence from trusted "
                "sources, and deliver a TRUE / FALSE / UNCERTAIN verdict."
            ),
        )
        class VerificationTask:

            @step(order=1, prompt="Identify and isolate the specific claim(s) to verify.")
            async def parse_claim(ctx):
                return f"claim parsed: {ctx.input}"

            @step(order=2, prompt="Search for corroborating and contradicting evidence.")
            async def search_evidence(ctx):
                return f"evidence: {ctx.input}"

            @step(order=3, prompt="Evaluate evidence quality and deliver a credibility verdict.")
            async def verdict(ctx):
                return f"verdict: {ctx.input}"


# ---------------------------------------------------------------------------
# 2. 출력 헬퍼
# ---------------------------------------------------------------------------

SEP = "─" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def print_plan(query: str, plan, docs: dict) -> None:
    print(f"\n  Query  : {query!r}")
    print(f"  Mode   : {plan.mode}")
    if getattr(plan, "rationale", ""):
        # 긴 rationale는 줄바꿈해서 보기 좋게
        words = plan.rationale.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 60:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        for i, l in enumerate(lines):
            prefix = "  Why    : " if i == 0 else "           "
            print(f"{prefix}{l}")
    print(f"  Selected ({len(plan.node_ids)} nodes):")
    for nid in plan.node_ids:
        doc = docs.get(nid)
        summary = getattr(doc, "summary", "")[:65] if doc else ""
        print(f"    • {nid}")
        if summary:
            print(f"      └─ {summary}")


# ---------------------------------------------------------------------------
# 3. 메인
# ---------------------------------------------------------------------------

async def main() -> None:

    # ── Phase 1: DAG 컴파일 ──────────────────────────────────────────────
    section("Phase 1 — Compile DAG")
    engine = FlowForge.compile(ResearchAssistant)
    all_nodes = engine.dag.get_all_nodes()
    print(f"\n  총 {len(all_nodes)}개 노드 컴파일")
    for n in all_nodes:
        print(f"    [{n.type.value:6}]  {n.id}")

    # 전체 DAG 구조 저장
    dag_md = Path("claude_dag_structure.md")
    dag_md.write_text(
        "# ResearchAssistant — Full DAG Structure\n\n"
        "```mermaid\n" + engine.mermaid() + "\n```\n"
    )
    print(f"\n  전체 DAG → {dag_md}  (VS Code에서 미리보기 가능)")

    # ── Phase 2: Claude로 doc 생성 ───────────────────────────────────────
    section("Phase 2 — Generate Docs with Claude")
    print("\n  각 노드의 역할·능력 문서를 Claude가 생성합니다 (캐시 있으면 즉시)...\n")
    docs = await engine.generate_docs()
    print(f"  {len(docs)}개 노드 doc 완료\n")
    for node_id, doc in sorted(docs.items()):
        summary = getattr(doc, "summary", "—")[:80]
        caps    = getattr(doc, "capabilities", [])
        print(f"  {node_id}")
        print(f"    summary      : {summary}")
        if caps:
            print(f"    capabilities : {', '.join(caps[:3])}")

    # ── Phase 3: AI Planner — 쿼리별 최적 서브트리 선택 ─────────────────
    section("Phase 3 — AI Planner: Claude selects optimal subtree per query")

    planner = LLMPlanner(mode="autonomous")

    queries = [
        # 예상 경로
        ("What is the boiling point of water?",               "→ quick_lookup 예상"),
        ("Analyze the impact of LLMs on software engineering","→ deep_research 예상"),
        ("Is it true that humans only use 10% of their brain?","→ fact_verification 예상"),
    ]

    print()
    for query, hint in queries:
        print(f"  [{hint}]")
        plan = await planner.plan(query, engine.dag, docs, CLAUDE)
        print_plan(query, plan, docs)

    # ── Phase 4: 실제 실행 (planning_mode="autonomous" → 플래너가 선택한 경로만 실행)
    section("Phase 4 — Execute & Trace  (autonomous planning)")
    run_query = queries[1][0]   # deep_research 쿼리로 실행
    print(f"\n  실행 쿼리  : {run_query!r}")
    print(f"  planning   : autonomous  (Claude가 최적 서브트리 선택 → 해당 노드만 실행)\n")

    result, trace = await engine.run_traced(run_query, planning_mode="autonomous")

    print(f"  결과        : {result}")
    print(f"  성공 여부   : {'✓ OK' if trace.succeeded else '✗ FAILED'}")
    print(f"  총 시간     : {trace.duration_ms:.1f} ms")
    print(f"  실행된 노드 : {len(trace.executed_node_ids)} / {len(all_nodes)}")

    # ── Phase 5: 터미널 실행 요약 테이블 ────────────────────────────────
    section("Phase 5 — Execution Summary Table")
    engine.print_run_summary()

    # ── Phase 6: 전체 DAG vs 실행 경로 비교 저장 ────────────────────────
    section("Phase 6 — Compare: Full DAG vs Executed Path")
    compare_md = Path("claude_run_compare.md")
    compare_md.write_text(engine.compare_mermaid())
    print(f"\n  비교 Mermaid 저장 → {compare_md}")
    print(
        "\n  보는 방법:"
        "\n    1) VS Code: 파일 열고 Cmd+Shift+V"
        "\n    2) CLI:     flowforge run examples/example_gemini.py -q '...' --compare"
        "\n               (ANTHROPIC_API_KEY 환경변수 필요)"
        "\n    3) 웹:      각 mermaid 블록을 https://mermaid.live 에 붙여넣기"
    )

    # ── Phase 7: 실행 경로 Mermaid stdout ───────────────────────────────
    section("Phase 7 — Executed Path Mermaid")
    print(
        "\n  색상 범례:"
        "\n    ■ 진한 파랑(#154360)  = 실행된 flow"
        "\n    ■ 진한 초록(#145A32)  = 실행된 task"
        "\n    ■ 진한 보라(#512E5F)  = 실행된 step"
        "\n    ■ 빨강(#C0392B)       = 에러 발생 노드"
        "\n    □ 회색 점선            = 컴파일됐지만 이번 실행에서 건너뜀\n"
    )
    print(engine.run_mermaid())

    section("Done")
    print(f"\n  생성 파일:")
    print(f"    {dag_md}        ← 전체 DAG 구조")
    print(f"    {compare_md}    ← 전체 DAG + 실행 경로 비교\n")


if __name__ == "__main__":
    asyncio.run(main())
