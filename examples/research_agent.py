"""Example research agent from FlowForge design spec section 13.

Demonstrates:
- Nested @flow (search sub-flow inside research flow)
- Container @task (analyze_and_format with child tasks)
- @step with branch dispatching (replaces the old standalone @branch)
"""
import asyncio
from pydantic import BaseModel

from flowforge import global_config, flow, task, step, FlowForge, LLMConfig, BranchCondition


# ─── Schemas ───
class UserQuery(BaseModel):
    query: str
    language: str = "ko"

class AnalyzedQuery(BaseModel):
    intent: str
    keywords: list[str]
    source_preference: str  # "web" | "db" | "api"

class SearchResult(BaseModel):
    results: list[dict]
    source: str

class FormattedAnswer(BaseModel):
    answer: str
    citations: list[str]


# ─── Branch Handlers ───
async def web_search_handler(ctx):
    return SearchResuldk t(results=[{"title": "Web result", "url": "https://example.com"}], source="web")

async def db_search_handler(ctx):
    return SearchResult(results=[{"title": "DB result", "id": "123"}], source="db")

async def api_search_handler(ctx):
    return SearchResult(results=[{"title": "API result", "data": {}}], source="api")


# ─── Agent Definition ───
@global_config(
    prompt="다국어 리서치 어시스턴트. 정확한 출처와 함께 답변한다.",
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

        # ─── 자식 Flow: 검색 ───
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
                async def optimize_query(ctx):
                    print(f"[Step 1] Optimizing query: {ctx.input}")
                    return ctx.input

                # @step with condition/branches — branch dispatching is now
                # a built-in capability of @step (no separate @branch needed).
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
                async def route_source(ctx):
                    ...

                @step(order=3, prompt="검색 결과 정제 및 중복 제거")
                async def deduplicate(ctx):
                    print(f"[Step 3] Deduplicating results")
                    return ctx.input

        # ─── 메인 Task: 분석 + 포맷 ───
        @task(name="analyze_and_format", prompt="쿼리 분석 및 최종 답변 포맷팅")
        class AnalyzeAndFormatTask:

            @task(name="analyze", prompt="유저 쿼리의 의도와 키워드를 분석")
            class AnalyzeTask:
                @step(order=1, prompt="쿼리 의도 분류")
                async def classify_intent(ctx):
                    print(f"[Analyze] Classifying intent")
                    return AnalyzedQuery(
                        intent="search",
                        keywords=["AI", "agent"],
                        source_preference="web",
                    )

            @task(name="format", prompt="검색 결과를 최종 답변으로 포맷팅")
            class FormatTask:
                @step(order=1, prompt="답변 초안 생성")
                async def draft_answer(ctx):
                    print(f"[Format] Drafting answer")
                    return FormattedAnswer(
                        answer="AI 에이전트 프레임워크 트렌드: ...",
                        citations=["https://example.com"],
                    )

                @step(order=2, prompt="출처 인용 추가")
                async def add_citations(ctx):
                    print(f"[Format] Adding citations")
                    return ctx.input


async def main():
    # Compile
    engine = FlowForge.compile(ResearchAgent)
    print(f"Compiled DAG with {len(engine.dag)} nodes")
    print("\nMermaid diagram:")
    print(engine.mermaid())

    # Run
    result = await engine.run(UserQuery(query="2026년 AI 에이전트 프레임워크 트렌드"))
    print(f"\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
