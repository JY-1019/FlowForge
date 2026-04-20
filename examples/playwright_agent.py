"""Playwright MCP를 활용한 웹 스크래핑 & 분석 에이전트.

=======================================================================
실행 방법
=======================================================================

1. Playwright MCP 서버 설치 & 실행
   ──────────────────────────────
   npx @anthropic/mcp-playwright --port 3847

2. ANTHROPIC_API_KEY 설정
   ──────────────────────
       export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxx"

3. 에이전트 실행
   ────────────
       python examples/playwright_agent.py             # 단일 에이전트
       python examples/playwright_agent.py --route     # route 실행 예제
       python examples/playwright_agent.py --planner   # AI Planner 자동 선택 예제

=======================================================================
파이프라인
=======================================================================

  입력 (WebQuery)
    │
    ▼
  [validate_url]       ← Code step: URL 검증, https 보장
    │
    ▼
  [browse_and_extract] ← AI step: Playwright MCP tool-use loop
    │                     LLM이 browser_navigate → browser_snapshot 순서로 도구 호출
    │                     → 결과를 받아 structured_output(PageContent)으로 반환
    ▼
  [analyze_content]    ← AI step: 추출된 콘텐츠를 goal에 맞게 분석
    │                     structured_output(AnalysisReport) 로 반환
    ▼
  출력 (AnalysisReport)
"""
import asyncio
import logging
import sys
import textwrap
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from flowforge import global_config, flow, task, step, FlowForge
from flowforge.types import LLMConfig, MCPServer


# ─────────────────────────────────────────────────────────────────────
# I/O 스키마
# ─────────────────────────────────────────────────────────────────────

class WebQuery(BaseModel):
    """에이전트 입력: 탐색할 URL과 분석 목적."""
    url: str = Field(description="탐색할 웹 페이지 URL")
    goal: str = Field(
        default="페이지 내용을 요약해줘",
        description="무엇을 분석/추출할지 자연어로 기술",
    )


class ValidatedQuery(BaseModel):
    """Step 1 출력: 검증된 URL과 목적."""
    url: str
    goal: str
    domain: str


class PageContent(BaseModel):
    """Step 2 출력: 페이지에서 추출한 콘텐츠."""
    url: str
    title: str = ""
    content: str = ""
    links: list[str] = Field(default_factory=list, description="주요 링크 목록")
    goal: str = ""


class AnalysisReport(BaseModel):
    """에이전트 최종 출력: 분석 리포트."""
    title: str
    url: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    extracted_data: list[dict] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Playwright MCP 도구 설정
# ─────────────────────────────────────────────────────────────────────

playwright_navigate = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_navigate",
    description="지정된 URL로 브라우저를 이동시킨다",
)

playwright_snapshot = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_snapshot",
    description="현재 페이지의 접근성 스냅샷(텍스트 콘텐츠)을 가져온다",
)

playwright_click = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_click",
    description="페이지 내 특정 요소를 클릭한다",
)

playwright_scroll = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_scroll_down",
    description="페이지를 아래로 스크롤한다",
)

playwright_type = MCPServer(
    url="http://localhost:3847/mcp",
    name="browser_type",
    description="페이지 내 입력 필드에 텍스트를 입력한다",
)


# ─────────────────────────────────────────────────────────────────────
# 에이전트 정의
# ─────────────────────────────────────────────────────────────────────

@global_config(
    prompt=(
        "너는 웹 페이지 분석 전문 에이전트야. "
        "Playwright MCP를 사용해서 웹 브라우저를 제어하고, "
        "페이지 콘텐츠를 추출·분석한다. "
        "항상 한국어로 응답한다."
    ),
    llm_config=LLMConfig(
        model="claude-sonnet-4-6",
        temperature=0.2,
        max_tokens=4096,
    ),
    tools=[playwright_navigate, playwright_snapshot],
)
class WebAnalysisAgent:

    @flow(
        name="web_analysis",
        prompt="웹 페이지를 탐색하고 사용자 목적에 맞게 콘텐츠를 분석한다",
        input_schema=WebQuery,
        output_schema=AnalysisReport,
        tools=[playwright_click, playwright_scroll],
    )
    class WebAnalysisFlow:

        @task(
            name="scrape_and_analyze",
            prompt="웹 페이지 스크래핑 및 분석",
            on_error="skip_remaining",
        )
        class ScrapeAndAnalyzeTask:

            @step(
                order=1,
                prompt="URL 형식을 검증하고 정규화한다",
                input_schema=WebQuery,
                output_schema=ValidatedQuery,
            )
            async def validate_url(ctx):
                data = ctx.input
                url = data.url.strip()
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                parsed = urlparse(url)
                if not parsed.netloc:
                    raise ValueError(f"유효하지 않은 URL: '{data.url}'")
                return ValidatedQuery(
                    url=url, goal=data.goal, domain=parsed.netloc,
                )

            @step(
                order=2,
                prompt=(
                    "너는 웹 브라우저 조작 전문가야. "
                    "Playwright MCP 도구를 사용해서 웹 페이지를 탐색하고 "
                    "콘텐츠를 추출해. 페이지가 동적이면 스크롤해서 "
                    "더 많은 콘텐츠를 로드해."
                ),
                input_schema=ValidatedQuery,
                output_schema=PageContent,
                tools=[playwright_type],
            )
            async def browse_and_extract(ctx):
                result = await ctx.call_llm("""
                    '{url}' 페이지를 분석해야 해.
                    목적: {goal}
                    도메인: {domain}

                    다음 순서로 진행해:
                    1. <browser_navigate> 로 URL에 접속해
                    2. <browser_snapshot> 으로 페이지 콘텐츠를 읽어
                    3. 콘텐츠가 부족하면 <browser_scroll_down> 으로 스크롤해서 더 로드해
                    4. 필요하면 <browser_click> 으로 '더보기' 같은 버튼을 클릭해
                    5. 검색이 필요하면 <browser_type> 으로 검색창에 입력해

                    추출된 내용을 반환해.
                """)
                return result

            @step(
                order=3,
                prompt=(
                    "너는 콘텐츠 분석 전문가야. "
                    "웹 페이지에서 추출된 데이터를 사용자의 목적에 맞게 "
                    "분석하고 구조화된 리포트를 생성해."
                ),
                input_schema=PageContent,
                output_schema=AnalysisReport,
            )
            async def analyze_content(ctx):
                result = await ctx.call_llm("""
                    아래 웹 페이지 데이터를 분석해서 리포트를 만들어줘.

                    URL: {url}
                    제목: {title}
                    분석 목적: {goal}

                    추출된 콘텐츠:
                    {content}

                    주요 링크: {links}
                """)
                return result


# ─────────────────────────────────────────────────────────────────────
# 출력 헬퍼
# ─────────────────────────────────────────────────────────────────────

SEP = "=" * 70
THIN = "-" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def sub(title: str) -> None:
    print(f"\n{THIN}\n  {title}\n{THIN}")


def kv(key: str, value, indent: int = 4, max_width: int = 100) -> None:
    """Print key: value with optional wrapping."""
    prefix = " " * indent
    val = str(value)
    if len(val) > max_width:
        val = val[:max_width] + "..."
    print(f"{prefix}{key}: {val}")


def print_trace_detail(trace) -> None:
    """각 노드의 input/output을 상세히 출력."""
    for nt in trace.nodes:
        icon = "✓" if nt.succeeded else "✗"
        dur = f"{nt.duration_ms:.0f}ms" if nt.duration_ms is not None else "-"
        branch_info = f"  branch={nt.selected_branch}" if nt.selected_branch else ""

        sub(f"{icon} [{nt.node_type}] {nt.node_id}  ({dur}){branch_info}")

        if nt.input_repr and nt.input_repr != "None":
            print(f"    INPUT:")
            for line in textwrap.wrap(nt.input_repr, width=90):
                print(f"      {line}")

        if nt.output_repr and nt.output_repr != "None":
            print(f"    OUTPUT:")
            for line in textwrap.wrap(nt.output_repr, width=90):
                print(f"      {line}")

        if nt.error:
            print(f"    ERROR: {nt.error}")


# ─────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────

async def main():
    # ── 로깅 설정: flowforge 내부 로그를 INFO 레벨로 출력 ──
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-30s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # flowforge 실행 로그 INFO, LLM 호출 상세는 DEBUG
    logging.getLogger("flowforge.execution").setLevel(logging.INFO)
    logging.getLogger("flowforge.execution.llm").setLevel(logging.DEBUG)

    # ── Phase 1: 컴파일 ──
    header("Phase 1 — Compile")
    engine = FlowForge.compile(WebAnalysisAgent)
    all_nodes = engine.dag.get_all_nodes()
    print(f"\n  총 {len(all_nodes)}개 노드 컴파일 완료")
    for n in all_nodes:
        print(f"    [{n.type.value:6}]  {n.id}")

    # ── Phase 2: DAG 구조 ──
    header("Phase 2 — DAG Structure")
    print(engine.mermaid())

    # ── Phase 3: 실행 ──
    header("Phase 3 — Execute")
    query = WebQuery(
        url="https://news.ycombinator.com",
        goal="오늘의 상위 5개 기사 제목과 포인트를 추출해줘",
    )
    print(f"\n  URL  : {query.url}")
    print(f"  Goal : {query.goal}")
    print()

    result, trace = await engine.run_traced(query)

    # ── Phase 4: 실행 트레이스 상세 ──
    header("Phase 4 — Execution Trace (Detail)")
    print_trace_detail(trace)

    # ── Phase 5: 실행 요약 테이블 ──
    header("Phase 5 — Summary Table")
    engine.print_run_summary(trace)

    # ── Phase 6: 최종 결과 ──
    header("Phase 6 — Final Result")

    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    else:
        print(f"  Raw result: {result}")
        return

    kv("Title", data.get("title", ""))
    kv("URL", data.get("url", ""))
    kv("Summary", data.get("summary", ""))

    findings = data.get("key_findings", [])
    if findings:
        print("    Key Findings:")
        for i, f in enumerate(findings, 1):
            print(f"      {i}. {f}")

    extracted = data.get("extracted_data", [])
    if extracted:
        print("    Extracted Data:")
        for d in extracted:
            if isinstance(d, dict):
                label = d.get("label", d.get("title", ""))
                value = d.get("value", d.get("points", ""))
                print(f"      - {label}: {value}")
            else:
                print(f"      - {d}")

    # ── Phase 7: 비교 다이어그램 ──
    header("Phase 7 — DAG vs Executed Path")
    print(engine.compare_mermaid(trace))


# ═════════════════════════════════════════════════════════════════════
# Multi-flow 에이전트 — route 실행 예제
# ═════════════════════════════════════════════════════════════════════
#
# 구조:
#   @global_config (MultiFlowAgent)
#   ├─ @flow "web_analysis"       ← Playwright 웹 분석
#   │   └─ @task "scrape_and_analyze"
#   │       ├─ step[1] validate_url        (code)
#   │       ├─ step[2] browse_and_extract  (AI + MCP)
#   │       └─ step[3] analyze_content     (AI)
#   ├─ @flow "health_check"       ← URL 접속 가능 여부만 확인
#   │   └─ @task "check"
#   │       └─ step[1] ping
#   └─ @flow "summary"            ← 텍스트 요약 (웹 없이)
#       └─ @task "summarize"
#           └─ step[1] do_summary
#
# route 사용법:
#   route="web_analysis"                  → 웹 분석만
#   route="health_check"                  → 헬스체크만
#   route="summary"                       → 요약만
#   route=["health_check", "web_analysis"]→ 둘 다
# ═════════════════════════════════════════════════════════════════════


class HealthResult(BaseModel):
    url: str
    reachable: bool
    status_code: int = 0


class SummaryResult(BaseModel):
    original_length: int
    summary: str


@global_config(
    prompt=(
        "너는 다목적 웹 유틸리티 에이전트야. "
        "웹 분석, 헬스체크, 텍스트 요약을 수행한다."
    ),
    llm_config=LLMConfig(
        model="claude-sonnet-4-6",
        temperature=0.2,
        max_tokens=4096,
    ),
    tools=[playwright_navigate, playwright_snapshot],
)
class MultiFlowAgent:

    # ─── Flow 1: 웹 분석 (Playwright) ────────────────────────────────
    @flow(
        name="web_analysis",
        prompt="웹 페이지를 탐색하고 사용자 목적에 맞게 콘텐츠를 분석한다",
        input_schema=WebQuery,
        output_schema=AnalysisReport,
        tools=[playwright_click, playwright_scroll],
    )
    class WebAnalysisFlow:
        @task(
            name="scrape_and_analyze",
            prompt="웹 페이지 스크래핑 및 분석",
            on_error="skip_remaining",
        )
        class ScrapeAndAnalyzeTask:
            @step(order=1, prompt="URL 검증", input_schema=WebQuery, output_schema=ValidatedQuery)
            async def validate_url(ctx):
                data = ctx.input
                url = data.url.strip()
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                parsed = urlparse(url)
                if not parsed.netloc:
                    raise ValueError(f"유효하지 않은 URL: '{data.url}'")
                return ValidatedQuery(url=url, goal=data.goal, domain=parsed.netloc)

            @step(
                order=2,
                prompt=(
                    "Playwright MCP 도구를 사용해서 웹 페이지를 탐색하고 "
                    "콘텐츠를 추출해. 페이지가 동적이면 스크롤해서 더 많은 "
                    "콘텐츠를 로드해."
                ),
                input_schema=ValidatedQuery,
                output_schema=PageContent,
                tools=[playwright_type],
            )
            async def browse_and_extract(ctx):
                return await ctx.call_llm("""
                    '{url}' 페이지를 분석해야 해.
                    목적: {goal}
                    도메인: {domain}

                    다음 순서로 진행해:
                    1. <browser_navigate> 로 URL에 접속해
                    2. <browser_snapshot> 으로 페이지 콘텐츠를 읽어
                    3. 콘텐츠가 부족하면 <browser_scroll_down> 으로 스크롤
                    4. 필요하면 <browser_click> 으로 버튼 클릭
                    추출된 내용을 반환해.
                """)

            @step(
                order=3,
                prompt="추출된 콘텐츠를 분석하여 리포트 생성",
                input_schema=PageContent,
                output_schema=AnalysisReport,
            )
            async def analyze_content(ctx):
                return await ctx.call_llm("""
                    아래 데이터로 리포트를 만들어줘.
                    URL: {url} / 제목: {title} / 목적: {goal}
                    콘텐츠: {content}
                """)

    # ─── Flow 2: 헬스체크 (코드만, LLM 없음) ────────────────────────
    @flow(name="health_check", prompt="URL 접속 가능 여부를 확인한다")
    class HealthCheckFlow:
        @task(name="check", prompt="HTTP 요청으로 상태 확인")
        class CheckTask:
            @step(order=1, prompt="URL에 HTTP 요청을 보내 응답 코드를 확인한다")
            async def ping(ctx):
                import asyncio
                import urllib.request
                import urllib.error

                data = ctx.input
                if isinstance(data, dict):
                    url = data.get("url", "")
                elif hasattr(data, "url"):
                    url = data.url
                else:
                    url = str(data)
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    loop = asyncio.get_running_loop()
                    resp = await loop.run_in_executor(
                        None,
                        lambda: urllib.request.urlopen(req, timeout=5),
                    )
                    return HealthResult(url=url, reachable=True, status_code=resp.status)
                except urllib.error.HTTPError as e:
                    return HealthResult(url=url, reachable=True, status_code=e.code)
                except Exception:
                    return HealthResult(url=url, reachable=False, status_code=0)

    # ─── Flow 3: 텍스트 요약 (LLM, MCP 불필요) ──────────────────────
    @flow(name="summary", prompt="입력 텍스트를 요약한다")
    class SummaryFlow:
        @task(name="summarize", prompt="텍스트를 3줄로 요약")
        class SummarizeTask:
            @step(
                order=1,
                prompt="입력 텍스트를 핵심만 남겨 간결하게 요약해.",
                output_schema=SummaryResult,
            )
            async def do_summary(ctx):
                return await ctx.call_llm("""
                    다음 텍스트를 3줄 이내로 요약해.
                    원문 길이(글자수)도 함께 알려줘.

                    텍스트:
                    {text}
                """)


# ─────────────────────────────────────────────────────────────────────
# Route 실행 예제
# ─────────────────────────────────────────────────────────────────────

def print_result(result) -> None:
    """결과를 보기 좋게 출력."""
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    else:
        print(f"    {result}")
        return
    for k, v in data.items():
        val = str(v)
        if len(val) > 120:
            val = val[:120] + "..."
        print(f"    {k}: {val}")


async def main_route():
    """route= 파라미터로 특정 flow만 실행하는 예제."""

    # ── 로깅 ──
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-35s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("flowforge.execution").setLevel(logging.INFO)
    logging.getLogger("flowforge.execution.llm").setLevel(logging.DEBUG)

    engine = FlowForge.compile(MultiFlowAgent)

    # ── DAG 전체 구조 ──
    header("DAG Structure — MultiFlowAgent")
    all_nodes = engine.dag.get_all_nodes()
    print(f"\n  총 {len(all_nodes)}개 노드")
    for n in all_nodes:
        print(f"    [{n.type.value:6}]  {n.id}")
    print()
    print(engine.mermaid())

    query = WebQuery(
        url="https://news.ycombinator.com",
        goal="상위 5개 기사 제목 추출",
    )

    # ──────────────────────────────────────────────────────────────────
    # 1. route="health_check" — 헬스체크만 실행
    # ──────────────────────────────────────────────────────────────────
    header("Route 1 — health_check only")
    print('  engine.run(query, route="health_check")')
    print()

    result, trace = await engine.run_traced(query, route="health_check")

    print(f"  실행된 노드 ({len(trace.executed_node_ids)}개):")
    for nid in sorted(trace.executed_node_ids):
        print(f"    ✓ {nid}")
    print(f"\n  건너뛴 노드:")
    for n in all_nodes:
        if n.id not in trace.executed_node_ids:
            print(f"    · {n.id}")
    print(f"\n  결과:")
    print_result(result)

    # ──────────────────────────────────────────────────────────────────
    # 2. route="summary" — 요약만 실행
    # ──────────────────────────────────────────────────────────────────
    header("Route 2 — summary only")
    print('  engine.run(text_input, route="summary")')
    print()

    text_input = {
        "text": (
            "FlowForge는 Python 데코레이터만으로 AI Agent 시스템을 구축할 수 있는 프레임워크다. "
            "@flow, @task, @step 어노테이션으로 에이전트 구조를 정의하면 "
            "프레임워크가 DAG 스키마로 컴파일한다. AI Planner가 이 스키마를 보고 "
            "유저 요청에 최적화된 실행 경로를 선택한다."
        ),
    }
    result, trace = await engine.run_traced(text_input, route="summary")

    print(f"  실행된 노드 ({len(trace.executed_node_ids)}개):")
    for nid in sorted(trace.executed_node_ids):
        print(f"    ✓ {nid}")
    print(f"\n  결과:")
    print_result(result)

    # ──────────────────────────────────────────────────────────────────
    # 3. route="web_analysis" — 웹 분석만 실행
    # ──────────────────────────────────────────────────────────────────
    header("Route 3 — web_analysis only")
    print('  engine.run(query, route="web_analysis")')
    print()

    result, trace = await engine.run_traced(query, route="web_analysis")

    print(f"\n  실행된 노드 ({len(trace.executed_node_ids)}개):")
    for nid in sorted(trace.executed_node_ids):
        print(f"    ✓ {nid}")
    print(f"\n  결과:")
    print_result(result)

    # ── 트레이스 상세 ──
    sub("Execution Trace Detail")
    print_trace_detail(trace)

    # ──────────────────────────────────────────────────────────────────
    # 4. route=["health_check", "web_analysis"] — 복수 route
    # ──────────────────────────────────────────────────────────────────
    header("Route 4 — multiple: health_check + web_analysis")
    print('  engine.run(query, route=["health_check", "web_analysis"])')
    print()

    result, trace = await engine.run_traced(
        query, route=["health_check", "web_analysis"]
    )

    print(f"  실행된 노드 ({len(trace.executed_node_ids)}개):")
    for nid in sorted(trace.executed_node_ids):
        print(f"    ✓ {nid}")
    print(f"\n  건너뛴 노드 (summary는 실행 안 됨):")
    for n in all_nodes:
        if n.id not in trace.executed_node_ids:
            print(f"    · {n.id}")
    print(f"\n  결과 (마지막 flow인 web_analysis 출력):")
    print_result(result)

    # ── 비교 다이어그램 ──
    header("DAG vs Executed Path (Route 4)")
    print(engine.compare_mermaid(trace))

    # ── 요약 테이블 ──
    header("Summary Table (Route 4)")
    engine.print_run_summary(trace)


# ─────────────────────────────────────────────────────────────────────
# AI Planner 자동 경로 선택 예제
# ─────────────────────────────────────────────────────────────────────

async def main_planner():
    """AI Planner가 유저 질문을 분석하여 적절한 flow를 자동 선택하는 예제.

    route= 파라미터 없이 planning_mode="autonomous"를 사용하면
    LLM Planner가 DAG 구조와 각 노드의 doc(요약·capabilities)을 보고
    유저 요청에 맞는 flow만 골라서 실행한다.

    사전 조건:
      - ANTHROPIC_API_KEY 환경변수 필요 (doc 생성 + planner LLM 호출)
      - Playwright MCP 서버: npx @anthropic/mcp-playwright --port 3847
    """

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-35s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("flowforge.execution").setLevel(logging.INFO)
    logging.getLogger("flowforge.planner").setLevel(logging.DEBUG)

    # ── Phase 1: 컴파일 ──
    header("Phase 1 — Compile MultiFlowAgent")
    engine = FlowForge.compile(MultiFlowAgent)
    all_nodes = engine.dag.get_all_nodes()
    print(f"\n  총 {len(all_nodes)}개 노드")
    for n in all_nodes:
        print(f"    [{n.type.value:6}]  {n.id}")

    # ── Phase 2: Doc 생성 (Planner가 경로 선택에 활용) ──
    header("Phase 2 — Generate Docs (for Planner)")
    docs = await engine.generate_docs()
    print(f"\n  {len(docs)}개 노드의 doc 생성 완료")
    for node_id, doc in docs.items():
        summary = getattr(doc, "summary", "")[:80]
        caps = getattr(doc, "capabilities", [])[:3]
        caps_str = f"  caps={caps}" if caps else ""
        print(f"    {node_id}")
        print(f"      summary: {summary}")
        if caps_str:
            print(f"      {caps_str}")

    # ── Phase 3: 다양한 질문으로 Planner 자동 선택 테스트 ──
    test_cases = [
        {
            "label": "웹 분석 요청",
            "input": WebQuery(
                url="https://news.ycombinator.com",
                goal="상위 5개 기사 제목을 추출해줘",
            ),
            "expected_flow": "web_analysis",
        },
        {
            "label": "헬스체크 요청",
            "input": WebQuery(
                url="https://example.com",
                goal="이 URL에 접속이 되는지 확인해줘",
            ),
            "expected_flow": "health_check",
        },
        {
            "label": "텍스트 요약 요청",
            "input": {
                "text": (
                    "FlowForge는 Python 데코레이터만으로 AI Agent 시스템을 "
                    "구축할 수 있는 프레임워크다. @flow, @task, @step 어노테이션으로 "
                    "에이전트 구조를 정의하면 프레임워크가 DAG 스키마로 컴파일한다."
                ),
            },
            "expected_flow": "summary",
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        header(f"Test {i} — {tc['label']}")
        print(f"  입력: {tc['input']}")
        print(f"  기대하는 flow: {tc['expected_flow']}")
        print()

        try:
            result, trace = await engine.run_traced(
                tc["input"],
                planning_mode="autonomous",
            )

            # 어떤 flow가 실행되었는지 확인
            executed_flows = [
                nid for nid in trace.executed_node_ids
                if nid.startswith("global.") and nid.count(".") == 1
            ]
            print(f"  Planner가 선택한 flow: {executed_flows}")
            print(f"  실행된 노드 ({len(trace.executed_node_ids)}개):")
            for nid in sorted(trace.executed_node_ids):
                print(f"    ✓ {nid}")

            # 기대한 flow가 선택되었는지 체크
            expected_id = f"global.{tc['expected_flow']}"
            if expected_id in trace.executed_node_ids:
                print(f"\n  ✅ 기대한 flow '{tc['expected_flow']}' 가 올바르게 선택됨!")
            else:
                print(f"\n  ⚠️  기대한 flow '{tc['expected_flow']}' 가 선택되지 않음")

            print(f"\n  결과:")
            print_result(result)

        except Exception as e:
            print(f"  ❌ 실행 실패: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if "--planner" in sys.argv:
        asyncio.run(main_planner())
    elif "--route" in sys.argv:
        asyncio.run(main_route())
    else:
        asyncio.run(main())
