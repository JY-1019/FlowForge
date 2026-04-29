"""Dynamic paper-report example with arXiv — static pipeline + dynamic fetch.

The downstream pipeline is defined statically:

    paper_report_pipeline
        ├─ normalize_input
        ├─ paper_digest
        ├─ slide_briefing
        └─ ppt_render

What is intentionally missing is the upstream "fetch papers" flow.
With ``dynamic_flow=True`` enabled, the planner reports a partial gap,
the built-in ``_dynamic_generator`` injects the missing fetch flow, and
the engine replans so execution continues as:

    <dynamically generated fetch flow>  ->  paper_report_pipeline

This is a **zero pre-defined tools** example:
- No custom ``FunctionTool`` is registered on the agent.
- The dynamically generated fetch flow uses the builtin ``web_fetch_url``
  tool to call the arXiv API.
- The ``ppt_render`` step uses the builtin ``pptx_create`` tool.
- All tools come from ``include_builtin_tools=True`` (the default).

Run:

    python examples/dynamic_paper_report_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from flowforge import DynamicRunOptions, FlowForge, flow, global_config, step, task
from flowforge.types import DependencyPolicy, LLMConfig


LOG = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_paper_report"
GENERATED_DIR = "examples/_artifacts/dynamic_paper_report/generated"
ARXIV_API_URL = "https://export.arxiv.org/api/query"


# ---------------------------------------------------------------------------
# Normaliser — coerce dynamic fetch output into pipeline input
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value

    text = _strip_markdown_fences(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    raise ValueError("Could not parse JSON-like paper payload.")


def _first_present(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def _normalise_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(author).strip() for author in value if str(author).strip()]
    if isinstance(value, str):
        return [
            author.strip()
            for author in re.split(r",|\band\b", value)
            if author.strip()
        ]
    return []


def _abstract_url_from_pdf(pdf_url: str) -> str:
    if "/pdf/" not in pdf_url:
        return ""
    abstract_url = pdf_url.replace("/pdf/", "/abs/")
    return abstract_url.removesuffix(".pdf")


def _normalise_paper_record(record: Any, rank: int) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        record = record.model_dump()
    record = _parse_json_like(record)
    if not isinstance(record, dict):
        raise ValueError("Each paper item must be a dict-like object.")

    url = str(_first_present(record, "abstract_url", "url", "link"))
    pdf_url = str(_first_present(record, "pdf_url", "pdf", "pdf_link"))
    if not pdf_url and ("/pdf/" in url or url.endswith(".pdf")):
        pdf_url = url

    abstract_url = str(_first_present(record, "abstract_url", "abs_url"))
    if not abstract_url and url and url != pdf_url:
        abstract_url = url
    if not abstract_url and pdf_url:
        abstract_url = _abstract_url_from_pdf(pdf_url)

    rank_value = _first_present(record, "rank", "index", "position", default=rank)
    try:
        rank_value = int(rank_value)
    except (TypeError, ValueError):
        rank_value = rank

    return {
        "rank": rank_value,
        "title": _first_present(record, "title", "paper_title", "name"),
        "authors": _normalise_authors(record.get("authors", [])),
        "published": _first_present(
            record, "published", "published_date", "date", "updated"
        ),
        "abstract_url": abstract_url,
        "pdf_url": pdf_url,
        "summary_snippet": _first_present(
            record, "summary_snippet", "summary", "abstract", "abstract_text",
            "description",
        ),
    }


def _normalise_paper_records(papers: Any) -> list[dict[str, Any]]:
    papers = _parse_json_like(papers)
    if isinstance(papers, dict) and "papers" in papers:
        papers = _parse_json_like(papers["papers"])
    if isinstance(papers, dict):
        papers = [papers]
    if not isinstance(papers, list):
        raise ValueError("Paper list must be a list or dict-like object.")
    return [
        _normalise_paper_record(paper, index)
        for index, paper in enumerate(papers, start=1)
    ]


def normalise_paper_payload(data: Any) -> dict[str, Any]:
    """Coerce dynamic fetch-flow output into the pipeline input contract."""
    if hasattr(data, "model_dump"):
        data = data.model_dump()

    data = _parse_json_like(data)
    source_url = ARXIV_API_URL
    fetched_at = datetime.now(timezone.utc).isoformat()

    if isinstance(data, list):
        papers = _normalise_paper_records(data)
        payload = {
            "papers": papers,
            "source_url": source_url,
            "fetched_at": fetched_at,
        }
        return PaperPayload.model_validate(payload).model_dump()

    if not isinstance(data, dict):
        raise ValueError("Paper payload must be a dict-like object.")

    source_url = _first_present(data, "source_url", "source", default=source_url)
    fetched_at = _first_present(data, "fetched_at", "generated_at", default=fetched_at)

    if {"papers", "source_url", "fetched_at"}.issubset(data):
        payload = {
            "papers": _normalise_paper_records(data["papers"]),
            "source_url": data["source_url"],
            "fetched_at": data["fetched_at"],
        }
        return PaperPayload.model_validate(payload).model_dump()

    candidate: Any = None
    for key in ("papers_payload", "papers", "payload", "result", "data"):
        if key in data:
            candidate = data[key]
            break

    if candidate is None:
        if any(key in data for key in ("title", "paper_title", "abstract")):
            candidate = data
        else:
            raise ValueError(
                "Dynamic fetch flow did not return a recognised paper payload shape."
            )

    candidate = _parse_json_like(candidate)

    if isinstance(candidate, dict):
        if "papers" in candidate:
            papers = _parse_json_like(candidate["papers"])
            source_url = _first_present(
                candidate, "source_url", "source", default=source_url
            )
            fetched_at = _first_present(
                candidate, "fetched_at", "generated_at", default=fetched_at
            )
        else:
            papers = [candidate]
    elif isinstance(candidate, list):
        papers = candidate
    else:
        raise ValueError("Dynamic fetch flow returned an unsupported payload value.")

    payload = {
        "papers": _normalise_paper_records(papers),
        "source_url": source_url,
        "fetched_at": fetched_at,
    }
    return PaperPayload.model_validate(payload).model_dump()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _llm_config_from_env() -> LLMConfig:
    provider = os.getenv("FLOWFORGE_PROVIDER", "").strip().lower()
    model = os.getenv("FLOWFORGE_MODEL", "").strip()

    if provider == "openai":
        return LLMConfig.for_openai(
            model=model or "gpt-4o",
            verify_ssl=False,
        )
    if provider == "google":
        return LLMConfig.for_gemini(
            model=model or "gemini-2.0-flash",
            verify_ssl=False,
        )
    return LLMConfig.for_claude(
        model=model or "claude-sonnet-4-6",
        verify_ssl=False,
    )


def _dynamic_options(
    *,
    generated_dir: str = GENERATED_DIR,
    auto_load_generated: bool = True,
) -> DynamicRunOptions:
    """Options tuned for examples: fast generation, persisted generated files."""
    return DynamicRunOptions(
        project_root=str(ROOT_DIR.parent),
        generated_dir=generated_dir,
        auto_load_generated=auto_load_generated,
        persist_generated=True,
        include_builtin_tools=True,
        allow_codegen_tool_use=False,
        allowed_shell_modes=["readonly", "project_exec"],
        shell_output_max_chars=int(os.getenv("FLOWFORGE_PAPER_FETCH_CHARS", "50000")),
        project_context_max_chars=4000,
        dependency_policy=DependencyPolicy(allow_install=True),
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    rank: int
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    abstract_url: str = ""
    pdf_url: str = ""
    summary_snippet: str = ""


class PaperPayload(BaseModel):
    papers: list[Paper] = Field(min_length=1)
    source_url: str
    fetched_at: str


class PaperDigest(BaseModel):
    rank: int
    title: str
    authors: list[str] = Field(default_factory=list)
    headline: str
    easy_summary: str
    problem: str
    approach: str
    why_it_matters: str
    caveats: list[str] = Field(default_factory=list)


class PaperDigestPayload(BaseModel):
    papers: list[PaperDigest]
    source_url: str
    fetched_at: str


class SlideSpec(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list)
    speaker_note: str = ""


class SlideDeckData(BaseModel):
    deck_title: str
    deck_subtitle: str
    source_url: str
    fetched_at: str
    slides: list[SlideSpec]


class PresentationArtifact(BaseModel):
    pptx_path: str
    slide_count: int
    source_url: str
    generated_at: str


def _short_text(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fallback_digest_payload(payload: dict[str, Any]) -> PaperDigestPayload:
    digests: list[dict[str, Any]] = []
    for paper in payload.get("papers", []):
        title = str(paper.get("title", "")).strip()
        snippet = _short_text(paper.get("summary_snippet", ""), 360)
        authors = paper.get("authors", [])
        if not isinstance(authors, list):
            authors = _normalise_authors(authors)
        digests.append({
            "rank": int(paper.get("rank") or len(digests) + 1),
            "title": title,
            "authors": authors,
            "headline": _short_text(title, 90),
            "easy_summary": snippet or "arXiv 초록을 기반으로 한 요약을 생성할 수 없습니다.",
            "problem": "논문 초록에서 제시한 연구 문제를 다룹니다.",
            "approach": "제공된 초록과 메타데이터를 바탕으로 접근 방식을 정리했습니다.",
            "why_it_matters": "최신 AI 연구 흐름을 빠르게 파악하는 데 도움이 됩니다.",
            "caveats": ["LLM 호출 실패로 세부 해석은 원문 확인이 필요합니다."],
        })
    return PaperDigestPayload.model_validate({
        "papers": digests,
        "source_url": payload["source_url"],
        "fetched_at": payload["fetched_at"],
    })


def _fallback_slide_deck(payload: dict[str, Any]) -> SlideDeckData:
    papers = payload.get("papers", [])
    titles = [str(paper.get("title", "")).strip() for paper in papers]
    slides: list[dict[str, Any]] = [
        {
            "title": "최신 AI 논문 3선",
            "bullets": [
                "arXiv 최신 항목을 기준으로 구성",
                "각 논문의 문제, 접근, 의미를 간단히 정리",
                "네트워크/LLM 장애 시에도 생성되는 안정화 deck",
            ],
            "speaker_note": "자동 fallback으로 생성된 보고서입니다.",
        },
        {
            "title": "한눈에 보는 요약",
            "bullets": [
                _short_text(title, 95) for title in titles[:3]
            ] or ["가져온 논문 목록을 확인하세요."],
            "speaker_note": "논문 제목 중심의 빠른 개요입니다.",
        },
    ]

    for index in range(3):
        paper = papers[index] if index < len(papers) else {}
        slides.append({
            "title": f"논문 {index + 1}: {_short_text(paper.get('title', '확인 필요'), 70)}",
            "bullets": [
                _short_text(paper.get("headline", paper.get("title", "")), 100),
                _short_text(paper.get("easy_summary", ""), 120),
                _short_text(paper.get("why_it_matters", ""), 100),
            ],
            "speaker_note": _short_text(paper.get("problem", ""), 220),
        })

    slides.extend([
        {
            "title": "비교 / 공통 인사이트",
            "bullets": [
                "최신 AI 연구는 모델 효율, 추론 품질, 실제 적용성을 함께 다룹니다.",
                "각 논문은 서로 다른 문제 설정에서 개선 방향을 제시합니다.",
                "초록 기반 요약이므로 세부 실험 수치는 원문 검증이 필요합니다.",
            ],
            "speaker_note": "원문 기반 추가 검토를 권장합니다.",
        },
        {
            "title": "결론 / 추천 액션",
            "bullets": [
                "관심 논문의 abstract와 PDF를 먼저 확인",
                "관련 코드/데이터 공개 여부 점검",
                "팀 과제와 연결되는 아이디어를 후속 조사",
            ],
            "speaker_note": "보고서의 다음 액션을 정리합니다.",
        },
    ])
    return SlideDeckData.model_validate({
        "deck_title": "최신 AI 논문 3선",
        "deck_subtitle": "arXiv 기반 자동 보고서",
        "source_url": payload["source_url"],
        "fetched_at": payload["fetched_at"],
        "slides": slides[:7],
    })


# ---------------------------------------------------------------------------
# Static downstream pipeline
# ---------------------------------------------------------------------------

@flow(
    name="paper_report_pipeline",
    prompt=(
        "Already-fetched paper payload is provided as input. "
        "This flow normalizes the payload, writes easy Korean summaries, "
        "prepares a 7-slide report deck, and renders the final PPTX. "
        "It does NOT fetch papers by itself."
    ),
)
class PaperReportPipeline:

    @task(
        name="normalize_input",
        prompt=(
            "Normalize any dynamically generated fetch-flow output into the "
            "strict PaperPayload schema expected by the pipeline."
        ),
    )
    class NormalizeInputTask:
        @step(
            order=1,
            prompt="Coerce wrapper JSON or markdown-fenced JSON into the paper payload schema.",
            output_schema=PaperPayload,
        )
        async def normalize(ctx):
            normalized = normalise_paper_payload(ctx.input)
            return PaperPayload.model_validate(normalized)

    @task(name="paper_digest", prompt="summarize each paper for Korean readers")
    class PaperDigestTask:
        @step(
            order=1,
            prompt=(
                "You explain research papers in very easy Korean for busy "
                "non-specialists. Keep the tone concrete, warm, and simple."
            ),
            input_schema=PaperPayload,
            output_schema=PaperDigestPayload,
        )
        async def digest(ctx):
            payload = ctx.input.model_dump()
            compact_payload = {
                "papers": [
                    {
                        "rank": paper["rank"],
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "summary_snippet": paper["summary_snippet"],
                    }
                    for paper in payload["papers"]
                ],
                "source_url": payload["source_url"],
                "fetched_at": payload["fetched_at"],
            }
            prompt = (
                "다음 논문 데이터를 바탕으로 한국어 쉬운 설명 자료를 만들어줘.\n"
                "- 비전공자도 이해 가능한 표현을 사용해.\n"
                "- 각 논문마다 headline, easy_summary, problem, approach, "
                "why_it_matters, caveats를 채워.\n"
                "- 과장하지 말고, 본문에 없는 내용은 추측하지 마.\n\n"
                f"{json.dumps(compact_payload, ensure_ascii=False)}"
            )
            try:
                return await ctx.call_llm(prompt)
            except Exception as exc:
                LOG.warning(
                    "paper_digest LLM failed; using deterministic fallback: %s",
                    exc,
                )
                return _fallback_digest_payload(payload)

    @task(name="slide_briefing", prompt="prepare a fixed 7-slide report deck")
    class SlideBriefingTask:
        @step(
            order=1,
            prompt=(
                "You are a presentation strategist. Produce a Korean report-style "
                "deck with exactly 7 slides and concise bullets."
            ),
            input_schema=PaperDigestPayload,
            output_schema=SlideDeckData,
        )
        async def brief(ctx):
            payload = ctx.input.model_dump()
            prompt = (
                "다음 한국어 논문 요약을 바탕으로 정확히 7장의 보고서형 슬라이드를 만들어줘.\n"
                "슬라이드 순서는 반드시 다음과 같아:\n"
                "1. 표지\n"
                "2. 한눈에 보는 요약\n"
                "3. 논문 1\n"
                "4. 논문 2\n"
                "5. 논문 3\n"
                "6. 비교/공통 인사이트\n"
                "7. 결론/추천 액션\n"
                "각 슬라이드는 title, bullets, speaker_note를 채워.\n"
                "bullets는 3~5개 정도의 짧은 문장으로 써.\n\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
            try:
                return await ctx.call_llm(prompt)
            except Exception as exc:
                LOG.warning(
                    "slide_briefing LLM failed; using deterministic fallback: %s",
                    exc,
                )
                return _fallback_slide_deck(payload)

    @task(name="ppt_render", prompt="render the final pptx artifact using builtin pptx_create tool")
    class PptRenderTask:
        @step(
            order=1,
            prompt="Render the prepared slide data into a .pptx file using the builtin pptx_create tool.",
            input_schema=SlideDeckData,
            output_schema=PresentationArtifact,
        )
        async def render(ctx):
            deck = ctx.input.model_dump()
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            rel_path = f"_artifacts/dynamic_paper_report/arxiv_report_{timestamp}.pptx"

            slides_for_pptx = [
                {
                    "title": slide.get("title", ""),
                    "bullets": slide.get("bullets", []),
                }
                for slide in (
                    s.model_dump() if hasattr(s, "model_dump") else s
                    for s in ctx.input.slides
                )
            ]

            # Use the builtin pptx_create tool
            result = await ctx.call_tool(
                "pptx_create",
                path=rel_path,
                slides=json.dumps(slides_for_pptx, ensure_ascii=False),
            )

            if not result.get("ok"):
                # Fallback: save as JSON if pptx_create fails
                json_rel = rel_path.replace(".pptx", ".json")
                await ctx.call_tool(
                    "files_write_text",
                    path=json_rel,
                    content=json.dumps(slides_for_pptx, ensure_ascii=False, indent=2),
                )
                LOG.warning("pptx_create failed — saved slides as JSON: %s", json_rel)
                rel_path = json_rel

            return PresentationArtifact(
                pptx_path=rel_path,
                slide_count=len(ctx.input.slides),
                source_url=deck["source_url"],
                generated_at=datetime.now(timezone.utc).isoformat(),
            )


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

@global_config(
    prompt=(
        "너는 최신 AI 논문을 찾아 쉬운 한국어 발표 자료로 바꾸는 에이전트다. "
        "기존 Flow가 부족하면 dynamic flow가 생성될 수 있다. "
        "이미 있는 도구와 Flow를 최대한 활용하고, 최종 결과는 보고서 형식의 "
        "읽기 쉬운 PPT여야 한다."
    ),
    llm_config=_llm_config_from_env(),
    tools=[],
    dynamic_flow=True,
)
class DynamicPaperReportAgent:
    PaperReportPipeline = PaperReportPipeline


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_pipeline_gap_example() -> None:
    """Static downstream pipeline + dynamic upstream fetch-flow generation."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    options = _dynamic_options()

    engine = FlowForge.compile(DynamicPaperReportAgent, dynamic_options=options)
    await engine.generate_docs(planning_only=True)

    user_request = (
        "arXiv에서 최신 AI 논문 3개를 가져와 한국어 보고서형 PPT로 만들어줘. "
        "논문 목록이 비어 있거나 실제 arXiv 데이터가 아니면 성공으로 처리하지 말고 에러를 내."
    )
    result, trace = await engine.run_traced(
        user_request,
        planning_mode="autonomous",
        dynamic_options=options,
    )

    mermaid_path = ARTIFACT_DIR / "dynamic_paper_report_run.md"
    mermaid_path.write_text(engine.compare_mermaid(trace), encoding="utf-8")

    dynamic_info = engine.last_dynamic_generation or {}
    if dynamic_info.get("generated_code"):
        code_path = ARTIFACT_DIR / f"{dynamic_info.get('dynamic_flow', 'dynamic_flow')}.py"
        code_path.write_text(dynamic_info["generated_code"], encoding="utf-8")
        print(f"Generated flow source: {code_path}")

    if hasattr(result, "model_dump"):
        result_data = result.model_dump()
    else:
        result_data = result

    print("Final result:")
    print(json.dumps(result_data, ensure_ascii=False, indent=2))
    print(f"Run trace Mermaid: {mermaid_path}")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await run_pipeline_gap_example()


if __name__ == "__main__":
    asyncio.run(main())
