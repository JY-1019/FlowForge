"""Tests for the dynamic paper report example helpers."""

from pathlib import Path

import pytest

from examples.dynamic_paper_report_agent import normalise_paper_payload
from flowforge.tools.builtin import _make_pptx_create_tool


def test_pptx_create_builtin_smoke(tmp_path: Path):
    """Verify the builtin pptx_create tool produces a valid file."""
    pytest.importorskip("pptx")

    tool = _make_pptx_create_tool(tmp_path)
    import json

    slides = [
        {"title": "표지", "bullets": ["트렌딩 논문 보고서", "2026-04-24"]},
        {"title": "요약", "bullets": ["핵심 포인트 1", "핵심 포인트 2"]},
    ]
    result = tool(path="report.pptx", slides=json.dumps(slides, ensure_ascii=False))

    assert result["ok"] is True
    assert result["native_objects"] is True
    assert (tmp_path / "report.pptx").exists()
    assert (tmp_path / "report.pptx").stat().st_size > 0


def test_normalise_paper_payload_from_wrapper_json():
    wrapped = {
        "papers_payload": """```json
        [
          {
            "rank": 1,
            "title": "Paper One",
            "authors": ["Author A"],
            "abstract_url": "https://arxiv.org/abs/1111.1111",
            "pdf_url": "https://arxiv.org/pdf/1111.1111",
            "summary_snippet": "Summary"
          }
        ]
        ```""",
        "source": "arxiv",
    }

    normalized = normalise_paper_payload(wrapped)

    assert normalized["papers"][0]["title"] == "Paper One"
    assert normalized["source_url"]
    assert normalized["fetched_at"]


def test_normalise_paper_payload_direct():
    direct = {
        "papers": [
            {
                "rank": 1,
                "title": "Direct Paper",
                "authors": [],
                "summary_snippet": "A summary",
            }
        ],
        "source_url": "https://arxiv.org/api/query",
        "fetched_at": "2026-04-25T00:00:00+00:00",
    }

    normalized = normalise_paper_payload(direct)

    assert normalized["papers"][0]["title"] == "Direct Paper"
    assert normalized["source_url"] == "https://arxiv.org/api/query"


def test_normalise_paper_payload_from_top_level_json_string():
    generated_shape = """```json
    {
      "papers": [
        {
          "title": "Generated Paper",
          "authors": "Author A, Author B",
          "abstract": "A generated abstract",
          "url": "https://arxiv.org/pdf/2222.2222",
          "published_date": "2026-04-25T00:00:00Z"
        }
      ]
    }
    ```"""

    normalized = normalise_paper_payload(generated_shape)

    paper = normalized["papers"][0]
    assert paper["rank"] == 1
    assert paper["title"] == "Generated Paper"
    assert paper["authors"] == ["Author A", "Author B"]
    assert paper["published"] == "2026-04-25T00:00:00Z"
    assert paper["summary_snippet"] == "A generated abstract"
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2222.2222"
    assert paper["abstract_url"] == "https://arxiv.org/abs/2222.2222"
