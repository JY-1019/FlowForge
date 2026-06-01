"""Tests for the native ``doc-style-mirror`` skill: native skill discovery,
the SKILL.md packaging, and the ``extract_style`` style extractor.

Tests that need example office files generate them with the Pandoc engine and
skip when Pandoc (``pypandoc-binary``) is unavailable. The extractor itself has
no third-party dependency for ``.pptx`` / ``.docx``.
"""
import importlib.util
import sys

import pytest

from flowforge import AgentSkill
from flowforge.skills import (
    native_skill_names,
    native_skill_path,
    skill_path,
)
from flowforge.tools.doc_render import _pandoc_version, render_document

pandoc_required = pytest.mark.skipif(
    _pandoc_version() is None,
    reason="pandoc (pypandoc-binary) not installed",
)


def _load_extractor():
    """Import extract_style.py from the skill's scripts/ directory."""
    script = native_skill_path("doc-style-mirror") / "scripts" / "extract_style.py"
    spec = importlib.util.spec_from_file_location("_extract_style_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Native skill discovery + packaging
# ---------------------------------------------------------------------------

def test_native_skill_is_discoverable():
    assert "doc-style-mirror" in native_skill_names()
    path = native_skill_path("doc-style-mirror")
    assert (path / "SKILL.md").is_file()
    assert (path / "scripts" / "extract_style.py").is_file()


def test_skill_path_prefers_native():
    assert skill_path("doc-style-mirror") == native_skill_path("doc-style-mirror")


def test_unknown_native_skill_raises():
    with pytest.raises(FileNotFoundError):
        native_skill_path("does-not-exist")


def test_skill_loads_as_agent_skill():
    import flowforge.execution.llm as L

    skill = AgentSkill(path=str(native_skill_path("doc-style-mirror")))
    assert skill.name == "doc-style-mirror"
    block = L._load_agent_skill_prompt(skill)
    assert "doc-style-mirror" in block
    assert "mirror the theme" in block


# ---------------------------------------------------------------------------
# Style extractor
# ---------------------------------------------------------------------------

def test_extractor_rejects_unsupported_type(tmp_path):
    mod = _load_extractor()
    bogus = tmp_path / "note.txt"
    bogus.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.extract_style(bogus)


def test_build_html_css_from_profile():
    mod = _load_extractor()
    css = mod._build_html_css(
        {
            "palette": {"dark1": "#101820", "light1": "#FFFFFF", "accent1": "#C8102E"},
            "fonts": {"heading": "Pretendard", "body": "Pretendard"},
        }
    )
    assert "#C8102E" in css
    assert "Pretendard" in css
    assert "border-bottom: 3px solid #C8102E" in css


@pandoc_required
def test_extract_pptx_profile(tmp_path):
    mod = _load_extractor()
    example = tmp_path / "ex.pptx"
    render_document("# 회사 표지\n\n## 매출\n\n본문\n\n## 전략", "pptx", example)

    prof = mod.extract_style(example, html_theme=True)
    assert prof["type"] == "pptx"
    assert prof["palette"], "theme palette should be extracted"
    assert prof["fonts"].get("heading")
    assert prof["slide_count"] == 3
    assert len(prof["layouts"]) > 0
    # The heading outline (screen arrangement) is captured, Korean preserved.
    texts = [s["text"] for s in prof["sections"]]
    assert "회사 표지" in texts
    assert "html_css" in prof


@pandoc_required
def test_extract_docx_profile(tmp_path):
    mod = _load_extractor()
    example = tmp_path / "ex.docx"
    render_document("# 제목\n\n## 개요\n\n본문\n\n### 세부", "docx", example)

    prof = mod.extract_style(example)
    assert prof["type"] == "docx"
    assert prof["palette"]
    sections = [(s["level"], s["text"]) for s in prof["sections"]]
    assert (1, "제목") in sections
    assert (2, "개요") in sections


@pandoc_required
def test_extract_html_profile(tmp_path):
    mod = _load_extractor()
    example = tmp_path / "ex.html"
    render_document("# H1\n\n## H2\n\ntext", "html", example, theme="tech")

    prof = mod.extract_style(example, html_theme=True)
    assert prof["type"] == "html"
    assert prof["palette"]
    assert prof["fonts"]
    assert prof["html_css"]


# ---------------------------------------------------------------------------
# End-to-end: mirror an example into a new document
# ---------------------------------------------------------------------------

@pandoc_required
def test_mirror_pptx_via_reference_doc(tmp_path):
    example = tmp_path / "example.pptx"
    render_document("# 사내 표준", "pptx", example)

    out = tmp_path / "out.pptx"
    result = render_document(
        "# 2026 사업계획\n\n## 목표\n\n- 매출 +15%",
        "pptx",
        out,
        reference_doc=str(example),
    )
    assert result["ok"], result
    assert out.stat().st_size > 0


@pandoc_required
def test_mirror_html_via_custom_css(tmp_path):
    mod = _load_extractor()
    example = tmp_path / "example.html"
    render_document("# 표준\n\n## 소개", "html", example, theme="consulting")
    prof = mod.extract_style(example, html_theme=True)

    out = tmp_path / "out.html"
    result = render_document(
        "# 보고서\n\n## 요약", "html", out, custom_css=prof["html_css"]
    )
    assert result["ok"], result
    html = out.read_text(encoding="utf-8")
    assert "custom (extracted) overrides" in html
