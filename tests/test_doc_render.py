"""Smoke tests for the Pandoc-backed document rendering engine and the
``doc_render`` / ``html_create`` builtin tools.

All tests skip when Pandoc (``pypandoc-binary``) is not installed, so the
suite stays green in minimal environments.
"""
import pytest

from flowforge.tools.doc_render import (
    SUPPORTED_FORMATS,
    THEME_NAMES,
    _pandoc_version,
    render_document,
)

pandoc_required = pytest.mark.skipif(
    _pandoc_version() is None,
    reason="pandoc (pypandoc-binary) not installed",
)

_SAMPLE = """# Quarterly Review

## Revenue

- Up 12% YoY
- New markets opened

## Outlook

| Metric | Value |
|--------|-------|
| ARR    | $4.2M |

Strong **momentum** going into next quarter.
"""


def test_engine_constants():
    assert SUPPORTED_FORMATS == ("html", "docx", "pptx", "md")
    assert "default" in THEME_NAMES
    assert "tech" in THEME_NAMES


@pandoc_required
@pytest.mark.parametrize("fmt", ["html", "docx", "pptx", "md"])
def test_render_document_each_format(tmp_path, fmt):
    out = tmp_path / f"out.{fmt}"
    result = render_document(_SAMPLE, fmt, out, theme="consulting")
    assert result["ok"], result
    assert result["to"] == fmt
    assert result["engine"] == "pandoc"
    assert out.exists()
    assert out.stat().st_size > 0


@pandoc_required
def test_render_html_theme_and_title(tmp_path):
    out = tmp_path / "page.html"
    result = render_document(
        _SAMPLE, "html", out, theme="tech", toc=True, title="My Report"
    )
    assert result["ok"], result
    assert result["theme"] == "tech"
    html = out.read_text(encoding="utf-8")
    assert "My Report" in html
    # CSS from the chosen theme is injected into the document head.
    assert "<style>" in html


def test_render_rejects_unsupported_format(tmp_path):
    out = tmp_path / "out.xyz"
    result = render_document(_SAMPLE, "xyz", out)
    assert result["ok"] is False
    assert "Unsupported target format" in result["error"]


@pandoc_required
def test_render_unknown_theme_falls_back_to_default(tmp_path):
    out = tmp_path / "out.html"
    result = render_document(_SAMPLE, "html", out, theme="does-not-exist")
    assert result["ok"], result
    assert result["theme"] == "default"


# ---------------------------------------------------------------------------
# Builtin tool wiring
# ---------------------------------------------------------------------------

def _doc_tools(tmp_path):
    from flowforge.types import DynamicRunOptions
    from flowforge.tools.builtin import _create_document_tools

    opts = DynamicRunOptions(project_root=str(tmp_path))
    return {t.name: t for t in _create_document_tools(opts)}


def test_new_tools_registered(tmp_path):
    tools = _doc_tools(tmp_path)
    assert "doc_render" in tools
    assert "html_create" in tools
    # Legacy tools remain available for back-compat.
    assert {"pptx_create", "docx_create", "markdown_write"} <= set(tools)


@pandoc_required
@pytest.mark.parametrize(
    "filename,fmt",
    [("deck.pptx", "pptx"), ("report.docx", "docx"), ("page.html", "html"),
     ("notes.md", "md")],
)
def test_doc_render_tool_by_extension(tmp_path, filename, fmt):
    tool = _doc_tools(tmp_path)["doc_render"].func
    result = tool(path=filename, source=_SAMPLE, theme="academic")
    assert result["ok"], result
    assert result["to"] == fmt
    assert result["path"] == filename
    assert (tmp_path / filename).exists()


@pandoc_required
def test_doc_render_tool_explicit_to_overrides_extension(tmp_path):
    tool = _doc_tools(tmp_path)["doc_render"].func
    result = tool(path="artifact.out", source=_SAMPLE, to="html")
    assert result["ok"], result
    assert result["to"] == "html"


@pandoc_required
def test_html_create_tool(tmp_path):
    tool = _doc_tools(tmp_path)["html_create"].func
    result = tool(path="page.html", source=_SAMPLE, theme="dark", toc=True)
    assert result["ok"], result
    assert result["theme"] == "dark"
    assert (tmp_path / "page.html").exists()


def test_doc_render_tool_rejects_path_outside_root(tmp_path):
    tool = _doc_tools(tmp_path)["doc_render"].func
    result = tool(path="../escape.html", source=_SAMPLE)
    assert result["ok"] is False
