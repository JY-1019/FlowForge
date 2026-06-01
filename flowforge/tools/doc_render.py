"""Pandoc-backed document rendering engine.

A single Markdown source is converted into professional ``html`` / ``docx`` /
``pptx`` / ``md`` artifacts via `Pandoc <https://pandoc.org/>`_, the de-facto
universal document converter.  PPTX/DOCX output is **native and editable**
(not rasterised images), and HTML output ships with built-in CSS themes and
syntax highlighting.

Pandoc is provided by the ``pypandoc-binary`` PyPI package, which bundles a
prebuilt ``pandoc`` binary — no system-level install required.  Install with::

    pip install "flowforge[pandoc]"        # or: pip install pypandoc-binary

The public entry point is :func:`render_document`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.types import DynamicRunOptions


# Formats Pandoc can emit from a Markdown source through this engine.
SUPPORTED_FORMATS = ("html", "docx", "pptx", "md")

# Pandoc input flavour.  The ``markdown`` superset handles pipe tables, fenced
# code, footnotes, and the headings used to split slides.
_INPUT_FORMAT = "markdown"


# ---------------------------------------------------------------------------
# Built-in HTML themes (CSS injected into a standalone Pandoc HTML document)
# ---------------------------------------------------------------------------

_BASE_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0 auto; max-width: 820px; padding: 3rem 1.25rem 6rem;
  line-height: 1.65; }
img { max-width: 100%; }
pre { padding: 1rem; border-radius: 8px; overflow-x: auto; }
code { font-size: 0.92em; }
table { border-collapse: collapse; width: 100%; margin: 1.25rem 0; }
th, td { border: 1px solid; padding: 0.5rem 0.75rem; text-align: left; }
blockquote { margin: 1.25rem 0; padding: 0.25rem 1rem; border-left: 4px solid; }
h1, h2, h3 { line-height: 1.25; }
"""

_HTML_THEMES: dict[str, str] = {
    "default": _BASE_CSS + """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  color: #1f2328; background: #ffffff; }
a { color: #0969da; }
pre, code { background: #f6f8fa; }
pre code { background: transparent; }
th { background: #f6f8fa; }
th, td { border-color: #d0d7de; }
blockquote { border-color: #d0d7de; color: #57606a; }
h1, h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }
""",
    "dark": _BASE_CSS + """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  color: #e6edf3; background: #0d1117; }
a { color: #4493f8; }
pre, code { background: #161b22; }
pre code { background: transparent; }
th { background: #161b22; }
th, td { border-color: #30363d; }
blockquote { border-color: #30363d; color: #8b949e; }
h1, h2 { border-bottom: 1px solid #30363d; padding-bottom: 0.3rem; }
""",
    "consulting": _BASE_CSS + """
body { font-family: "Segoe UI", "Helvetica Neue", Arial,
  "Malgun Gothic", sans-serif; color: #1a2332; background: #ffffff;
  max-width: 880px; }
a { color: #c8102e; }
pre, code { background: #f2f4f7; }
pre code { background: transparent; }
th { background: #1a2332; color: #ffffff; }
th, td { border-color: #c9d2dd; }
blockquote { border-color: #c8102e; color: #3a4452; font-style: italic; }
h1 { color: #1a2332; border-bottom: 3px solid #c8102e; padding-bottom: 0.4rem; }
h2 { color: #1a2332; }
h3 { color: #c8102e; }
""",
    "academic": _BASE_CSS + """
body { font-family: Georgia, "Times New Roman", "Batang", serif;
  color: #222; background: #fdfdfb; max-width: 720px; line-height: 1.8; }
a { color: #8b0000; }
pre, code { background: #f3f1ea; font-family: "SF Mono", Menlo, monospace; }
pre code { background: transparent; }
th { background: #efece2; }
th, td { border-color: #d8d3c4; }
blockquote { border-color: #8b0000; color: #444; font-style: italic; }
h1 { text-align: center; }
h1, h2, h3 { font-family: Georgia, serif; }
""",
    "tech": _BASE_CSS + """
body { font-family: "Inter", -apple-system, "Segoe UI", Roboto,
  "Malgun Gothic", sans-serif; color: #0f172a; background: #f8fafc;
  max-width: 860px; }
a { color: #2563eb; }
pre, code { background: #0f172a; color: #e2e8f0; }
pre code { background: transparent; }
th { background: #1e293b; color: #f1f5f9; }
th, td { border-color: #cbd5e1; }
blockquote { border-color: #2563eb; color: #334155; }
h1 { color: #0f172a; }
h2 { color: #1e293b; border-bottom: 2px solid #2563eb; padding-bottom: 0.3rem; }
h3 { color: #2563eb; }
""",
}

THEME_NAMES = tuple(_HTML_THEMES.keys())


# ---------------------------------------------------------------------------
# Pandoc availability
# ---------------------------------------------------------------------------

def ensure_pandoc(options: "DynamicRunOptions | None" = None) -> dict[str, Any]:
    """Ensure ``pypandoc`` and the ``pandoc`` binary are importable.

    Returns a dict ``{"ok": bool, ...}``.  When pandoc is missing and
    ``options.dependency_policy.allow_install`` is set, ``pypandoc-binary`` is
    installed automatically and the result records the install attempt.
    """
    version = _pandoc_version()
    if version is not None:
        return {"ok": True, "pandoc_version": version}

    if options is not None and options.dependency_policy.allow_install:
        from flowforge.tools.builtin import _make_pip_install_tool

        install_attempt = _make_pip_install_tool(options)("pypandoc-binary")
        if install_attempt.get("ok"):
            version = _pandoc_version()
            if version is not None:
                return {
                    "ok": True,
                    "pandoc_version": version,
                    "installed_dependency": True,
                    "install_attempt": install_attempt,
                }
        return {
            "ok": False,
            "error": (
                "pandoc is unavailable and automatic installation of "
                "'pypandoc-binary' failed: "
                f"{install_attempt.get('error') or install_attempt.get('stderr')}"
            ),
            "install_attempt": install_attempt,
        }

    return {
        "ok": False,
        "error": (
            "pandoc is not available. Install it with "
            "`pip install \"flowforge[pandoc]\"` (or `pip install "
            "pypandoc-binary`), or enable DependencyPolicy(allow_install=True) "
            "so FlowForge can install it for you."
        ),
    }


def _pandoc_version() -> str | None:
    try:
        import pypandoc
    except ImportError:
        return None
    try:
        return str(pypandoc.get_pandoc_version())
    except (OSError, RuntimeError):
        # pypandoc is importable but no pandoc binary is reachable.
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_document(
    source: str,
    to: str,
    output_path: Path,
    *,
    theme: str = "default",
    standalone: bool = True,
    toc: bool = False,
    slide_level: int | None = None,
    reference_doc: str | None = None,
    title: str | None = None,
    custom_css: str | None = None,
    options: "DynamicRunOptions | None" = None,
) -> dict[str, Any]:
    """Render a Markdown ``source`` to ``to`` at ``output_path`` via Pandoc.

    Parameters
    ----------
    source:
        Markdown text.
    to:
        One of :data:`SUPPORTED_FORMATS` (``html`` / ``docx`` / ``pptx`` / ``md``).
    output_path:
        Destination file (already validated/safe).
    theme:
        HTML CSS theme name (see :data:`THEME_NAMES`).  Ignored for non-HTML
        output unless a ``reference_doc`` is supplied.
    standalone:
        Emit a complete document (HTML ``<head>``/``<body>``).  Always true for
        ``docx``/``pptx``.
    toc:
        Insert a table of contents.
    slide_level:
        For ``pptx``: heading level that starts a new slide (Pandoc default
        when ``None``).
    reference_doc:
        Path to a ``.docx``/``.pptx`` template used to style ``docx``/``pptx``
        output (Pandoc ``--reference-doc``).
    title:
        Document title metadata (defaults to the output file stem).
    custom_css:
        For ``html`` output, extra CSS layered on top of the base stylesheet
        (overrides the named ``theme``'s accent rules).  Use this to apply a
        palette/font set extracted from an example document.
    """
    fmt = (to or "").strip().lower()
    if fmt in ("markdown", "gfm"):
        fmt = "md"
    if fmt not in SUPPORTED_FORMATS:
        return {
            "ok": False,
            "error": (
                f"Unsupported target format {to!r}. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}."
            ),
        }

    ready = ensure_pandoc(options)
    if not ready.get("ok"):
        return ready

    import pypandoc

    pandoc_to = "gfm" if fmt == "md" else fmt
    extra_args: list[str] = []
    header_file: str | None = None

    try:
        if toc:
            extra_args.append("--toc")

        if fmt == "html":
            if standalone:
                extra_args.append("--standalone")
            extra_args.append("--highlight-style=tango")
            doc_title = title or output_path.stem or "Document"
            extra_args += ["--metadata", f"title={doc_title}"]
            css = _HTML_THEMES.get(theme) or _HTML_THEMES["default"]
            if custom_css:
                css = css + "\n/* custom (extracted) overrides */\n" + custom_css
            header_file = _write_temp_header(css)
            extra_args += ["--include-in-header", header_file]
        else:
            if reference_doc:
                extra_args.append(f"--reference-doc={reference_doc}")
            if fmt == "pptx" and slide_level is not None:
                extra_args.append(f"--slide-level={slide_level}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pypandoc.convert_text(
            source,
            to=pandoc_to,
            format=_INPUT_FORMAT,
            outputfile=str(output_path),
            extra_args=extra_args,
        )
    except (RuntimeError, OSError) as e:
        return {"ok": False, "error": f"Pandoc conversion failed: {e}"}
    finally:
        if header_file is not None:
            Path(header_file).unlink(missing_ok=True)

    size = output_path.stat().st_size if output_path.exists() else 0
    if size == 0:
        return {"ok": False, "error": "Pandoc produced an empty file."}

    result: dict[str, Any] = {
        "ok": True,
        "to": fmt,
        "engine": "pandoc",
        "pandoc_version": ready.get("pandoc_version"),
        "size": size,
    }
    if fmt == "html":
        result["theme"] = theme if theme in _HTML_THEMES else "default"
    if ready.get("installed_dependency"):
        result["installed_dependency"] = True
    return result


def _write_temp_header(css: str) -> str:
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    try:
        fh.write(f"<style>\n{css}\n</style>\n")
        return fh.name
    finally:
        fh.close()
