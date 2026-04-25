"""Built-in tools for dynamic FlowForge agents.

Two categories of tools are provided:

1. **Shell / runtime tools** — project inspection, test running, dependency
   checking.  Gated by ``DynamicRunOptions.allowed_shell_modes``.
2. **General-purpose utility tools** — ``web_fetch_url``, ``json_select_fields``,
   ``files_read_text``, ``files_write_text``, ``files_list_dir``.  Always
   included when ``include_builtin_tools=True`` (the default).  These are
   pure-Python tools that never shell out.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from flowforge.types import DynamicRunOptions, FunctionTool

_CONTROL_TOKENS = (";", "&&", "||", "|", ">", "<", "$(", "`")
_DEFAULT_MAX_OUTPUT_CHARS = 4000


def create_builtin_tool_pack(options: DynamicRunOptions) -> list[FunctionTool]:
    """Return builtin tools enabled by *options*.

    Shell tools are gated by ``options.allowed_shell_modes``.  General-purpose
    utility tools (web, json, files) are always included — they are safe,
    sandboxed, and high-reuse.
    """
    tools: list[FunctionTool] = []
    modes = set(options.allowed_shell_modes)

    # ── Always-on utility tools ───────────────────────────────────────
    tools.append(FunctionTool(
        func=_make_import_check_tool(),
        name="python_import_check",
        description=(
            "Check whether Python modules can be imported and report their "
            "installed package versions when available."
        ),
    ))
    tools.append(FunctionTool(
        func=_make_mcp_start_tool(options),
        name="mcp_start_server",
        description=(
            "Start an MCP server command declared in DynamicRunOptions."
        ),
    ))

    tools.append(FunctionTool(
        func=_make_pip_install_tool(options),
        name="pip_install",
        description=(
            "Install one or more Python packages using pip. Accepts a "
            "comma-separated or space-separated list of package names "
            "(e.g. 'python-pptx, httpx'). Installation is gated by the "
            "agent's DependencyPolicy."
        ),
    ))

    # General-purpose utility tools.
    tools.extend(_create_utility_tools(options))

    # Document processing tools (PDF, PPTX, CSV, DOCX, Markdown, Chart).
    tools.extend(_create_document_tools(options))

    # ── Shell tools (mode-gated) ──────────────────────────────────────
    if "readonly" in modes:
        tools.append(FunctionTool(
            func=_make_shell_tool(options, "readonly"),
            name="shell_readonly",
            description=(
                "Run a read-only shell command inside the project. Useful for "
                "pwd, ls, find, rg, cat, sed, head, tail, wc, and git inspect."
            ),
        ))
    if "project_exec" in modes:
        tools.append(FunctionTool(
            func=_make_shell_tool(options, "project_exec"),
            name="shell_project_exec",
            description=(
                "Run a project command such as python -m pytest, pytest, uv, "
                "npm, pnpm, or yarn inside the project."
            ),
        ))
    if "workspace_write" in modes:
        tools.append(FunctionTool(
            func=_make_shell_tool(options, "workspace_write"),
            name="shell_workspace_write",
            description=(
                "Run a limited workspace-writing shell command inside the "
                "project, such as mkdir, touch, cp, or mv."
            ),
        ))
    if "install_dependency" in modes:
        tools.append(FunctionTool(
            func=_make_shell_tool(options, "install_dependency"),
            name="shell_install_dependency",
            description=(
                "Install a dependency with pip, uv, npm, pnpm, or yarn when "
                "the dynamic dependency policy allows installation."
            ),
        ))

    return tools


# ---------------------------------------------------------------------------
# General-purpose utility tools (always included)
# ---------------------------------------------------------------------------

def _create_utility_tools(options: DynamicRunOptions) -> list[FunctionTool]:
    """Create pure-Python utility tools for web, JSON, and file operations."""
    project_root = Path(options.project_root or Path.cwd()).expanduser().resolve()
    max_output = max(500, options.shell_output_max_chars or _DEFAULT_MAX_OUTPUT_CHARS)

    return [
        FunctionTool(
            func=_make_web_fetch_url_tool(max_output),
            name="web_fetch_url",
            description=(
                "Fetch the content of a URL via HTTP GET and return the "
                "response body as text. Supports HTML, JSON, and plain text."
            ),
        ),
        FunctionTool(
            func=_make_json_select_fields_tool(),
            name="json_select_fields",
            description=(
                "Select specific top-level fields from a JSON string or dict. "
                "Returns a new dict containing only the requested keys."
            ),
        ),
        FunctionTool(
            func=_make_files_read_text_tool(project_root, max_output),
            name="files_read_text",
            description=(
                "Read a text file inside the project and return its contents. "
                "Path must be relative to the project root."
            ),
        ),
        FunctionTool(
            func=_make_files_write_text_tool(project_root),
            name="files_write_text",
            description=(
                "Write text content to a file inside the project. "
                "Path must be relative to the project root. Creates parent "
                "directories if they do not exist."
            ),
        ),
        FunctionTool(
            func=_make_files_list_dir_tool(project_root),
            name="files_list_dir",
            description=(
                "List files and directories at a path inside the project. "
                "Path must be relative to the project root."
            ),
        ),
    ]


def _make_web_fetch_url_tool(max_chars: int):
    def _tool(url: str, timeout_seconds: int = 30) -> dict[str, Any]:
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FlowForge/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                content_type = resp.headers.get("Content-Type", "")
                return {
                    "ok": True,
                    "status": resp.status,
                    "content_type": content_type,
                    "body": body[:max_chars],
                    "truncated": len(body) > max_chars,
                    "url": url,
                }
        except urllib.error.HTTPError as e:
            return {
                "ok": False,
                "status": e.code,
                "error": str(e),
                "url": url,
            }
        except Exception as e:
            return {
                "ok": False,
                "status": 0,
                "error": str(e),
                "url": url,
            }

    _tool.__name__ = "builtin_web_fetch_url"
    return _tool


def _make_json_select_fields_tool():
    import json as _json

    def _tool(data: str, fields: str) -> dict[str, Any]:
        try:
            if isinstance(data, str):
                parsed = _json.loads(data)
            else:
                parsed = data
        except (_json.JSONDecodeError, TypeError) as e:
            return {"ok": False, "error": f"Invalid JSON: {e}"}

        if not isinstance(parsed, dict):
            return {"ok": False, "error": "Input is not a JSON object"}

        field_list = [f.strip() for f in fields.split(",") if f.strip()]
        selected = {k: parsed[k] for k in field_list if k in parsed}
        missing = [k for k in field_list if k not in parsed]
        return {
            "ok": True,
            "selected": selected,
            "missing": missing,
        }

    _tool.__name__ = "builtin_json_select_fields"
    return _tool


def _make_files_read_text_tool(project_root: Path, max_chars: int):
    def _tool(path: str, encoding: str = "utf-8") -> dict[str, Any]:
        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        if not resolved.is_file():
            return {"ok": False, "error": f"File not found: {path}"}

        try:
            content = resolved.read_text(encoding=encoding)
            return {
                "ok": True,
                "content": content[:max_chars],
                "truncated": len(content) > max_chars,
                "path": path,
                "size": len(content),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    _tool.__name__ = "builtin_files_read_text"
    return _tool


def _make_files_write_text_tool(project_root: Path):
    def _tool(path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding=encoding)
            return {
                "ok": True,
                "path": path,
                "size": len(content),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    _tool.__name__ = "builtin_files_write_text"
    return _tool


def _make_files_list_dir_tool(project_root: Path):
    def _tool(path: str = ".") -> dict[str, Any]:
        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        if not resolved.is_dir():
            return {"ok": False, "error": f"Not a directory: {path}"}

        try:
            entries = []
            for item in sorted(resolved.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                })
            return {
                "ok": True,
                "path": path,
                "entries": entries[:200],
                "truncated": len(entries) > 200,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    _tool.__name__ = "builtin_files_list_dir"
    return _tool


# ---------------------------------------------------------------------------
# Document processing tools (PDF, PPTX, CSV, DOCX, Markdown, Chart)
# ---------------------------------------------------------------------------

def _create_document_tools(options: DynamicRunOptions) -> list[FunctionTool]:
    """Create document processing tools for reading/writing various formats."""
    project_root = Path(options.project_root or Path.cwd()).expanduser().resolve()
    max_output = max(500, options.shell_output_max_chars or _DEFAULT_MAX_OUTPUT_CHARS)

    return [
        FunctionTool(
            func=_make_pdf_read_text_tool(project_root, max_output),
            name="pdf_read_text",
            description=(
                "Extract text from a PDF file. Returns the text content of "
                "each page. Requires the 'pypdf' package (use pip_install "
                "to install if missing). Path must be relative to the project root."
            ),
        ),
        FunctionTool(
            func=_make_pptx_create_tool(project_root),
            name="pptx_create",
            description=(
                "Create a PowerPoint (.pptx) presentation from a JSON array "
                "of slide objects. Each slide supports: layout ('cover', "
                "'content', 'section', 'comparison', 'table', 'blank'), "
                "title, subtitle, body, bullets (list of strings), "
                "left/right (comparison columns with heading+bullets), "
                "table (headers+rows), image_path, speaker_note. "
                "Optional theme parameter: 'default' or 'dark'. "
                "Requires 'python-pptx' (use pip_install if missing). "
                "Path must be relative to the project root."
            ),
        ),
        FunctionTool(
            func=_make_csv_read_tool(project_root, max_output),
            name="csv_read",
            description=(
                "Read a CSV file and return its rows as a list of dicts "
                "(using the header row as keys). Path must be relative to "
                "the project root. No external dependencies required."
            ),
        ),
        FunctionTool(
            func=_make_csv_write_tool(project_root),
            name="csv_write",
            description=(
                "Write data to a CSV file. Accepts a list of dicts (each "
                "dict is a row, keys become column headers). Path must be "
                "relative to the project root. No external dependencies required."
            ),
        ),
        FunctionTool(
            func=_make_docx_create_tool(project_root),
            name="docx_create",
            description=(
                "Create a Word (.docx) document from structured content. "
                "Accepts a list of content blocks, each with 'type' "
                "('heading', 'paragraph', 'bullets') and 'text' or 'items'. "
                "Requires the 'python-docx' package (use pip_install to "
                "install if missing). Path must be relative to the project root."
            ),
        ),
        FunctionTool(
            func=_make_markdown_write_tool(project_root),
            name="markdown_write",
            description=(
                "Write a Markdown (.md) file. Accepts the markdown content "
                "as a string. Path must be relative to the project root. "
                "No external dependencies required."
            ),
        ),
        FunctionTool(
            func=_make_chart_create_tool(project_root),
            name="chart_create",
            description=(
                "Create a chart image (PNG) from data. Supports chart types: "
                "'bar', 'line', 'pie', 'scatter'. Accepts labels, values, "
                "title, xlabel, ylabel. Requires the 'matplotlib' package "
                "(use pip_install to install if missing). Path must be "
                "relative to the project root."
            ),
        ),
        FunctionTool(
            func=_make_image_create_tool(project_root),
            name="image_create",
            description=(
                "Create an image file. Two modes:\n"
                "1) Programmatic: draw text, rectangles, ellipses, lines on "
                "a canvas using the 'elements' JSON parameter. Requires "
                "'Pillow' package.\n"
                "2) AI-generated: set 'ai_prompt' to generate an image using "
                "OpenAI DALL-E 3 (requires OPENAI_API_KEY env var).\n"
                "Supports .png, .jpg, .webp output. Path must be relative "
                "to the project root."
            ),
        ),
    ]


def _make_pdf_read_text_tool(project_root: Path, max_chars: int):
    def _tool(path: str, pages: str = "") -> dict[str, Any]:
        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        if not resolved.is_file():
            return {"ok": False, "error": f"File not found: {path}"}

        try:
            import pypdf
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "The 'pypdf' package is not installed. "
                    "Use the pip_install tool to install it first: "
                    "pip_install(packages='pypdf')"
                ),
            }

        try:
            reader = pypdf.PdfReader(str(resolved))
            total_pages = len(reader.pages)

            if pages:
                page_indices = _parse_page_range(pages, total_pages)
            else:
                page_indices = list(range(total_pages))

            extracted: list[dict[str, Any]] = []
            total_len = 0
            for i in page_indices:
                text = reader.pages[i].extract_text() or ""
                total_len += len(text)
                extracted.append({"page": i + 1, "text": text})
                if total_len > max_chars:
                    break

            full_text = "\n".join(p["text"] for p in extracted)
            return {
                "ok": True,
                "path": path,
                "total_pages": total_pages,
                "pages_read": len(extracted),
                "text": full_text[:max_chars],
                "truncated": len(full_text) > max_chars,
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to read PDF: {e}"}

    _tool.__name__ = "builtin_pdf_read_text"
    return _tool


def _parse_page_range(pages_str: str, total: int) -> list[int]:
    """Parse page range string like '1-3,5,7-9' into 0-based indices."""
    indices: list[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            s = max(1, int(start.strip()))
            e = min(total, int(end.strip()))
            indices.extend(range(s - 1, e))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.append(idx)
    return indices


def _make_pptx_create_tool(project_root: Path):
    def _tool(path: str, slides: str, theme: str = "default") -> dict[str, Any]:
        """Create a PPTX presentation from a JSON array of slide objects.

        Each slide object supports the following fields:

        - ``layout`` (str): ``"cover"``, ``"content"`` (default), ``"section"``,
          ``"comparison"``, ``"table"``, ``"blank"``.
        - ``title`` (str): slide title.
        - ``subtitle`` (str): subtitle (cover/section slides only).
        - ``body`` (str): plain text body (content layout).
        - ``bullets`` (list[str]): bullet point list (content layout).
        - ``left``/``right`` (dict): comparison columns, each with ``heading``
          and ``bullets`` keys.
        - ``table`` (dict): ``{"headers": [...], "rows": [[...], ...]}``
          for table layout.
        - ``image_path`` (str): path to an image file (relative to project root)
          to embed in the slide.
        - ``speaker_note`` (str): speaker notes text.

        ``theme``: ``"default"`` or ``"dark"``.  Dark theme applies a dark
        background with white text.
        """
        import json as _json

        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        try:
            slide_data = _json.loads(slides) if isinstance(slides, str) else slides
        except (_json.JSONDecodeError, TypeError) as e:
            return {"ok": False, "error": f"Invalid slides JSON: {e}"}

        if not isinstance(slide_data, list):
            return {"ok": False, "error": "slides must be a JSON array of slide objects"}

        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt, Emu
            from pptx.enum.text import PP_ALIGN
            from pptx.dml.color import RGBColor
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "The 'python-pptx' package is not installed. "
                    "Use the pip_install tool to install it first: "
                    "pip_install(packages='python-pptx')"
                ),
            }

        try:
            prs = Presentation()
            is_dark = theme.lower() == "dark"

            for idx, slide_info in enumerate(slide_data):
                layout_name = slide_info.get("layout", "content")
                title = slide_info.get("title", "")
                subtitle = slide_info.get("subtitle", "")
                body = slide_info.get("body", "")
                bullets = slide_info.get("bullets", [])
                speaker_note = slide_info.get("speaker_note", "")
                image_path = slide_info.get("image_path", "")

                # Select layout index based on layout name.
                layout_map = {
                    "cover": 0,       # Title Slide
                    "content": 1,     # Title and Content
                    "section": 2,     # Section Header
                    "comparison": 3,  # Two Content
                    "blank": 6,       # Blank
                    "table": 1,       # Use content layout, add table manually
                }
                layout_idx = layout_map.get(layout_name, 1)
                # Clamp to available layouts.
                layout_idx = min(layout_idx, len(prs.slide_layouts) - 1)
                slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])

                # Dark theme: set background.
                if is_dark:
                    bg = slide.background
                    fill = bg.fill
                    fill.solid()
                    fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x2E)

                # Title.
                if slide.shapes.title:
                    slide.shapes.title.text = title
                    if is_dark:
                        for para in slide.shapes.title.text_frame.paragraphs:
                            for run in para.runs:
                                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

                # Cover / Section layout.
                if layout_name in ("cover", "section"):
                    if len(slide.placeholders) > 1:
                        ph = slide.placeholders[1]
                        if subtitle:
                            ph.text = subtitle
                        elif bullets:
                            ph.text = "\n".join(str(b) for b in bullets)
                        else:
                            ph.text = body
                        if is_dark:
                            for para in ph.text_frame.paragraphs:
                                for run in para.runs:
                                    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

                # Content layout with bullets or body.
                elif layout_name == "content" or layout_name not in layout_map:
                    if len(slide.placeholders) > 1:
                        ph = slide.placeholders[1]
                        if bullets:
                            tf = ph.text_frame
                            tf.clear()
                            for i, bullet in enumerate(bullets):
                                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                                p.text = str(bullet)
                                p.level = 0
                                if is_dark:
                                    for run in p.runs:
                                        run.font.color.rgb = RGBColor(0xE0, 0xE0, 0xE0)
                        else:
                            ph.text = body
                            if is_dark:
                                for para in ph.text_frame.paragraphs:
                                    for run in para.runs:
                                        run.font.color.rgb = RGBColor(0xE0, 0xE0, 0xE0)

                # Comparison layout (left/right columns).
                elif layout_name == "comparison":
                    left_data = slide_info.get("left", {})
                    right_data = slide_info.get("right", {})
                    for ph_idx, col_data in [(1, left_data), (2, right_data)]:
                        if ph_idx < len(slide.placeholders):
                            ph = slide.placeholders[ph_idx]
                            tf = ph.text_frame
                            tf.clear()
                            heading = col_data.get("heading", "")
                            if heading:
                                p = tf.paragraphs[0]
                                p.text = heading
                                p.font.bold = True
                                for b in col_data.get("bullets", []):
                                    p = tf.add_paragraph()
                                    p.text = str(b)
                                    p.level = 0
                            else:
                                for i, b in enumerate(col_data.get("bullets", [])):
                                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                                    p.text = str(b)

                # Table layout.
                elif layout_name == "table":
                    tbl_data = slide_info.get("table", {})
                    headers = tbl_data.get("headers", [])
                    rows = tbl_data.get("rows", [])
                    if headers:
                        n_rows = len(rows) + 1
                        n_cols = len(headers)
                        left = Inches(0.5)
                        top = Inches(2.0)
                        width = Inches(9.0)
                        height = Inches(0.4) * n_rows
                        table = slide.shapes.add_table(
                            n_rows, n_cols, left, top, width, height,
                        ).table
                        for i, h in enumerate(headers):
                            cell = table.cell(0, i)
                            cell.text = str(h)
                            for para in cell.text_frame.paragraphs:
                                para.font.bold = True
                                para.font.size = Pt(11)
                        for r_idx, row in enumerate(rows):
                            for c_idx, val in enumerate(row):
                                if c_idx < n_cols:
                                    cell = table.cell(r_idx + 1, c_idx)
                                    cell.text = str(val)
                                    for para in cell.text_frame.paragraphs:
                                        para.font.size = Pt(10)

                # Embed image if provided.
                if image_path:
                    try:
                        img_resolved = _resolve_safe_path(project_root, image_path)
                        if img_resolved.is_file():
                            slide.shapes.add_picture(
                                str(img_resolved),
                                Inches(1.0), Inches(2.5),
                                width=Inches(8.0),
                            )
                    except (ValueError, Exception):
                        pass  # Skip image on error.

                # Speaker notes.
                if speaker_note:
                    notes_slide = slide.notes_slide
                    notes_slide.notes_text_frame.text = speaker_note

            resolved.parent.mkdir(parents=True, exist_ok=True)
            prs.save(str(resolved))
            return {
                "ok": True,
                "path": path,
                "slide_count": len(slide_data),
                "size": resolved.stat().st_size,
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to create PPTX: {e}"}

    _tool.__name__ = "builtin_pptx_create"
    return _tool


def _make_csv_read_tool(project_root: Path, max_chars: int):
    def _tool(path: str, encoding: str = "utf-8", max_rows: int = 500) -> dict[str, Any]:
        import csv as _csv

        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        if not resolved.is_file():
            return {"ok": False, "error": f"File not found: {path}"}

        try:
            with resolved.open(encoding=encoding, newline="") as f:
                reader = _csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(dict(row))
                headers = reader.fieldnames or []
            return {
                "ok": True,
                "path": path,
                "headers": list(headers),
                "row_count": len(rows),
                "rows": rows,
                "truncated": len(rows) >= max_rows,
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to read CSV: {e}"}

    _tool.__name__ = "builtin_csv_read"
    return _tool


def _make_csv_write_tool(project_root: Path):
    def _tool(path: str, data: str, encoding: str = "utf-8") -> dict[str, Any]:
        import csv as _csv
        import json as _json

        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        try:
            rows = _json.loads(data) if isinstance(data, str) else data
        except (_json.JSONDecodeError, TypeError) as e:
            return {"ok": False, "error": f"Invalid data JSON: {e}"}

        if not isinstance(rows, list) or not rows:
            return {"ok": False, "error": "data must be a non-empty JSON array of objects"}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            headers = list(rows[0].keys())
            with resolved.open("w", encoding=encoding, newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            return {
                "ok": True,
                "path": path,
                "row_count": len(rows),
                "headers": headers,
                "size": resolved.stat().st_size,
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to write CSV: {e}"}

    _tool.__name__ = "builtin_csv_write"
    return _tool


def _make_docx_create_tool(project_root: Path):
    def _tool(path: str, content: str) -> dict[str, Any]:
        import json as _json

        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        try:
            blocks = _json.loads(content) if isinstance(content, str) else content
        except (_json.JSONDecodeError, TypeError) as e:
            return {"ok": False, "error": f"Invalid content JSON: {e}"}

        if not isinstance(blocks, list):
            return {"ok": False, "error": "content must be a JSON array of block objects"}

        try:
            from docx import Document
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "The 'python-docx' package is not installed. "
                    "Use the pip_install tool to install it first: "
                    "pip_install(packages='python-docx')"
                ),
            }

        try:
            doc = Document()
            for block in blocks:
                block_type = block.get("type", "paragraph")
                if block_type == "heading":
                    level = block.get("level", 1)
                    doc.add_heading(block.get("text", ""), level=level)
                elif block_type == "paragraph":
                    doc.add_paragraph(block.get("text", ""))
                elif block_type == "bullets":
                    for item in block.get("items", []):
                        doc.add_paragraph(str(item), style="List Bullet")
                elif block_type == "table":
                    rows_data = block.get("rows", [])
                    headers = block.get("headers", [])
                    if headers:
                        table = doc.add_table(rows=1, cols=len(headers))
                        table.style = "Table Grid"
                        for i, h in enumerate(headers):
                            table.rows[0].cells[i].text = str(h)
                        for row_data in rows_data:
                            row = table.add_row()
                            for i, val in enumerate(row_data):
                                if i < len(headers):
                                    row.cells[i].text = str(val)

            resolved.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(resolved))
            return {
                "ok": True,
                "path": path,
                "block_count": len(blocks),
                "size": resolved.stat().st_size,
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to create DOCX: {e}"}

    _tool.__name__ = "builtin_docx_create"
    return _tool


def _make_markdown_write_tool(project_root: Path):
    def _tool(path: str, content: str) -> dict[str, Any]:
        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "path": path,
                "size": len(content),
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to write Markdown: {e}"}

    _tool.__name__ = "builtin_markdown_write"
    return _tool


def _make_chart_create_tool(project_root: Path):
    def _tool(
        path: str,
        chart_type: str,
        labels: str,
        values: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> dict[str, Any]:
        import json as _json

        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        try:
            label_list = _json.loads(labels) if isinstance(labels, str) else labels
            value_list = _json.loads(values) if isinstance(values, str) else values
        except (_json.JSONDecodeError, TypeError) as e:
            return {"ok": False, "error": f"Invalid JSON for labels/values: {e}"}

        if chart_type not in ("bar", "line", "pie", "scatter"):
            return {
                "ok": False,
                "error": f"Unsupported chart_type: {chart_type!r}. "
                         f"Use 'bar', 'line', 'pie', or 'scatter'.",
            }

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "The 'matplotlib' package is not installed. "
                    "Use the pip_install tool to install it first: "
                    "pip_install(packages='matplotlib')"
                ),
            }

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            if chart_type == "bar":
                ax.bar(label_list, value_list)
            elif chart_type == "line":
                ax.plot(label_list, value_list, marker="o")
            elif chart_type == "pie":
                ax.pie(value_list, labels=label_list, autopct="%1.1f%%")
            elif chart_type == "scatter":
                ax.scatter(label_list, value_list)

            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)

            plt.tight_layout()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(resolved), dpi=150)
            plt.close(fig)

            return {
                "ok": True,
                "path": path,
                "chart_type": chart_type,
                "size": resolved.stat().st_size,
            }
        except Exception as e:
            plt.close("all")
            return {"ok": False, "error": f"Failed to create chart: {e}"}

    _tool.__name__ = "builtin_chart_create"
    return _tool


def _make_image_create_tool(project_root: Path):
    """Create images programmatically using Pillow.

    Supports: blank canvas, text rendering, basic shapes, compositing.
    For AI-generated images, uses OpenAI DALL-E when ``OPENAI_API_KEY`` is set.
    """

    def _tool(
        path: str,
        width: int = 1024,
        height: int = 768,
        background: str = "#FFFFFF",
        elements: str = "[]",
        ai_prompt: str = "",
        ai_provider: str = "openai",
        ai_size: str = "1024x1024",
    ) -> dict[str, Any]:
        """Create an image file.

        Parameters
        ----------
        path : str
            Output file path (relative to project root). Supports .png, .jpg, .webp.
        width / height : int
            Canvas dimensions (ignored when ``ai_prompt`` is set).
        background : str
            Background colour (hex like ``"#FF0000"`` or name like ``"white"``).
        elements : str
            JSON array of drawing elements. Each element is an object with:
            - ``"type"``: ``"text"``, ``"rectangle"``, ``"ellipse"``, ``"line"``
            - ``"text"`` type: ``text``, ``x``, ``y``, ``size`` (font size),
              ``color``, ``bold`` (bool)
            - ``"rectangle"`` type: ``x``, ``y``, ``w``, ``h``, ``color``,
              ``fill`` (optional fill colour)
            - ``"ellipse"`` type: ``x``, ``y``, ``w``, ``h``, ``color``, ``fill``
            - ``"line"`` type: ``x1``, ``y1``, ``x2``, ``y2``, ``color``,
              ``width`` (line width)
        ai_prompt : str
            When non-empty, generate the image using an AI model instead of
            drawing elements.  Requires ``OPENAI_API_KEY`` env var for OpenAI
            DALL-E.
        ai_provider : str
            AI provider: ``"openai"`` (default).
        ai_size : str
            AI image size: ``"1024x1024"``, ``"1792x1024"``, ``"1024x1792"``.
        """
        import json as _json

        try:
            resolved = _resolve_safe_path(project_root, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        # ── AI image generation ──────────────────────────────────────
        if ai_prompt:
            return _generate_ai_image(
                resolved, ai_prompt, ai_provider, ai_size,
            )

        # ── Programmatic image creation with Pillow ──────────────────
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "The 'Pillow' package is not installed. "
                    "Use the pip_install tool to install it first: "
                    "pip_install(packages='Pillow')"
                ),
            }

        try:
            element_list = _json.loads(elements) if isinstance(elements, str) else elements
        except (_json.JSONDecodeError, TypeError) as e:
            return {"ok": False, "error": f"Invalid elements JSON: {e}"}

        if not isinstance(element_list, list):
            element_list = []

        try:
            img = Image.new("RGBA", (width, height), background)
            draw = ImageDraw.Draw(img)

            for elem in element_list:
                etype = elem.get("type", "")

                if etype == "text":
                    text = elem.get("text", "")
                    x, y = elem.get("x", 0), elem.get("y", 0)
                    color = elem.get("color", "#000000")
                    font_size = elem.get("size", 24)
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except (OSError, IOError):
                        try:
                            font = ImageFont.truetype(
                                "/System/Library/Fonts/Helvetica.ttc",
                                font_size,
                            )
                        except (OSError, IOError):
                            font = ImageFont.load_default()
                    draw.text((x, y), text, fill=color, font=font)

                elif etype == "rectangle":
                    x, y = elem.get("x", 0), elem.get("y", 0)
                    w, h = elem.get("w", 100), elem.get("h", 100)
                    color = elem.get("color", "#000000")
                    fill = elem.get("fill", None)
                    draw.rectangle([x, y, x + w, y + h], outline=color, fill=fill)

                elif etype == "ellipse":
                    x, y = elem.get("x", 0), elem.get("y", 0)
                    w, h = elem.get("w", 100), elem.get("h", 100)
                    color = elem.get("color", "#000000")
                    fill = elem.get("fill", None)
                    draw.ellipse([x, y, x + w, y + h], outline=color, fill=fill)

                elif etype == "line":
                    x1, y1 = elem.get("x1", 0), elem.get("y1", 0)
                    x2, y2 = elem.get("x2", 100), elem.get("y2", 100)
                    color = elem.get("color", "#000000")
                    lw = elem.get("width", 2)
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=lw)

            # Save
            resolved.parent.mkdir(parents=True, exist_ok=True)
            # Convert RGBA to RGB for JPEG
            ext = resolved.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                img = img.convert("RGB")
            img.save(str(resolved))

            return {
                "ok": True,
                "path": path,
                "width": width,
                "height": height,
                "element_count": len(element_list),
                "size": resolved.stat().st_size,
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to create image: {e}"}

    _tool.__name__ = "builtin_image_create"
    return _tool


def _generate_ai_image(
    resolved: Path,
    prompt: str,
    provider: str,
    size: str,
) -> dict[str, Any]:
    """Generate an image using an AI provider API."""
    import os as _os

    if provider == "openai":
        api_key = _os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return {
                "ok": False,
                "error": (
                    "OPENAI_API_KEY environment variable is not set. "
                    "Set it to use AI image generation with OpenAI DALL-E."
                ),
            }

        try:
            import urllib.request
            import urllib.error
            import json as _json
            import base64

            req_body = _json.dumps({
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/images/generations",
                data=req_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read().decode("utf-8"))

            b64_data = result["data"][0]["b64_json"]
            img_bytes = base64.b64decode(b64_data)

            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(img_bytes)

            return {
                "ok": True,
                "path": str(resolved.name),
                "provider": "openai",
                "model": "dall-e-3",
                "size": size,
                "file_size": len(img_bytes),
                "revised_prompt": result["data"][0].get("revised_prompt", ""),
            }
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return {"ok": False, "error": f"OpenAI API error {e.code}: {body[:500]}"}
        except Exception as e:
            return {"ok": False, "error": f"AI image generation failed: {e}"}

    return {"ok": False, "error": f"Unsupported AI image provider: {provider!r}"}


def _resolve_safe_path(project_root: Path, path: str) -> Path:
    """Resolve a relative path, ensuring it stays inside project_root."""
    raw = Path(path).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"path must stay inside project root: {resolved}")
    return resolved


def _make_import_check_tool():
    def _tool(module_names: str, package_names: str = "") -> dict[str, Any]:
        import importlib.metadata
        import importlib.util

        modules = _split_names(module_names)
        packages = _split_names(package_names)
        results: list[dict[str, Any]] = []

        for index, module_name in enumerate(modules):
            package_name = packages[index] if index < len(packages) else module_name
            spec = importlib.util.find_spec(module_name)
            version = None
            if spec is not None:
                try:
                    version = importlib.metadata.version(package_name)
                except importlib.metadata.PackageNotFoundError:
                    version = None
            results.append({
                "module": module_name,
                "package": package_name,
                "available": spec is not None,
                "version": version,
            })

        return {"ok": all(item["available"] for item in results), "results": results}

    _tool.__name__ = "builtin_python_import_check"
    return _tool


def _make_pip_install_tool(options: DynamicRunOptions):
    def _tool(packages: str) -> dict[str, Any]:
        policy = options.dependency_policy
        if not policy.allow_install:
            return {
                "ok": False,
                "error": (
                    "Package installation is disabled. Set "
                    "DependencyPolicy(allow_install=True) to enable it."
                ),
            }
        if "pip" not in policy.allowed_managers:
            return {
                "ok": False,
                "error": "pip is not in allowed_managers.",
            }

        names = _split_names(packages)
        if not names:
            return {"ok": False, "error": "No package names provided."}

        # Validate against the dependency policy.
        if policy.allowed_packages:
            disallowed = [p for p in names if p not in policy.allowed_packages]
            if disallowed:
                return {
                    "ok": False,
                    "error": f"Packages not in allowed_packages: {disallowed}",
                }
        denied = [p for p in names if p in policy.denied_packages]
        if denied:
            return {
                "ok": False,
                "error": f"Packages denied by policy: {denied}",
            }

        import sys

        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *names]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=options.shell_timeout_seconds,
                check=False,
            )
            max_output = max(500, options.shell_output_max_chars or _DEFAULT_MAX_OUTPUT_CHARS)
            return {
                "ok": completed.returncode == 0,
                "packages": names,
                "returncode": completed.returncode,
                "stdout": _trim_output(completed.stdout, max_output),
                "stderr": _trim_output(completed.stderr, max_output),
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "packages": names,
                "error": "pip install timed out",
            }

    _tool.__name__ = "builtin_pip_install"
    return _tool


def _make_mcp_start_tool(options: DynamicRunOptions):
    def _tool(server_name: str, cwd: str | None = None) -> dict[str, Any]:
        command = options.mcp_server_commands.get(server_name)
        if not command:
            return {
                "ok": False,
                "server_name": server_name,
                "stderr": (
                    "MCP server is not declared in DynamicRunOptions."
                ),
            }

        project_root = Path(options.project_root or Path.cwd()).expanduser().resolve()
        run_cwd = _resolve_cwd(project_root, cwd)
        proc = subprocess.Popen(
            command,
            cwd=run_cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(min(options.mcp_start_timeout_seconds, 2))
        running = proc.poll() is None
        return {
            "ok": running,
            "server_name": server_name,
            "pid": proc.pid,
            "command": command,
            "cwd": str(run_cwd),
            "stderr": "" if running else "MCP server command exited immediately.",
        }

    _tool.__name__ = "builtin_mcp_start_server"
    return _tool


def _make_shell_tool(options: DynamicRunOptions, mode: str):
    def _tool(
        command: str,
        cwd: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        return _run_shell_command(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            mode=mode,
            options=options,
        )

    _tool.__name__ = f"builtin_{mode}"
    return _tool


def _split_names(value: str) -> list[str]:
    return [
        item.strip()
        for chunk in value.split(",")
        for item in chunk.split()
        if item.strip()
    ]


def _run_shell_command(
    *,
    command: str,
    cwd: str | None,
    timeout_seconds: int | None,
    mode: str,
    options: DynamicRunOptions,
) -> dict[str, Any]:
    project_root = Path(options.project_root or Path.cwd()).expanduser().resolve()
    run_cwd = _resolve_cwd(project_root, cwd)
    timeout = timeout_seconds or options.shell_timeout_seconds

    error = _validate_command(command, mode, options)
    if error:
        return {
            "ok": False,
            "returncode": 126,
            "stdout": "",
            "stderr": error,
            "command": command,
            "cwd": str(run_cwd),
            "mode": mode,
        }

    completed = subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=run_cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    max_output = max(500, options.shell_output_max_chars or _DEFAULT_MAX_OUTPUT_CHARS)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": _trim_output(completed.stdout, max_output),
        "stderr": _trim_output(completed.stderr, max_output),
        "truncated": (
            len(completed.stdout) > max_output or len(completed.stderr) > max_output
        ),
        "command": command,
        "cwd": str(run_cwd),
        "mode": mode,
    }


def _trim_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        f"[truncated to last {max_chars} chars]\n"
        + text[-max_chars:]
    )


def _resolve_cwd(project_root: Path, cwd: str | None) -> Path:
    raw = Path(cwd).expanduser() if cwd else project_root
    resolved = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"cwd must stay inside project_root: {resolved}") from exc
    return resolved


def _validate_command(
    command: str,
    mode: str,
    options: DynamicRunOptions,
) -> str | None:
    stripped = command.strip()
    if not stripped:
        return "command must not be empty"
    if any(token in stripped for token in _CONTROL_TOKENS):
        return "shell control operators are not allowed in builtin shell tools"

    try:
        parts = shlex.split(stripped)
    except ValueError as exc:
        return f"invalid shell command: {exc}"
    if not parts:
        return "command must not be empty"

    executable = parts[0]
    if mode == "readonly":
        return _validate_readonly(parts)
    if mode == "project_exec":
        if executable in {"python", "python3", "pytest", "uv", "npm", "pnpm", "yarn"}:
            return None
        return f"command {executable!r} is not allowed for project_exec"
    if mode == "workspace_write":
        if executable in {"mkdir", "touch", "cp", "mv"}:
            return None
        return f"command {executable!r} is not allowed for workspace_write"
    if mode == "install_dependency":
        return _validate_install(parts, options)
    return f"unknown shell mode: {mode}"


def _validate_readonly(parts: list[str]) -> str | None:
    executable = parts[0]
    if executable in {"pwd", "ls", "find", "rg", "grep", "cat", "sed", "awk", "head", "tail", "wc"}:
        return None
    if executable == "git":
        subcommand = parts[1] if len(parts) > 1 else ""
        if subcommand in {"status", "diff", "log", "show", "branch"}:
            return None
        return f"git subcommand {subcommand!r} is not read-only allowlisted"
    return f"command {executable!r} is not allowed for readonly"


def _validate_install(parts: list[str], options: DynamicRunOptions) -> str | None:
    policy = options.dependency_policy
    if not policy.allow_install:
        return "dependency installation is disabled by DynamicRunOptions"

    manager = parts[0]
    effective_manager = manager
    if manager in {"python", "python3"} and parts[1:4] == ["-m", "pip", "install"]:
        effective_manager = "pip"

    if effective_manager not in policy.allowed_managers:
        return f"dependency manager {manager!r} is not allowed"

    if effective_manager in {"pip", "uv"} and "install" not in parts and "add" not in parts:
        return f"{effective_manager} command must use install/add for dependency installation"
    if effective_manager in {"npm", "pnpm", "yarn"} and not any(
        token in parts for token in {"install", "add"}
    ):
        return f"{effective_manager} command must use install/add for dependency installation"

    requested = _extract_requested_packages(parts)
    if policy.allowed_packages:
        disallowed = [pkg for pkg in requested if pkg not in policy.allowed_packages]
        if disallowed:
            return f"packages are not in allowed_packages: {disallowed}"
    denied = [pkg for pkg in requested if pkg in policy.denied_packages]
    if denied:
        return f"packages are denied by dependency policy: {denied}"
    return None


def _extract_requested_packages(parts: list[str]) -> list[str]:
    packages: list[str] = []
    skip_next = False
    for token in parts[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in {"install", "add", "-m", "pip"}:
            continue
        if token.startswith("-"):
            if token in {"-r", "--requirement"}:
                skip_next = True
            continue
        packages.append(token.split("==", 1)[0].split(">=", 1)[0])
    return packages
