#!/usr/bin/env python3
"""Extract a reusable *style profile* from an example office/web document.

Given a company ``.pptx`` / ``.docx`` / ``.html`` example, emit a JSON profile
describing its visual identity so a new document can be generated to match:

* ``palette``  - theme colour scheme (dark/light/accents) as ``#RRGGBB``
* ``fonts``    - heading + body typefaces
* ``layouts``  - (pptx) slide-layout inventory with placeholder types
* ``sections`` - heading outline (the "screen arrangement" / structure)
* ``html_css`` - a ready-to-use CSS block built from the palette + fonts,
                 suitable to pass to ``doc_render`` / ``html_create`` as
                 ``custom_css`` (only emitted with ``--html-theme``)

Office files are parsed straight from their OOXML zip with the standard
library, so this script needs **no third-party packages** for ``.pptx`` /
``.docx``.  HTML parsing uses ``beautifulsoup4`` when available and falls back
to regex otherwise.

Usage::

    python extract_style.py company_deck.pptx
    python extract_style.py report.docx --html-theme
    python extract_style.py landing.html --html-theme
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Theme colour-scheme child order → friendly names.
_CLR_SLOTS = [
    ("dk1", "dark1"),
    ("lt1", "light1"),
    ("dk2", "dark2"),
    ("lt2", "light2"),
    ("accent1", "accent1"),
    ("accent2", "accent2"),
    ("accent3", "accent3"),
    ("accent4", "accent4"),
    ("accent5", "accent5"),
    ("accent6", "accent6"),
    ("hlink", "hyperlink"),
    ("folHlink", "followed_hyperlink"),
]


# ---------------------------------------------------------------------------
# OOXML (pptx/docx) helpers
# ---------------------------------------------------------------------------

def _import_et():
    import xml.etree.ElementTree as ET
    return ET


def _color_from_node(node) -> str | None:
    """Return ``#RRGGBB`` from an a:srgbClr / a:sysClr child, if present."""
    srgb = node.find(f"{_A}srgbClr")
    if srgb is not None and srgb.get("val"):
        return "#" + srgb.get("val").upper()
    sysclr = node.find(f"{_A}sysClr")
    if sysclr is not None and sysclr.get("lastClr"):
        return "#" + sysclr.get("lastClr").upper()
    return None


def _parse_theme_xml(data: bytes) -> dict[str, Any]:
    ET = _import_et()
    root = ET.fromstring(data)
    elements = root.find(f"{_A}themeElements")
    palette: dict[str, str] = {}
    fonts: dict[str, str] = {}
    if elements is None:
        return {"palette": palette, "fonts": fonts}

    clr = elements.find(f"{_A}clrScheme")
    if clr is not None:
        for tag, friendly in _CLR_SLOTS:
            child = clr.find(f"{_A}{tag}")
            if child is not None:
                hexval = _color_from_node(child)
                if hexval:
                    palette[friendly] = hexval

    font_scheme = elements.find(f"{_A}fontScheme")
    if font_scheme is not None:
        major = font_scheme.find(f"{_A}majorFont/{_A}latin")
        minor = font_scheme.find(f"{_A}minorFont/{_A}latin")
        if major is not None and major.get("typeface"):
            fonts["heading"] = major.get("typeface")
        if minor is not None and minor.get("typeface"):
            fonts["body"] = minor.get("typeface")
    return {"palette": palette, "fonts": fonts}


def _extract_pptx(path: Path, ET) -> dict[str, Any]:
    profile: dict[str, Any] = {"type": "pptx"}
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        theme_name = next(
            (n for n in sorted(names) if n.startswith("ppt/theme/theme")), None
        )
        if theme_name:
            profile.update(_parse_theme_xml(zf.read(theme_name)))

        # Slide-layout inventory: name + placeholder types.
        layouts: list[dict[str, Any]] = []
        for n in sorted(n for n in names if
                        re.match(r"ppt/slideLayouts/slideLayout\d+\.xml$", n)):
            root = ET.fromstring(zf.read(n))
            csld = root.find(f"{_P}cSld")
            name = csld.get("name") if csld is not None else None
            ph_types = []
            for ph in root.iter(f"{_P}ph"):
                ph_types.append(ph.get("type") or "body")
            layouts.append({
                "name": name or Path(n).stem,
                "type": root.get("type"),
                "placeholders": ph_types,
            })
        profile["layouts"] = layouts

        # Slide outline: titles in document order.
        slide_files = sorted(
            (n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)),
            key=lambda s: int(re.search(r"(\d+)", s).group(1)),
        )
        sections: list[dict[str, Any]] = []
        for n in slide_files:
            root = ET.fromstring(zf.read(n))
            title = _pptx_slide_title(root)
            sections.append({"level": 1, "text": title or "(untitled slide)"})
        profile["slide_count"] = len(slide_files)
        profile["sections"] = sections
    return profile


def _pptx_slide_title(root) -> str | None:
    for sp in root.iter(f"{_P}sp"):
        ph = sp.find(f".//{_P}nvSpPr/{_P}nvPr/{_P}ph")
        if ph is not None and (ph.get("type") in ("title", "ctrTitle")):
            texts = [t.text or "" for t in sp.iter(f"{_A}t")]
            joined = "".join(texts).strip()
            if joined:
                return joined
    return None


def _extract_docx(path: Path, ET) -> dict[str, Any]:
    profile: dict[str, Any] = {"type": "docx"}
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        if "word/theme/theme1.xml" in names:
            profile.update(_parse_theme_xml(zf.read("word/theme/theme1.xml")))

        # Heading style fonts/sizes from styles.xml.
        heading_styles: dict[str, dict[str, Any]] = {}
        if "word/styles.xml" in names:
            root = ET.fromstring(zf.read("word/styles.xml"))
            for style in root.iter(f"{_W}style"):
                name_el = style.find(f"{_W}name")
                sid = (name_el.get(f"{_W}val") if name_el is not None else "") or ""
                if not sid.lower().startswith("heading"):
                    continue
                rpr = style.find(f"{_W}rPr")
                info: dict[str, Any] = {}
                if rpr is not None:
                    fonts_el = rpr.find(f"{_W}rFonts")
                    if fonts_el is not None and fonts_el.get(f"{_W}ascii"):
                        info["font"] = fonts_el.get(f"{_W}ascii")
                    sz = rpr.find(f"{_W}sz")
                    if sz is not None and sz.get(f"{_W}val"):
                        info["size_pt"] = int(sz.get(f"{_W}val")) / 2
                    color = rpr.find(f"{_W}color")
                    if color is not None and color.get(f"{_W}val"):
                        val = color.get(f"{_W}val")
                        if val and val != "auto":
                            info["color"] = "#" + val.upper()
                heading_styles[sid] = info
        profile["heading_styles"] = heading_styles

        # Heading outline from document.xml.
        sections: list[dict[str, Any]] = []
        if "word/document.xml" in names:
            root = ET.fromstring(zf.read("word/document.xml"))
            for para in root.iter(f"{_W}p"):
                ppr = para.find(f"{_W}pPr")
                if ppr is None:
                    continue
                pstyle = ppr.find(f"{_W}pStyle")
                sid = (pstyle.get(f"{_W}val") if pstyle is not None else "") or ""
                m = re.match(r"[Hh]eading(\d+)", sid)
                if not m:
                    continue
                text = "".join(t.text or "" for t in para.iter(f"{_W}t")).strip()
                if text:
                    sections.append({"level": int(m.group(1)), "text": text})
        profile["sections"] = sections
    return profile


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b")
_FONT_RE = re.compile(r"font-family\s*:\s*([^;}{]+)", re.IGNORECASE)


def _extract_html(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    profile: dict[str, Any] = {"type": "html"}

    # Palette: most frequent hex colours.
    counts: dict[str, int] = {}
    for hexval in _HEX_RE.findall(text):
        key = hexval.upper()
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts, key=lambda c: counts[c], reverse=True)
    palette: dict[str, str] = {}
    for i, hexval in enumerate(ranked[:8]):
        palette[f"color{i + 1}"] = hexval
    profile["palette"] = palette

    # Fonts: first font-family stacks seen.
    families = []
    for stack in _FONT_RE.findall(text):
        first = stack.split(",")[0].strip().strip("'\"")
        if first and first not in families:
            families.append(first)
    fonts: dict[str, str] = {}
    if families:
        fonts["heading"] = families[0]
        fonts["body"] = families[1] if len(families) > 1 else families[0]
    profile["fonts"] = fonts

    # Section outline: heading tags in document order.
    sections: list[dict[str, Any]] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(text, "html.parser")
        for el in soup.find_all(re.compile(r"^h[1-6]$")):
            sections.append({
                "level": int(el.name[1]),
                "text": el.get_text(strip=True),
            })
    except Exception:
        for level, body in re.findall(
            r"<h([1-6])[^>]*>(.*?)</h\1>", text, re.IGNORECASE | re.DOTALL
        ):
            clean = re.sub(r"<[^>]+>", "", body).strip()
            if clean:
                sections.append({"level": int(level), "text": clean})
    profile["sections"] = sections
    return profile


# ---------------------------------------------------------------------------
# CSS theme synthesis
# ---------------------------------------------------------------------------

def _build_html_css(profile: dict[str, Any]) -> str:
    palette = profile.get("palette", {})
    fonts = profile.get("fonts", {})

    def pick(*keys, default):
        for k in keys:
            if palette.get(k):
                return palette[k]
        return default

    text_color = pick("dark1", "dark2", "color1", default="#1f2328")
    bg_color = pick("light1", "light2", default="#ffffff")
    accent = pick("accent1", "color2", "hyperlink", default="#2563eb")
    accent2 = pick("accent2", "accent3", "color3", default=accent)
    heading_font = fonts.get("heading", "Segoe UI, sans-serif")
    body_font = fonts.get("body", heading_font)

    return (
        "body { font-family: %(body)s; color: %(text)s; background: %(bg)s; }\n"
        "a { color: %(accent)s; }\n"
        "h1, h2, h3 { font-family: %(head)s; color: %(text)s; }\n"
        "h1 { border-bottom: 3px solid %(accent)s; padding-bottom: 0.3rem; }\n"
        "h2 { border-bottom: 1px solid %(accent2)s; padding-bottom: 0.2rem; }\n"
        "h3 { color: %(accent)s; }\n"
        "th { background: %(accent)s; color: %(bg)s; }\n"
        "blockquote { border-left: 4px solid %(accent)s; }\n"
        % {
            "body": body_font,
            "head": heading_font,
            "text": text_color,
            "bg": bg_color,
            "accent": accent,
            "accent2": accent2,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_style(path: Path, *, html_theme: bool = False) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pptx":
        profile = _extract_pptx(path, _import_et())
    elif suffix == ".docx":
        profile = _extract_docx(path, _import_et())
    elif suffix in (".html", ".htm"):
        profile = _extract_html(path)
    else:
        raise ValueError(
            f"Unsupported example type {suffix!r}. Use .pptx, .docx, or .html."
        )

    profile["source"] = str(path)
    profile.setdefault("palette", {})
    profile.setdefault("fonts", {})
    if html_theme:
        profile["html_css"] = _build_html_css(profile)
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", help="Path to a .pptx / .docx / .html example")
    parser.add_argument(
        "--html-theme",
        action="store_true",
        help="Also emit a 'html_css' block synthesised from the palette/fonts.",
    )
    args = parser.parse_args(argv)

    path = Path(args.example).expanduser()
    if not path.is_file():
        print(json.dumps({"ok": False, "error": f"File not found: {path}"}))
        return 1
    try:
        profile = extract_style(path, html_theme=args.html_theme)
    except Exception as e:  # noqa: BLE001 - surface a clean JSON error
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1

    print(json.dumps({"ok": True, **profile}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
