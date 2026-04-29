"""FlowForge wrapper around the vendored PPT Master SVG pipeline."""

from __future__ import annotations

import re
import tempfile
import textwrap
from html import escape
from pathlib import Path
from typing import Any

from flowforge.tools._ppt_master.svg_to_pptx import create_pptx_with_native_svg

SLIDE_W = 1280
SLIDE_H = 720


PALETTES: dict[str, dict[str, str]] = {
    "default": {
        "bg": "F7F8FB", "fg": "172033", "muted": "5F6B7A",
        "accent": "2F6FED", "accent2": "11A683", "panel": "FFFFFF",
        "line": "D8DEE8",
    },
    "dark": {
        "bg": "1E1E2E", "fg": "FFFFFF", "muted": "C9CEDA",
        "accent": "8AB4FF", "accent2": "7AE1B8", "panel": "2B2D42",
        "line": "4A4E69",
    },
    "editorial": {
        "bg": "F6F0E8", "fg": "1C1A17", "muted": "6F6257",
        "accent": "C2472D", "accent2": "243B53", "panel": "FFF9F1",
        "line": "DACBBB",
    },
    "consulting": {
        "bg": "FFFFFF", "fg": "111827", "muted": "4B5563",
        "accent": "0B5CAD", "accent2": "E87500", "panel": "F3F6FA",
        "line": "CBD5E1",
    },
    "academic": {
        "bg": "FAFAF8", "fg": "1F2937", "muted": "56616F",
        "accent": "6A1B9A", "accent2": "0F766E", "panel": "FFFFFF",
        "line": "D6D3D1",
    },
    "tech": {
        "bg": "07111F", "fg": "F8FAFC", "muted": "A7B1C2",
        "accent": "38BDF8", "accent2": "A3E635", "panel": "102033",
        "line": "29445F",
    },
}


def create_native_pptx_from_slide_data(
    *,
    project_root: Path,
    output_path: Path,
    slide_data: list[dict[str, Any]],
    theme: str = "default",
    verbose: bool = False,
) -> dict[str, Any]:
    """Render slide objects through PPT Master's SVG -> DrawingML converter."""
    root = project_root.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="flowforge-ppt-master-") as temp:
        svg_dir = Path(temp) / "svg_output"
        svg_dir.mkdir(parents=True, exist_ok=True)

        svg_files: list[Path] = []
        notes: dict[str, str] = {}
        layouts: list[str] = []

        for idx, slide in enumerate(slide_data, start=1):
            layout = str(slide.get("layout", "content")).lower()
            layouts.append(layout)
            stem = f"{idx:02d}_{_slug(slide.get('title') or layout or 'slide')}"
            svg_path = svg_dir / f"{stem}.svg"

            raw_svg = _load_svg(project_root=root, slide=slide)
            svg = raw_svg if raw_svg is not None else _render_slide_svg(slide, theme, idx)
            svg_path.write_text(_ensure_svg_canvas(svg), encoding="utf-8")
            svg_files.append(svg_path)

            note = slide.get("speaker_note", slide.get("notes", ""))
            if note:
                notes[stem] = str(note)

        ok = create_pptx_with_native_svg(
            svg_files=svg_files,
            output_path=output_path,
            canvas_format=None,
            verbose=verbose,
            transition=None,
            use_native_shapes=True,
            notes=notes,
            enable_notes=True,
        )

    return {
        "ok": bool(ok),
        "engine": "ppt-master",
        "native_objects": True,
        "layouts": layouts,
    }


def _load_svg(*, project_root: Path, slide: dict[str, Any]) -> str | None:
    if slide.get("svg"):
        return str(slide["svg"])

    raw_path = slide.get("svg_path")
    if not raw_path:
        return None

    path = Path(str(raw_path)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"svg_path must stay inside project root: {resolved}")
    return resolved.read_text(encoding="utf-8")


def _ensure_svg_canvas(svg: str) -> str:
    text = svg.strip()
    if "<svg" not in text[:200].lower():
        raise ValueError("svg slide content must contain an <svg> root")
    if "xmlns=" not in text[:300]:
        text = text.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if "viewBox=" not in text[:500]:
        text = text.replace("<svg", f'<svg viewBox="0 0 {SLIDE_W} {SLIDE_H}"', 1)
    return text


def _render_slide_svg(slide: dict[str, Any], theme: str, idx: int) -> str:
    pal = _palette(slide, theme)
    layout = str(slide.get("layout", "content")).lower()
    body: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SLIDE_W} {SLIDE_H}">',
        _rect(0, 0, SLIDE_W, SLIDE_H, pal["bg"]),
    ]

    if layout == "cover":
        _cover(body, slide, pal)
    elif layout == "section":
        _section(body, slide, pal)
    elif layout == "metric":
        _metric(body, slide, pal)
    elif layout in {"timeline", "process"}:
        _timeline(body, slide, pal)
    elif layout == "cards":
        _cards(body, slide, pal)
    elif layout == "comparison":
        _comparison(body, slide, pal)
    elif layout == "table":
        _table(body, slide, pal)
    elif layout == "chart":
        _chart(body, slide, pal)
    elif layout == "quote":
        _quote(body, slide, pal)
    else:
        _content(body, slide, pal)

    _custom_shapes(body, slide, pal)
    _footer(body, slide, pal, idx)
    body.append("</svg>")
    return "\n".join(body)


def _palette(slide: dict[str, Any], theme: str) -> dict[str, str]:
    pal = dict(PALETTES.get(str(theme).lower(), PALETTES["default"]))
    slide_theme = str(slide.get("theme", "")).lower()
    if slide_theme in PALETTES:
        pal.update(PALETTES[slide_theme])
    for key in ("bg", "fg", "muted", "accent", "accent2", "panel", "line"):
        if key in slide:
            pal[key] = _hex(slide[key], pal[key])
    return pal


def _hex(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        int(text, 16)
    except ValueError:
        text = fallback
    if len(text) != 6:
        text = fallback
    return text.upper()


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text[:36] or "slide"


def _as_dict(value: Any, fallback_key: str = "title") -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {fallback_key: str(value)}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None, rx: float = 0) -> str:
    stroke_attr = f' stroke="#{stroke}" stroke-width="2"' if stroke else ""
    rx_attr = f' rx="{rx}" ry="{rx}"' if rx else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#{fill}"{stroke_attr}{rx_attr}/>'


def _circle(cx: float, cy: float, r: float, fill: str, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="#{stroke}" stroke-width="2"' if stroke else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#{fill}"{stroke_attr}/>'


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 3) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#{stroke}" stroke-width="{width}"/>'


def _text(
    x: float,
    y: float,
    text: Any,
    *,
    size: float,
    fill: str,
    weight: str = "400",
    anchor: str = "start",
    max_chars: int | None = None,
    max_lines: int = 5,
    line_height: float = 1.2,
) -> str:
    value = str(text or "")
    if max_chars:
        lines = textwrap.wrap(value, max_chars, break_long_words=False)[:max_lines]
    else:
        lines = value.splitlines()[:max_lines] or [""]
    items = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * line_height
        items.append(
            f'<text x="{x}" y="{y + dy}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="#{fill}" '
            f'text-anchor="{anchor}">{escape(line)}</text>'
        )
    return "\n".join(items)


def _bullet_list(body: list[str], bullets: list[Any], x: float, y: float, pal: dict[str, str], *, size: float = 28) -> None:
    for i, bullet in enumerate(bullets[:7]):
        yy = y + i * (size * 1.55)
        body.append(_circle(x, yy - size * 0.28, size * 0.13, pal["accent"]))
        body.append(_text(x + size * 0.8, yy, bullet, size=size, fill=pal["fg"], max_chars=54, max_lines=2))


def _title(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    kicker = slide.get("kicker") or slide.get("eyebrow")
    if kicker:
        body.append(_text(72, 45, str(kicker).upper(), size=17, fill=pal["accent"], weight="700"))
    body.append(_text(72, 94, slide.get("title", ""), size=46, fill=pal["fg"], weight="700", max_chars=34, max_lines=2))


def _footer(body: list[str], slide: dict[str, Any], pal: dict[str, str], idx: int) -> None:
    footer = slide.get("footer", "")
    if footer or slide.get("show_page_number"):
        text = str(footer)
        if slide.get("show_page_number"):
            text = f"{text}  {idx}".strip()
        body.append(_text(72, 690, text, size=14, fill=pal["muted"]))


def _cover(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    body.append(_rect(0, 0, 18, SLIDE_H, pal["accent"]))
    body.append(_rect(914, 0, 366, SLIDE_H, pal["panel"]))
    kicker = slide.get("kicker") or slide.get("eyebrow")
    if kicker:
        body.append(_text(82, 135, str(kicker).upper(), size=18, fill=pal["accent"], weight="700"))
    body.append(_text(82, 235, slide.get("title", ""), size=58, fill=pal["fg"], weight="700", max_chars=25, max_lines=3))
    subtitle = slide.get("subtitle") or slide.get("body")
    if subtitle:
        body.append(_text(86, 465, subtitle, size=27, fill=pal["muted"], max_chars=42, max_lines=2))
    bullets = list(slide.get("bullets") or [])[:3]
    _bullet_list(body, bullets, 92, 560, pal, size=22)


def _section(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    body.append(_rect(78, 135, 10, 420, pal["accent"]))
    body.append(_text(124, 326, slide.get("title", ""), size=58, fill=pal["fg"], weight="700", max_chars=30, max_lines=2))
    subtitle = slide.get("subtitle") or slide.get("body")
    if subtitle:
        body.append(_text(128, 455, subtitle, size=29, fill=pal["muted"], max_chars=48, max_lines=2))


def _content(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    if slide.get("bullets"):
        _bullet_list(body, list(slide["bullets"]), 92, 220, pal, size=28)
    elif slide.get("body"):
        body.append(_text(86, 230, slide["body"], size=30, fill=pal["fg"], max_chars=55, max_lines=8))


def _metric(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    metrics = list(slide.get("metrics") or [])
    if not metrics and slide.get("bullets"):
        metrics = [{"value": item, "label": ""} for item in slide["bullets"][:3]]
    for i, raw_metric in enumerate(metrics[:3]):
        metric = _as_dict(raw_metric, "value")
        x = 92 + i * 390
        body.append(_text(x, 300, metric.get("value", ""), size=70, fill=pal["accent" if i == 0 else "fg"], weight="700", max_chars=9))
        body.append(_text(x + 3, 360, metric.get("label", ""), size=27, fill=pal["fg"], weight="700", max_chars=20))
        note = metric.get("note") or metric.get("delta")
        if note:
            body.append(_text(x + 3, 415, note, size=20, fill=pal["muted"], max_chars=28, max_lines=2))
    if slide.get("body"):
        body.append(_text(92, 565, slide["body"], size=25, fill=pal["muted"], max_chars=70, max_lines=2))


def _cards(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    cards = list(slide.get("cards") or [])
    if not cards and slide.get("bullets"):
        cards = [{"title": item, "body": ""} for item in slide["bullets"]]
    n = max(1, min(len(cards), 4))
    card_w = 1100 / n
    for i, raw_card in enumerate(cards[:4]):
        card = _as_dict(raw_card)
        x = 82 + i * card_w
        body.append(_rect(x, 215, card_w - 24, 330, pal["panel"], pal["line"], rx=18))
        body.append(_text(x + 28, 280, card.get("title", ""), size=29, fill=pal["accent"], weight="700", max_chars=18, max_lines=2))
        if card.get("body"):
            body.append(_text(x + 28, 365, card["body"], size=20, fill=pal["fg"], max_chars=22, max_lines=4))
        if card.get("bullets"):
            _bullet_list(body, list(card["bullets"])[:3], x + 34, 455, pal, size=17)


def _timeline(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    items = list(slide.get("items") or slide.get("timeline") or slide.get("steps") or [])
    if not items and slide.get("bullets"):
        items = [{"title": item} for item in slide["bullets"]]
    n = max(1, min(len(items), 5))
    body.append(_line(120, 342, 1160, 342, pal["line"], width=4))
    for i, raw_item in enumerate(items[:5]):
        item = _as_dict(raw_item)
        x = 120 + i * (1040 / max(1, n - 1))
        body.append(_circle(x, 342, 16, pal["accent"]))
        body.append(_text(x, 290, item.get("label") or item.get("date") or i + 1, size=18, fill=pal["accent"], weight="700", anchor="middle"))
        body.append(_text(x, 412, item.get("title", ""), size=22, fill=pal["fg"], weight="700", anchor="middle", max_chars=13, max_lines=2))
        if item.get("body"):
            body.append(_text(x, 490, item["body"], size=16, fill=pal["muted"], anchor="middle", max_chars=18, max_lines=3))


def _comparison(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    for i, key in enumerate(("left", "right")):
        col = _as_dict(slide.get(key, {}), "heading")
        x = 82 + i * 590
        body.append(_rect(x, 210, 530, 375, pal["panel"], pal["line"], rx=18))
        body.append(_text(x + 32, 280, col.get("heading", f"Option {i + 1}"), size=31, fill=pal["accent" if i == 0 else "accent2"], weight="700", max_chars=20))
        _bullet_list(body, list(col.get("bullets", [])), x + 42, 360, pal, size=21)


def _table(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    table = slide.get("table", {}) or {}
    headers = list(table.get("headers", []))
    rows = list(table.get("rows", []))
    if not headers:
        return
    x, y, w = 74, 205, 1132
    col_w = w / max(1, len(headers))
    row_h = 46
    body.append(_rect(x, y, w, row_h, pal["accent"]))
    for c, header in enumerate(headers):
        body.append(_text(x + c * col_w + 18, y + 31, header, size=18, fill="FFFFFF", weight="700", max_chars=16))
    for r, row in enumerate(rows[:8]):
        yy = y + row_h * (r + 1)
        body.append(_rect(x, yy, w, row_h, pal["panel"] if r % 2 else pal["bg"], pal["line"]))
        for c in range(len(headers)):
            value = row[c] if isinstance(row, (list, tuple)) and c < len(row) else ""
            body.append(_text(x + c * col_w + 18, yy + 30, value, size=17, fill=pal["fg"], max_chars=18))


def _chart(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    _title(body, slide, pal)
    chart = slide.get("chart", {}) or {}
    categories = list(chart.get("categories") or [])
    series = chart.get("series") or [{"name": chart.get("name", "Series"), "values": chart.get("values", [])}]
    if isinstance(series, dict):
        series = [series]
    values = list(_as_dict(series[0]).get("values", [])) if series else []
    if not categories or not values:
        return
    numeric_values = [_number(v) for v in values]
    max_value = max(numeric_values + [1.0])
    x0, y0, w, h = 150, 550, 930, 300
    body.append(_line(x0, y0, x0 + w, y0, pal["line"], width=3))
    bar_w = w / max(1, len(values)) * 0.55
    gap = w / max(1, len(values))
    for i, value in enumerate(values):
        val = numeric_values[i]
        bh = (val / max_value) * h
        x = x0 + i * gap + gap * 0.22
        body.append(_rect(x, y0 - bh, bar_w, bh, pal["accent"]))
        label = categories[i] if i < len(categories) else str(i + 1)
        body.append(_text(x + bar_w / 2, 590, label, size=17, fill=pal["muted"], anchor="middle", max_chars=10))
        body.append(_text(x + bar_w / 2, y0 - bh - 14, value, size=17, fill=pal["fg"], anchor="middle"))


def _quote(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    body.append(_text(78, 175, "\"", size=90, fill=pal["accent"], weight="700"))
    body.append(_text(142, 280, slide.get("quote") or slide.get("body") or slide.get("title", ""), size=50, fill=pal["fg"], weight="700", max_chars=35, max_lines=4))
    attribution = slide.get("attribution") or slide.get("subtitle")
    if attribution:
        body.append(_text(150, 520, attribution, size=25, fill=pal["muted"], max_chars=45))


def _custom_shapes(body: list[str], slide: dict[str, Any], pal: dict[str, str]) -> None:
    for raw in list(slide.get("shapes") or slide.get("objects") or []):
        obj = _as_dict(raw, "text")
        fill = _hex(obj.get("fill", pal["panel"]), pal["panel"])
        stroke = _hex(obj.get("stroke", obj.get("line", pal["line"])), pal["line"])
        x = float(obj.get("x", 0)) * 96
        y = float(obj.get("y", 0)) * 96
        w = float(obj.get("w", 1)) * 96
        h = float(obj.get("h", 1)) * 96
        if str(obj.get("type", "rect")).lower() in {"circle", "ellipse", "oval"}:
            body.append(f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" rx="{w / 2}" ry="{h / 2}" fill="#{fill}" stroke="#{stroke}" stroke-width="2"/>')
        else:
            body.append(_rect(x, y, w, h, fill, stroke, rx=12 if str(obj.get("type", "")).lower() == "round_rect" else 0))
        if obj.get("text"):
            body.append(_text(x + w / 2, y + h / 2 + 8, obj["text"], size=float(obj.get("font_size", 18)), fill=_hex(obj.get("color", pal["fg"]), pal["fg"]), weight="700" if obj.get("bold") else "400", anchor="middle", max_chars=20, max_lines=2))
