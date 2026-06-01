---
name: doc-style-mirror
description: Generate new documents that mirror the theme, template, and screen layout of example company files. Given one or more example .pptx / .docx / .html documents, this skill extracts their colour palette, fonts, slide-layout inventory, and section structure, then produces a new PPTX, DOCX, or HTML document in the same visual identity. Use it whenever a user supplies a sample deck, report, or page and asks for a new document "in the same style / template / format".
license: Apache-2.0 (FlowForge)
allowed-tools: doc_render, html_create, pptx_create, docx_create, pdf_read_text, files_read_text, files_list_dir, shell_project_exec
---

# Document Style Mirror

Produce new documents that visually match a company's existing materials. The
user provides example files (a slide deck, a Word report, an HTML page); you
reproduce their **theme** (colours + fonts), **template** (slide masters /
styles), and **screen arrangement** (layout + section structure) in a brand-new
document with fresh content.

## When to use

Activate this skill when the user:

- gives an example `.pptx` / `.docx` / `.html` and asks for a new document
  "in the same style", "matching our template", "following this format/layout",
  or "using our brand";
- wants a deck/report/page that looks like an existing internal document but
  with different content.

## Inputs

- **Example document(s)** — one or more `.pptx`, `.docx`, or `.html` files that
  define the target look. Pick the example whose format matches the requested
  output (deck → `.pptx`, report → `.docx`, page → `.html`). If the user gives
  an example in a different format than requested, you can still mirror its
  palette/fonts via the extracted CSS / structure.
- **New content** — the subject matter for the new document, from the user.

## Workflow

### 1. Analyse the example → style profile

Run the bundled extractor on each example file. It reads the file with no
external dependencies (Office files are parsed straight from their OOXML zip)
and prints a JSON style profile.

```bash
python scripts/extract_style.py <example_path> --html-theme
```

The profile contains:

- `palette` — theme colours as `#RRGGBB` (`dark1`, `light1`, `accent1`…`accent6`,
  `hyperlink`, …). For HTML the most frequent hex colours are returned as
  `color1`, `color2`, …
- `fonts` — `heading` and `body` typefaces.
- `layouts` *(pptx)* — every slide layout with its `name`, `type`, and the
  placeholder types it exposes (e.g. `title`, `body`, `pic`, `tbl`). Use this to
  decide how to arrange each new slide.
- `heading_styles` *(docx)* — per-heading font/size/colour.
- `sections` — the heading outline of the example, i.e. its on-screen
  arrangement and information hierarchy. Mirror this structure in the new doc.
- `html_css` *(with `--html-theme`)* — a ready-to-use CSS block synthesised from
  the palette + fonts; pass it verbatim as `custom_css` when rendering HTML.

Read the profile before writing anything. Note the palette, fonts, the number
and order of sections/slides, and which layouts the example favours.

### 2. Draft the new content as Markdown

Write the new document body in Markdown, **echoing the example's structure**:

- Match its section depth and ordering (use the `sections` outline as a
  skeleton — same number of top-level sections / slides where it makes sense).
- For decks, one top-level heading (`#`) per slide; use sub-bullets, tables, and
  short lines that suit the layouts you saw in `layouts`.
- Keep the user's language (e.g. Korean) and tone.

### 3. Render in the example's visual identity

Choose the renderer by output format. **Prefer reusing the example file itself**
as a template so the company's master slides, fonts, and colours carry over
natively and remain fully editable.

- **PPTX** — `doc_render(path="deck.pptx", source=<markdown>,
  reference_doc=<example.pptx>)`. The new deck inherits the example's slide
  masters, theme colours, and fonts. Tune `slide_level` if a different heading
  depth should start a new slide.
- **DOCX** — `doc_render(path="report.docx", source=<markdown>,
  reference_doc=<example.docx>)`. Inherits the example's styles (heading fonts,
  sizes, colours) and page setup.
- **HTML** — `html_create(path="page.html", source=<markdown>,
  custom_css=<html_css from step 1>)`. Layers the extracted palette/fonts over a
  base theme. Pick a close built-in `theme` (default, dark, consulting,
  academic, tech) as the base, then override with `custom_css`.

`reference_doc` and example paths must be **project-relative** (inside the
project root). If the example lives elsewhere, copy it into the project first.

### 4. Verify and report

Confirm the tool returned `ok: True` and a non-zero `size`. Tell the user which
example was mirrored, the palette/fonts that were reused, and the output path.
If a profile field was empty (e.g. the example had no theme part), say so and
fall back to the closest built-in theme.

## Tips

- The example only needs to be a *representative* file — even one slide / one
  page is enough to capture the theme and template.
- For multi-format requests (deck **and** report), extract once per example and
  render each format with its matching `reference_doc`.
- Never invent brand colours: only use values present in the extracted
  `palette`. If none were found, ask the user or use a neutral built-in theme.
- This skill is provider-neutral (`AgentSkill`); it works with any LLM backend
  because all styling is done locally via the extractor + Pandoc.
