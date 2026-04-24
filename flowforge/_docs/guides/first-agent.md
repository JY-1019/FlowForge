# Your First Agent

Build a complete document-processing agent from scratch.

---

## What We're Building

A `DocumentAgent` that:

1. Validates the input document format
2. Routes to the correct parser based on format (`json` / `csv` / `text`)
3. Extracts key information
4. Formats a final summary

---

## Step 1 — Define Schemas

```python
# schemas.py
from pydantic import BaseModel

class RawDocument(BaseModel):
    content: str
    filename: str

class ValidatedDocument(BaseModel):
    content: str
    filename: str
    format: str           # "json" | "csv" | "text"

class ParsedDocument(BaseModel):
    records: list[dict]
    source_format: str

class DocumentSummary(BaseModel):
    title: str
    record_count: int
    preview: str
```

---

## Step 2 — Define Branch Handlers

Branch handlers are regular async functions defined **outside** the agent class.

```python
# handlers.py
async def parse_json(ctx):
    import json
    doc = ctx.input   # ValidatedDocument
    records = json.loads(doc.content)
    if isinstance(records, dict):
        records = [records]
    return ParsedDocument(records=records, source_format="json")

async def parse_csv(ctx):
    import csv, io
    doc = ctx.input
    reader = csv.DictReader(io.StringIO(doc.content))
    records = list(reader)
    return ParsedDocument(records=records, source_format="csv")

async def parse_text(ctx):
    doc = ctx.input
    records = [{"line": line} for line in doc.content.splitlines() if line.strip()]
    return ParsedDocument(records=records, source_format="text")
```

---

## Step 3 — Define the Agent

```python
# document_agent.py
from flowforge import global_config, flow, task, step, branch, FlowForge
from flowforge.types import BranchCondition

@global_config(prompt="You are a document processing specialist.")
class DocumentAgent:

    @flow(
        name="process",
        prompt="Validate, parse, and summarize the document",
        input_schema=RawDocument,
        output_schema=DocumentSummary,
    )
    class ProcessFlow:

        @task(
            name="validate_and_parse",
            prompt="Validate format and parse content",
        )
        class ValidateAndParseTask:

            @step(
                order=1,
                prompt="Detect the document format from content and filename",
                input_schema=RawDocument,
                output_schema=ValidatedDocument,
            )
            async def detect_format(ctx):
                doc: RawDocument = ctx.input
                if doc.filename.endswith(".json") or doc.content.strip().startswith("{"):
                    fmt = "json"
                elif doc.filename.endswith(".csv") or "," in doc.content.splitlines()[0]:
                    fmt = "csv"
                else:
                    fmt = "text"
                return ValidatedDocument(
                    content=doc.content,
                    filename=doc.filename,
                    format=fmt,
                )

            @branch(
                order=2,
                name="format_router",
                prompt="Route to the correct parser based on detected format",
                condition=BranchCondition(field="format", enum=["json", "csv", "text"]),
                branches={
                    "json": parse_json,
                    "csv":  parse_csv,
                    "text": parse_text,
                },
                fallback=parse_text,
            )
            async def route_parser(ctx): ...

        @task(
            name="summarize",
            prompt="Generate a human-readable summary of the parsed document",
        )
        class SummarizeTask:

            @step(
                order=1,
                prompt="Count records and extract a preview",
                input_schema=ParsedDocument,
                output_schema=DocumentSummary,
            )
            async def summarize(ctx):
                parsed: ParsedDocument = ctx.input
                preview_records = parsed.records[:3]
                preview = str(preview_records)[:200]
                return DocumentSummary(
                    title=f"Document ({parsed.source_format})",
                    record_count=len(parsed.records),
                    preview=preview,
                )
```

---

## Step 4 — Run the Agent

```python
# run.py
import asyncio
from document_agent import DocumentAgent, RawDocument
from flowforge import FlowForge

async def main():
    engine = FlowForge.compile(DocumentAgent)

    # Test with a CSV document
    csv_doc = RawDocument(
        content="name,age\nAlice,30\nBob,25",
        filename="people.csv",
    )
    result = await engine.run(csv_doc)
    print(f"Title: {result.title}")
    print(f"Records: {result.record_count}")
    print(f"Preview: {result.preview}")

    # Test with a JSON document
    json_doc = RawDocument(
        content='{"product": "FlowForge", "version": "1.0"}',
        filename="info.json",
    )
    result = await engine.run(json_doc)
    print(result)

asyncio.run(main())
```

---

## Step 5 — Validate and Visualize

```bash
# Validate the DAG
flowforge validate document_agent.py

# Print Mermaid diagram
flowforge viz document_agent.py --mermaid

# Run with trace table
flowforge run document_agent.py \
  -q '{"content": "a,b\n1,2", "filename": "data.csv"}' \
  --trace

# Render run subtree to SVG
flowforge run document_agent.py \
  -q '{"content": "a,b\n1,2", "filename": "data.csv"}' \
  --viz --viz-output run.svg
```

---

## Complete File

??? example "Full document_agent.py"

    ```python
    import asyncio
    import json
    import csv
    import io
    from pydantic import BaseModel
    from flowforge import global_config, flow, task, step, branch, FlowForge
    from flowforge.types import BranchCondition


    # ── Schemas ────────────────────────────────────────────────────────────────

    class RawDocument(BaseModel):
        content: str
        filename: str

    class ValidatedDocument(BaseModel):
        content: str
        filename: str
        format: str

    class ParsedDocument(BaseModel):
        records: list[dict]
        source_format: str

    class DocumentSummary(BaseModel):
        title: str
        record_count: int
        preview: str


    # ── Branch Handlers ─────────────────────────────────────────────────────────

    async def parse_json(ctx):
        doc = ctx.input
        records = json.loads(doc.content)
        if isinstance(records, dict):
            records = [records]
        return ParsedDocument(records=records, source_format="json")

    async def parse_csv(ctx):
        doc = ctx.input
        reader = csv.DictReader(io.StringIO(doc.content))
        return ParsedDocument(records=list(reader), source_format="csv")

    async def parse_text(ctx):
        doc = ctx.input
        records = [{"line": l} for l in doc.content.splitlines() if l.strip()]
        return ParsedDocument(records=records, source_format="text")


    # ── Agent ───────────────────────────────────────────────────────────────────

    @global_config(prompt="You are a document processing specialist.")
    class DocumentAgent:

        @flow(name="process", prompt="Validate, parse, and summarize",
              input_schema=RawDocument, output_schema=DocumentSummary)
        class ProcessFlow:

            @task(name="validate_and_parse", prompt="Validate format and parse")
            class ValidateAndParseTask:

                @step(order=1, prompt="Detect document format",
                      input_schema=RawDocument, output_schema=ValidatedDocument)
                async def detect_format(ctx):
                    doc = ctx.input
                    if doc.filename.endswith(".json") or doc.content.strip().startswith("{"):
                        fmt = "json"
                    elif doc.filename.endswith(".csv"):
                        fmt = "csv"
                    else:
                        fmt = "text"
                    return ValidatedDocument(content=doc.content,
                                             filename=doc.filename, format=fmt)

                @branch(order=2, name="format_router",
                        prompt="Route to correct parser",
                        condition=BranchCondition(field="format",
                                                  enum=["json", "csv", "text"]),
                        branches={"json": parse_json, "csv": parse_csv,
                                  "text": parse_text},
                        fallback=parse_text)
                async def route_parser(ctx): ...

            @task(name="summarize", prompt="Generate document summary")
            class SummarizeTask:

                @step(order=1, prompt="Count records and extract preview",
                      input_schema=ParsedDocument, output_schema=DocumentSummary)
                async def summarize(ctx):
                    parsed = ctx.input
                    preview = str(parsed.records[:3])[:200]
                    return DocumentSummary(
                        title=f"Document ({parsed.source_format})",
                        record_count=len(parsed.records),
                        preview=preview,
                    )


    # ── Entry Point ──────────────────────────────────────────────────────────────

    async def main():
        engine = FlowForge.compile(DocumentAgent)
        doc = RawDocument(content="name,age\nAlice,30\nBob,25", filename="data.csv")
        result = await engine.run(doc)
        print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```
