# Your First Agent

This guide builds a small document parser that detects input format, branches
to the right parser, and returns a normalized result.

## 1. Define Models

```python
from pydantic import BaseModel

class Document(BaseModel):
    content: str
    format: str

class ParsedDocument(BaseModel):
    records: list[dict]
    source_format: str
```

## 2. Define Branch Handlers

```python
import csv
import io

async def parse_csv(ctx):
    doc = ctx.input
    reader = csv.DictReader(io.StringIO(doc.content))
    return ParsedDocument(records=list(reader), source_format="csv")

async def parse_text(ctx):
    doc = ctx.input
    rows = [{"line": line} for line in doc.content.splitlines() if line.strip()]
    return ParsedDocument(records=rows, source_format="text")
```

## 3. Define The Agent

```python
from flowforge import BranchCondition, FlowForge, global_config, flow, task, step

@global_config(prompt="You parse documents into normalized records.")
class DocumentAgent:

    @flow(name="parse_document", prompt="Parse a document")
    class ParseDocumentFlow:

        @task(
            name="parse",
            prompt="Choose a parser and normalize output",
            input_schema=Document,
            output_schema=ParsedDocument,
        )
        class ParseTask:

            @step(
                order=1,
                prompt="Route to the correct parser",
                input_schema=Document,
                output_schema=ParsedDocument,
                condition=BranchCondition(field="format", enum=["csv", "text"]),
                branches={"csv": parse_csv, "text": parse_text},
                fallback=parse_text,
            )
            async def route(ctx):
                ...
```

## 4. Compile And Run

```python
import asyncio

async def main():
    engine = FlowForge.compile(DocumentAgent)
    result = await engine.run(Document(content="name\nAda\nGrace", format="csv"))
    print(result)

asyncio.run(main())
```

## 5. Inspect The Run

```python
result, trace = await engine.run_traced(Document(content="hello", format="text"))
engine.print_run_summary(trace)
print(engine.run_mermaid(trace))
```

You now have:

- A global agent configuration
- A flow
- A typed task
- A branching step
- Runtime trace visualization

## Next Steps

- Add `ctx.call_llm(...)` inside a step to summarize parsed records.
- Register a `FunctionTool` or `MCPServer` in `@global_config(tools=[...])`.
- Use `route="parse_document.parse"` to run only this capability.
