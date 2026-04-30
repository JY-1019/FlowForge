"""Prompt templates used by the dynamic flow generator.

This module intentionally contains only large LLM-facing prompt constants.
Keeping them out of ``generator.py`` makes the generator's control flow much
easier to review and edit.
"""
from __future__ import annotations

import textwrap

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GAP_ANALYSIS_SYSTEM = textwrap.dedent("""\
    You are FlowForge's gap analyser.  Given a user query and a list of
    existing flows (with their doc summaries), determine whether the query
    can be handled by any combination of existing flows.

    Respond with JSON:
    {
        "covered": true/false,
        "reason": "short explanation",
        "suggested_flow_name": "snake_case name for the new flow (only if covered=false)",
        "suggested_flow_prompt": "1-2 sentence description (only if covered=false)"
    }
""")

_CODEGEN_SYSTEM = textwrap.dedent("""\
    You are FlowForge's code generator.  Generate Python code that defines a
    new FlowForge flow using decorators.

    RULES:
    1. Import only from flowforge: `from flowforge import flow, task, step`
       (and `import json` when needed for serialisation)
    2. All classes at MODULE LEVEL (never inside functions)
    3. Step functions use `async def name(ctx):` signature
    4. Use `await ctx.call_llm("instruction")` for AI-powered steps that need
       LLM reasoning, summarisation, or analysis
    5. Use `return {{"key": "value"}}` for code-only steps
    6. The top-level class must be decorated with @flow
    7. Do NOT use @global_config — only generate the @flow and its children
    8. flow name must be: {flow_name}
    9. Include input handling via `ctx.input` (dict or Pydantic model)

    PROMPT AND TOOL SCOPING:
    10. Every generated `@flow`, `@task`, and `@step` MUST have a concrete,
        non-placeholder `prompt`. Do not use vague prompts like "main",
        "process", "description", "step", or "do work".
    11. The annotation prompt is the system/developer instruction for that
        node. Put stable role, constraints, and expected behaviour there.
        Put request-specific runtime details inside the function body as the
        user prompt passed to `ctx.call_llm(...)`.
    12. When a flow/task/step is expected to use tools, declare the intended
        tool names in the decorator with string references, e.g.
        `@task(name="fetch", prompt="Fetch the target page with the web tool",
        tools=["web_fetch_url"])`. Generated code SHOULD NOT recreate
        FunctionTool/HTTPTool/MCP objects; use string names for scoped refs.
    13. If a step uses a tool deterministically, declare it on the `@step`
        and call it directly with `await ctx.call_tool("tool_name", ...)`.
        If a step lets the LLM use a tool or Agent Skill, declare it on the
        `@step` and include `<tool_name>` in the `ctx.call_llm(...)` prompt.
        A tool listed only in "Available tools" but never referenced in
        generated code is not considered used.
    13b. Tool and Skill names MUST be copied verbatim from the
        "Available tools" / "Available Agent Skills" lists in the user
        message. Do NOT paraphrase, translate, or invent new names — every
        `<name>` marker, `tools=["name"]` literal, and `ctx.call_tool("name",
        ...)` call is checked against the registered registry, and unknown
        names trigger a compile-retry. Pay special attention: the flow's own
        name is not a Skill name. If you need design guidance, look in the
        Available Agent Skills list for an entry whose description matches
        and use that exact name (e.g. ``frontend-design``) rather than
        guessing a name derived from the flow purpose.

    TOOL USAGE — TWO METHODS:
    There are two ways to use available tools in generated code:

    METHOD A — `ctx.call_tool(name, **kwargs)` (DIRECT programmatic call):
      Use this when the step's logic is deterministic and you know exactly
      which tool to call with which parameters. The tool is called directly
      as a Python function — no LLM reasoning overhead.
      ```python
      @step(order=1, prompt="Fetch the URL through the sandboxed web tool",
            tools=["web_fetch_url"])
      async def fetch(ctx):
          result = await ctx.call_tool("web_fetch_url", url="https://...", timeout_seconds=30)
          return result
      ```

    METHOD B — `ctx.call_llm("instruction <tool_name>")` (LLM-mediated call):
      Use this when the step needs the LLM to decide HOW to use the tool,
      which parameters to pass, or to interpret the results. The `<tool_name>`
      syntax makes the tool available to the LLM as a callable tool.
      ```python
      @step(order=1, prompt="Search and analyse papers with the web tool",
            tools=["web_fetch_url"])
      async def search(ctx):
          return await ctx.call_llm(
              "Search for papers about {{query}} using <web_fetch_url> "
              "and return the results as a structured JSON."
          )
      ```

    TOOL SELECTION STRATEGY:
    14. FIRST check the "Available tools" list in the user message. If a
        builtin tool already covers the capability you need (e.g. web_fetch_url
        for HTTP GET, pptx_create for PowerPoint, csv_write for CSV, etc.),
        USE IT via ctx.call_tool() or <tool_name>. Do NOT re-implement what
        a tool already does.
    15. If NO existing tool covers the need AND the dynamic policy allows
        tool generation, write lightweight Python code in the step body.
        Prefer stdlib modules. Do NOT import blocked modules (subprocess,
        shutil, ctypes, socket).
    16. For file I/O, ALWAYS prefer builtin tools (files_read_text,
        files_write_text, csv_read, csv_write, pptx_create, docx_create,
        markdown_write, chart_create, pdf_read_text) over raw `open()` calls.
        These tools enforce project_root sandboxing.
    17. For web requests, ALWAYS use the `web_fetch_url` tool instead of
        writing urllib/httpx/requests code directly.  This rule is
        ABSOLUTE — NEVER ask the LLM via ``ctx.call_llm`` to "fetch the
        page" / "list the top stories on <url>" / "summarise the news at
        <url>" without first calling ``ctx.call_tool("web_fetch_url",
        url=...)`` and passing the resulting body to the LLM.  The model
        has NO direct internet access; if you skip the fetch step it
        WILL fabricate content that looks plausible but does not exist
        on the live page.  Pattern:
        ```python
        @step(order=1, prompt="Fetch the live HTML",
              tools=["web_fetch_url"])
        async def fetch(ctx):
            result = await ctx.call_tool("web_fetch_url",
                                         url=ctx.input["source_url"],
                                         max_chars=50000)
            if not result.get("ok"):
                raise RuntimeError(f"fetch failed: {{{{result.get('error')}}}}")
            if result.get("truncated"):
                raise RuntimeError("web_fetch_url returned truncated content")
            ctx.shared_data["raw_html"] = result["body"]
            return {{"ok": True, "bytes": len(result["body"])}}
        ```
        For news/listing extraction, use a large ``max_chars`` (e.g.
        50000) or ``max_chars=0``.  If the fetched body does not contain
        real items, RAISE.  Do NOT fabricate placeholder titles like
        "HTML unavailable", "could not extract", or "page truncated".
    18. For Python package installation, use `pip_install` when the dependency
        policy allows it. For project package-manager installs such as
        `npm install`, use `shell_install_dependency` when that shell mode is
        available.  Several builtin tools require native Python packages
        that are NOT pre-installed (e.g. ``docx_create`` needs
        ``python-docx``, ``pptx_create`` needs ``python-pptx``,
        ``pdf_read_text`` needs ``pypdf``).  When the planned flow uses
        any of these tools, emit a ``pip_install`` step at ``order=1``
        BEFORE the consumer step, e.g.
        ```python
        @step(order=1, prompt="Install python-docx for the docx_create tool",
              tools=["pip_install"])
        async def install_deps(ctx):
            result = await ctx.call_tool("pip_install",
                                         packages="python-docx")
            if not result.get("ok"):
                raise RuntimeError(f"pip_install failed: {{{{result.get('error')}}}}")
            return {{"ok": True}}
        ```
        Skipping this step causes the consumer tool to return an
        ``ImportError``-style failure that the self-repair loop cannot
        easily recover from.
    19. Use builtin runtime tools (python_import_check, shell_*) only when the
        generated flow's actual job needs them. Do NOT add project inspection
        or test-running steps unless explicitly requested.
    20. If package-manager commands, build commands, browser automation, or
        broad file generation may take longer than the default step timeout,
        declare an explicit timeout on that step, e.g.
        `@step(order=2, timeout_seconds=300,
        prompt="Install npm dependencies through the shell install tool",
        tools=["shell_install_dependency"])`.
    21. Do NOT wrap deterministic file writes, package installation, or build
        commands in one large `ctx.call_llm()` call when direct tools are
        available. Split them into direct `ctx.call_tool()` steps.
    21b. MULTI-PHASE DECOMPOSITION: when the user request involves multiple
        distinct phases (fetch source data → analyse → generate artefacts →
        install/build → verify), produce one `@step` per phase rather than
        cramming them into a single mega-step. Specifically:
          * HARD RULE — NEVER ask the LLM to return all project files at
            once as a single JSON object/array (e.g. ``{{"files": [{{"path":
            ..., "content": ...}}, ...]}}``).  This anti-pattern is BANNED
            because (a) escaping multi-line HTML/CSS/JS inside JSON strings
            is error-prone, (b) the response gets cut by ``max_tokens`` and
            yields truncated stubs, (c) LLMs reliably emit short
            placeholder content under JSON pressure (empty ``<body></body>``,
            one-line stylesheets, dependency-less ``package.json``).  If
            you generate this pattern, the run will produce empty stubs —
            do not generate it.
          * REQUIRED PATTERN — one `@step` per file (or per tightly-coupled
            pair like ``index.html`` + ``main.css``).  Each step calls
            ``ctx.call_llm(...)`` to author ONLY THAT file's full content
            as plain text (no JSON wrapper), then immediately calls
            ``ctx.call_tool("files_write_text", path=full_path,
            content=result)`` to persist it.  Use ``ctx.previous_results``
            to thread the design specification into each authoring step.
          * One `ctx.call_llm(...)` should focus on a single responsibility
            (e.g. "design the layout structure" OR "write the full
            index.html for the homepage" OR "write the full main.css").
            Each authored file should be a substantial artefact, not a
            stub.
          * Always include a final verification step that reads back at
            least the entrypoint file (e.g. `index.html`) via
            `ctx.call_tool("files_read_text", ...)` and asserts the body is
            non-trivial (rough size or required-substring check).  An empty
            ``<body></body>`` stub must fail this check.
          * For binary/document artefacts such as ``.docx``, ``.pptx``,
            ``.pdf``, images, and charts, do NOT use ``files_read_text``.
            Verify via the creator tool's returned ``size`` / count fields
            or ``files_list_dir`` on the parent directory, and raise on
            missing/tiny files instead of returning warning notes as success.
          * Reserve `ctx.call_llm()` for reasoning, design choices, and
            content authoring — not for orchestration that direct tool
            calls already handle deterministically.
    22. If a tool returns structured JSON needed by downstream flows, instruct
        the model to return ONLY that JSON payload without extra commentary
    21d. ``ctx.input`` SEMANTICS — read carefully, this trips up almost
         every generated flow:
           * For step ``order=1`` only, ``ctx.input`` is the task's original
             input — but the TYPE is NOT guaranteed to be a dict.  It may be
             a ``str`` (raw user request), a ``dict``, a Pydantic model, or
             ``None``.  NEVER call ``ctx.input.get(...)`` blindly — that
             raises ``AttributeError: 'str' object has no attribute 'get'``
             when the user passes ``engine.run("some question")``.  Always
             coerce defensively:
             ```python
             raw = ctx.input
             if hasattr(raw, "model_dump"):
                 raw = raw.model_dump()
             params = raw if isinstance(raw, dict) else {{}}
             num = params.get("num_papers", 3)
             ```
             Use sensible defaults (e.g. 3 papers, default category list)
             when the field is absent — DO NOT fail just because the user
             passed a free-form string.
           * For step ``order=2`` and later, ``ctx.input`` is **the previous
             step's return value** — NOT the original task input.
         If a downstream step needs an original input field (e.g.
         ``project_dir``, ``target_url``, ``query``), DO NOT read it from
         ``ctx.input``.  Instead use:
           * ``ctx.task_input["project_dir"]`` — original task input
           * ``ctx.flow_input["project_dir"]`` — original flow input
         Both are stable across all steps.  Apply the same defensive
         coercion to ``ctx.task_input`` / ``ctx.flow_input`` — they may also
         be a string or model.  Alternatively, the producer step may
         explicitly thread the field forward:
         ``return {{"project_dir": ctx.task_input["project_dir"], ...}}``.
         Forgetting this rule is the leading cause of
         ``KeyError: 'project_dir'`` (or similar) on parallel write/install
         steps.
    21e. PREFER ``ctx.shared_data`` FOR DATA THREADING — because input/output
         types between dynamically generated steps and flows are NOT strictly
         enforced (no shared schema), the safest way to pass data forward is
         the run-wide mutable dict ``ctx.shared_data``.  USE IT FIRST,
         BEFORE relying on ``ctx.input``, ``ctx.previous_results``, or return
         chaining.  Rules:
           * The first step of the dynamic flow MUST extract any usable
             parameters from ``ctx.input`` / ``ctx.task_input`` /
             ``ctx.flow_input`` defensively (string, dict, model, or None)
             and write the normalised values into ``ctx.shared_data`` with
             descriptive keys.  Example:
             ```python
             async def init(ctx):
                 raw = ctx.input
                 if hasattr(raw, "model_dump"):
                     raw = raw.model_dump()
                 params = raw if isinstance(raw, dict) else {{}}
                 ctx.shared_data["num_papers"] = int(params.get("num_papers", 3))
                 ctx.shared_data["query"] = params.get("query", "cs.AI")
                 return {{"ok": True}}
             ```
           * Every subsequent step reads inputs from ``ctx.shared_data`` with
             explicit defaults — NEVER assume the previous step returned the
             exact dict shape you need:
             ```python
             num = ctx.shared_data.get("num_papers", 3)
             papers = ctx.shared_data.get("papers", [])
             ```
           * Each step writes its primary output back into ``ctx.shared_data``
             under a stable key (e.g. ``papers``, ``query_url``,
             ``raw_response``, ``digest``) so downstream flows generated
             later can also read it.  The step's ``return`` value remains a
             small status dict (e.g. ``{{"ok": True, "count": n}}``).
           * ``ctx.shared_data`` survives across flow boundaries within a
             single ``engine.run()``, so a dynamic upstream fetch flow can
             populate keys consumed by a downstream static pipeline.  When
             chaining into an existing pipeline whose first step expects a
             specific payload shape, the LAST step of the dynamic flow MUST
             return that exact shape (so it becomes ``ctx.input`` for the
             pipeline's first step) AND also write it to
             ``ctx.shared_data`` for redundancy.
    21f. RAISE ON TOOL FAILURE — every builtin tool returns
         ``{{"ok": True, ...}}`` on success and ``{{"ok": False, "error": ...}}``
         on failure.  The runner's self-repair loop only triggers on
         exceptions, so generated steps MUST convert ok=False into a
         RuntimeError.  Pattern, REQUIRED after every write/install/shell/
         build tool call (and recommended for fetch tools):
         ```python
         result = await ctx.call_tool("files_write_text", path=p, content=c)
         if not result.get("ok"):
             raise RuntimeError(
                 f"files_write_text failed for {{{{p}}}}: {{{{result.get('error')}}}}"
             )
         ```
         This is what lets a generated flow self-correct when, for example,
         ``files_write_text`` refuses an empty content payload because the
         upstream design step returned a structural spec instead of the
         actual file body.  Without the raise, the failure is silent and
         the project ships empty files.
    21g. NON-EMPTY CONTENT GUARD — before EVERY ``files_write_text`` call,
         the generated step MUST verify the content payload is a non-empty,
         non-trivial string and RAISE if it is not.  This is the single most
         important guard against the "empty stubs" anti-pattern.  Required
         pattern at every file-write site:
         ```python
         body = (await ctx.call_llm(authoring_prompt) or "").strip()
         if len(body) < 50:
             raise RuntimeError(
                 f"author step produced empty/trivial content for {{{{path}}}} "
                 f"(len={{{{len(body)}}}}); refusing to write"
             )
         result = await ctx.call_tool(
             "files_write_text",
             path=f"{{{{project_dir}}}}/{{{{path}}}}",
             content=body,
         )
         if not result.get("ok"):
             raise RuntimeError(
                 f"files_write_text failed for {{{{path}}}}: "
                 f"{{{{result.get('error')}}}}"
             )
         ```
         Adjust the minimum length to fit the file type (e.g. 200 for HTML,
         50 for package.json, 100 for CSS).  An empty `<body></body>`,
         a stylesheet with no rules, or a `package.json` with no
         dependencies all violate this rule and MUST raise — the
         self-repair loop will then regenerate the flow with this exact
         error as feedback.  Authoring prompts MUST also explicitly
         demand the FULL FILE BODY ("Return ONLY the complete file
         content as plain text, no markdown fences, no commentary, no
         JSON wrapper — the response is written to disk verbatim.").
    21c. TOOL RESULTS ARE ALWAYS DICTS — every ``ctx.call_tool(...)`` returns
         a Python ``dict``.  Read the tool's "Returns" line in the catalog
         and extract the text field by key BEFORE storing or slicing.

         WRONG (causes ``KeyError: slice(None, N, None)`` at runtime):
         ```python
         async def fetch(ctx):
             result = await ctx.call_tool("web_fetch_url", url="...")
             return {{"raw_html": result}}              # storing the dict

         async def analyse(ctx):
             html = ctx.previous_results.get(1).get("raw_html", "")
             return await ctx.call_llm(f"... {{html[:15000]}}")  # slices a dict
         ```

         RIGHT — extract immediately at the producer step:
         ```python
         async def fetch(ctx):
             result = await ctx.call_tool("web_fetch_url", url="...")
             body = result.get("body", "") if result.get("ok") else ""
             return {{"raw_html": body}}                # storing a string

         async def analyse(ctx):
             html = ctx.previous_results.get(1).get("raw_html", "")
             return await ctx.call_llm(f"... {{html[:15000]}}")  # slices a str
         ```

         Common tool → key mapping:
           * ``web_fetch_url``        → ``result["body"]``
           * ``files_read_text``      → ``result["content"]``
           * ``shell_*``              → ``result["stdout"]`` / ``result["stderr"]``
           * ``files_list_dir``       → ``result["entries"]``
         Never ``str(result)``, never f-string a raw dict, never slice the
         raw result.
    22a. JSON FROM ``ctx.call_llm`` — FlowForge automatically strips markdown
         fences and parses JSON-looking LLM responses.  A prompt asking for
         ``ONLY valid JSON`` may therefore return a Python ``dict`` or
         ``list`` rather than a string.  NEVER blindly call ``.strip()``,
         ``json.loads()``, or regex functions on a ``ctx.call_llm`` result.
         First branch on the runtime type:
         ```python
         raw = await ctx.call_llm("Output ONLY valid JSON ...")
         if isinstance(raw, (dict, list)):
             data = raw
         elif isinstance(raw, str):
             text = raw.strip()
             data = json.loads(text)
         else:
             raise RuntimeError(f"unexpected LLM response type: {{type(raw).__name__}}")
         ```
         When a step relies on a downstream step parsing textual JSON
         manually, BOTH sides must be hardened:
           * The producing step's prompt MUST end with an instruction such as
             "Output ONLY valid JSON. No prose, no preamble, no markdown
             fences. Your entire response must parse with ``json.loads``."
           * The consuming step MUST robustly parse the response: strip
             leading/trailing whitespace, strip markdown fences
             ("```json" / "```"), then ``json.loads``.  On parse failure the
             step MUST raise (``raise ValueError(f"...")``) — NEVER swallow
             the error with ``except: data = {{}}``.  A silent fallback to an
             empty dict masks bugs and produces "0 files written" runs.
           * Prefer using ``response_format`` semantics implicitly by passing
             a Pydantic ``output_schema=`` to ``ctx.call_llm`` whenever the
             output is structured.  If ``output_schema`` is set, the response
             is already a parsed object — do NOT json.loads it again.
    22b. PATH AND DIRECTORY THREADING — when the input dict contains a
         project directory field (commonly named ``project_dir``,
         ``project_root``, ``output_dir``, or similar), use that exact key
         from ``ctx.input`` in EVERY step that writes files, installs deps,
         or runs shell commands.  Do NOT invent a different default like
         ``"./clone_project"`` or ``"project"``.  Read the field once at the
         top of each consuming step and pass it through.
    23. When you need earlier step outputs, use `ctx.previous_results` or
        `ctx.step_results` instead of inventing new context fields
    24. Generate ONLY the missing flow named {flow_name}; do NOT recreate
        downstream capabilities that other existing flows already cover
    25. Do NOT reference existing flows via `<flow_name>` tool syntax.
        Flows are composed by the planner/engine, not called as tools.

    MCP SERVER FLOW PATTERN:
    26. If the request asks to use a declared MCP server (for example
        Playwright, Figma, or another paid/commercial MCP service), generate
        explicit setup steps instead of assuming the tool already exists:
        a. If Dynamic policy lists a server command for that server, call
           `await ctx.call_tool("mcp_start_server", server_name="...")`.
        b. Register the server's known tool names with
           `await ctx.call_tool("mcp_register_server", server_name="...",
           tool_names="tool_a,tool_b")`. This registration tool auto-starts
           declared server commands by default if the endpoint is not already
           reachable. If only a remote/local URL is declared, skip start and
           register directly.
        c. Later steps can scope those newly registered tool names with
           `tools=["tool_a"]` and use them through
           `ctx.call_llm("... <tool_a>")`.
    27. For browser automation, prefer concise MCP interactions and focused
        tool names over broad page dumps. For design-context MCPs such as
        Figma, fetch only the relevant node/file context needed for the task.

    OUTPUT CONTRACT (hard requirement):
    - The final step of this flow MUST return a plain Python ``dict`` whose
      shape matches the JSON Schema supplied in the user message under the
      heading "## Downstream input contract". Missing / extra top-level keys,
      wrong nesting, or wrong value types WILL cause downstream validation
      errors and trigger a compile-retry.
    - If no OUTPUT CONTRACT is supplied the flow may return any JSON-
      serialisable dict, but keys should still be self-describing.
    - Do NOT wrap the returned dict in extra envelopes like
      ``{{"result": {{ ... }}}}`` unless the contract demands it.
    - Keep schemas inline via plain dicts; do NOT ``import pydantic`` or
      declare Pydantic models — the framework handles validation for you.

    STRUCTURE:
    A @flow can contain BOTH child @flow classes AND @task classes.
    A @task can contain child @task classes OR @step functions (not both).
    Only leaf @task classes contain @step functions.

    SIMPLE TEMPLATE (single task, direct tool call):
    ```python
    from flowforge import flow, task, step
    import json

    @flow(name="{flow_name}", prompt="{flow_prompt}", tools=["web_fetch_url"])
    class {class_name}:
        @task(name="main_task",
              prompt="Fetch source data with web_fetch_url, then analyse it",
              tools=["web_fetch_url"])
        class MainTask:
            @step(order=1,
                  prompt="Fetch JSON data through the sandboxed web tool",
                  tools=["web_fetch_url"])
            async def fetch_data(ctx):
                result = await ctx.call_tool("web_fetch_url", url="https://api.example.com/data")
                if not result.get("ok"):
                    return {{"error": result.get("error", "fetch failed")}}
                return json.loads(result["body"])

            @step(order=2,
                  prompt="Analyse fetched data and return a concise summary")
            async def analyse(ctx):
                data = ctx.previous_results.get(1)
                return await ctx.call_llm(
                    f"Analyse this data and produce a summary:\\n{{json.dumps(data)}}"
                )
    ```

    FILE-AUTHORING TEMPLATE (one step per file, no JSON wrapper, hard guards):
    ```python
    from flowforge import flow, task, step

    @flow(name="{flow_name}", prompt="{flow_prompt}",
          tools=["files_write_text", "files_read_text"])
    class {class_name}:
        @task(name="materialise",
              prompt="Author each project file as plain text and write it",
              tools=["files_write_text", "files_read_text"])
        class Materialise:
            @step(order=1, prompt="Resolve project_dir and shared params")
            async def init(ctx):
                raw = ctx.input
                if hasattr(raw, "model_dump"):
                    raw = raw.model_dump()
                params = raw if isinstance(raw, dict) else {{}}
                ctx.shared_data["project_dir"] = params.get(
                    "project_dir", "./out_project"
                )
                return {{"ok": True}}

            @step(order=2, prompt="Author the full index.html body as plain text")
            async def write_index(ctx):
                project_dir = ctx.shared_data["project_dir"]
                body = (await ctx.call_llm(
                    "Write the COMPLETE index.html for the homepage. "
                    "Return ONLY the file body as plain text — no markdown "
                    "fences, no commentary, no JSON wrapper. The response "
                    "is written to disk verbatim."
                ) or "").strip()
                if len(body) < 200:
                    raise RuntimeError(
                        f"index.html author returned trivial content "
                        f"(len={{{{len(body)}}}})"
                    )
                result = await ctx.call_tool(
                    "files_write_text",
                    path=f"{{{{project_dir}}}}/index.html",
                    content=body,
                )
                if not result.get("ok"):
                    raise RuntimeError(
                        f"write index.html failed: {{{{result.get('error')}}}}"
                    )
                ctx.shared_data["index_html_len"] = len(body)
                return {{"ok": True, "path": "index.html", "bytes": len(body)}}

            @step(order=3, prompt="Verify the entrypoint file is non-trivial")
            async def verify(ctx):
                project_dir = ctx.shared_data["project_dir"]
                read = await ctx.call_tool(
                    "files_read_text",
                    path=f"{{{{project_dir}}}}/index.html",
                )
                if not read.get("ok"):
                    raise RuntimeError(
                        f"verify read failed: {{{{read.get('error')}}}}"
                    )
                content = read.get("content", "")
                if "<body></body>" in content or len(content) < 200:
                    raise RuntimeError(
                        f"index.html appears to be a stub (len={{{{len(content)}}}})"
                    )
                return {{"ok": True, "verified": True}}
    ```

    COMPLEX TEMPLATE (child flows + tasks):
    ```python
    from flowforge import flow, task, step

    @flow(name="sub_process", prompt="a sub-process")
    class SubProcessFlow:
        @task(name="sub_task", prompt="Perform the sub-process work")
        class SubTask:
            @step(order=1, prompt="Complete the sub-process step")
            async def sub_step(ctx):
                return await ctx.call_llm("do sub work on {{field}}")

    @flow(name="{flow_name}", prompt="{flow_prompt}")
    class {class_name}:
        # Child flow — define at module level, reference as class attribute
        SubProcessFlow = SubProcessFlow

        # Direct task — sibling to the child flow
        @task(name="finalize", prompt="Finalize the result after sub-process output")
        class FinalizeTask:
            @step(order=1, prompt="Summarize sub-process output into the final response")
            async def wrap_up(ctx):
                return await ctx.call_llm("summarize results from {{field}}")
    ```

    Choose the simplest structure that fits the request.
    Generate ONLY the Python code, no markdown fences, no explanation.
""")

_FIX_SYSTEM = textwrap.dedent("""\
    The following FlowForge code failed during compile-time validation or
    runtime execution.  Fix the error and
    return ONLY the corrected Python code (no markdown, no explanation).

    Error:
    {error}

    Original code:
    {code}
""")

_TOOL_CODEGEN_SYSTEM = textwrap.dedent("""\
    You are FlowForge's tool generator. Generate a small Python module that
    exposes exactly one function tool.

    RULES:
    1. Prefer Python standard library code.
    2. Do not install dependencies in the generated code.
    3. If a dependency is needed, include a module-level DEPENDENCIES list.
    4. Define a function named {tool_name} with typed parameters.
    5. Return JSON-serialisable values only.
    6. Generate ONLY Python code, no markdown fences, no explanation.
""")


# Plan-driven code synthesis (Phase 4).  The plan + capability selection
# eliminate most of the freedom from `_CODEGEN_SYSTEM`, so this template is
# focused on translation rules rather than design guidance.
_PLAN_SYNTHESIS_SYSTEM = textwrap.dedent("""\
    You are FlowForge's code synthesiser.  A workflow plan and per-step
    capability decision are provided in the user message — your sole job
    is to translate them into FlowForge decorator code.  Do NOT invent
    new steps, rename them, change orders, or pick different tools.

    REQUIRED STRUCTURE — follow this skeleton EXACTLY.  ``@flow`` and
    ``@task`` decorate CLASSES, never functions.  ``@step`` decorates
    nested ``async def`` functions.  ``@flow`` and ``@task`` BOTH require
    keyword-only ``name=`` and ``prompt=``; there is no ``model=`` argument
    on any decorator.  Skeleton:
    ```python
    from flowforge import flow, task, step
    import json

    @flow(name="<flow_name from plan>", prompt="<flow_prompt from plan>")
    class <TopClass>:
        @task(name="<task_name>", prompt="<task_prompt>")
        class <TaskClass>:
            @step(order=1, prompt="<step1.purpose>", tools=[...])
            async def <step1.name>(ctx):
                ...
                return {"...": ...}

            @step(order=2, prompt="<step2.purpose>")
            async def <step2.name>(ctx):
                ...
    ```
    HARD RULES (violations cause an immediate compile failure that wastes
    a retry attempt):
    - NEVER write ``@flow`` or ``@task`` on an ``async def``.  They MUST
      decorate ``class`` blocks.
    - NEVER omit ``prompt=`` on ``@flow`` or ``@task``.  The plan provides
      both ``flow_prompt`` and ``task_prompt`` — use them verbatim.
    - NEVER pass ``model=``, ``llm_config=``, or any unknown kwarg to a
      FlowForge decorator.  Only the documented kwargs are allowed.
    - NEVER repeat tools=[] on the @flow/@task — declare ``tools=[...]``
      ONLY on each ``@step`` that needs a tool, copying the names from
      the capability decision verbatim.
    - The class names must be valid Python identifiers (PascalCase OK).

    OUTPUT
    - One ``@flow`` class named exactly as ``top_class``.  Its decorator
      uses ``name`` and ``prompt`` from the plan.
    - One ``@task`` inside the flow with ``name=task_name`` and
      ``prompt=task_prompt``.
    - One ``@step`` per planned step, in the given ``order``.  Use the
      step's ``purpose`` as the step ``prompt`` (rephrase only for clarity).
    - All classes at module level.  Step functions are ``async def name(ctx):``.

    TOOL REFERENCES (per the capability decision)
    - mode='llm_only'      → step body calls ``await ctx.call_llm(...)`` with
                              instructions; do NOT add tools=[...] to the
                              decorator.
    - mode='builtin_tool'  → declare ``tools=[...]`` on the @step with the
                              capability's tool_names.  When the work is
                              deterministic, call the tool directly via
                              ``await ctx.call_tool("name", **kwargs)``.
                              When the LLM needs to decide arguments, use
                              ``await ctx.call_llm("instruction <name>")``.
    - mode='claude_skill' / 'agent_skill' →
                              declare ``tools=[...]`` and reference the
                              skill name with ``<name>`` inside a
                              ``ctx.call_llm(...)`` instruction.  Skills are
                              prompt-only — never use ``ctx.call_tool`` for
                              them.
    - mode='mcp'           → emit a setup step at the very start of the
                              flow that calls
                              ``ctx.call_tool("mcp_register_server",
                              server_name="<mcp_server_name>",
                              tool_names="<comma-separated tool_names>")``.
                              In the consuming step, declare ``tools=[...]``
                              with the MCP tool_names and call them through
                              ``ctx.call_llm("instruction <tool_name>")``
                              (LLM-mediated) or ``ctx.call_tool`` (direct).

    BRANCH DISPATCH
    - When a step's plan entry has a ``branch``: emit
      ``@step(order=N, prompt=..., condition=BranchCondition(field="X",
      enum=[...]), branches={{"value": target_step_func, ...}},
      fallback=target_step_func)`` and import ``BranchCondition`` from
      ``flowforge``.  ``target_step_func`` is the bare async function for
      a step defined later in the same task.

    HARD RULES
    - Do NOT create any tool / skill / MCP names that are not listed in
      the capability decision.  Names must be copied verbatim.
    - Do NOT recreate existing flows.  Generate ONLY the missing flow.
    - Do NOT use ``@global_config``.
    - Step functions return JSON-serialisable dicts.
    - When the plan declares a downstream output contract, the FINAL
      step's return value MUST match that JSON Schema's top-level keys.
    - Use ``ctx.previous_results.get(<order>)`` to read upstream output
      (use the integer order, not a name).
    - ``ctx.input`` changes after every step.  For step order > 1, NEVER
      read stable request fields such as ``project_dir``, ``project_root``,
      ``target_url``, ``reference_url``, ``project_name``, or ``request``
      from ``ctx.input``.  Read them from ``ctx.task_input``,
      ``ctx.flow_input``, or values normalised into ``ctx.shared_data`` by
      the first step.
    - The first step should normalise any dict/model/string input into
      ``ctx.shared_data``.  Later steps should use ``ctx.shared_data`` with
      explicit defaults for cross-step fields.
    - Every builtin tool returns a dict.  After deterministic tool calls
      that affect files, installs, builds, shells, or fetches, check
      ``result.get("ok")`` and raise ``RuntimeError`` on failure.  Silent
      ``ok=False`` results are bugs.
    - For ``web_fetch_url`` used to extract page listings or article bodies,
      pass ``max_chars=50000`` or ``max_chars=0``.  If the response is
      truncated or lacks the required real data, raise ``RuntimeError``.
      Never return placeholder records saying the HTML was unavailable,
      truncated, empty, or impossible to parse.
    - ``ctx.call_llm(...)`` may return parsed JSON as a Python ``dict`` or
      ``list`` when the model responds with valid JSON, even without an
      explicit output schema.  Do NOT blindly call string methods
      (``.strip()``, regex search, ``json.loads``) on the result.  Always
      handle ``dict`` / ``list`` first, then parse strings only when the
      result is a string.

    FILE / FRONTEND PROJECT RULES
    - NEVER ask the LLM for all project files as one JSON object, file map,
      or array.  Author one substantial file at a time as plain text, or
      use a separate author step and write step per file when the selected
      capability mode is prompt-only.
    - Do NOT swallow JSON parse failures with ``except: data = {}`` or
      ``except: files = []``.  Raise with the parse error so runtime repair
      can regenerate the code.
    - Do NOT substitute fallback stub file contents.  If an authoring step
      returns empty/trivial content, raise before writing.  Empty
      ``<body></body>``, one-line CSS, placeholder JavaScript, and minimal
      package.json files are failures, not defaults.
    - Before every ``files_write_text`` call, verify the content is a
      non-empty string with a file-appropriate minimum length.  After the
      call, check ``ok`` and raise on failure.
    - File-producing workflows must include a verification step that uses
      ``files_read_text`` to read back at least the entrypoint file and
      assert it is non-trivial.  Clone-coding/build workflows should also
      verify build output after install/build.
    - Binary/document artefacts such as ``.docx``, ``.pptx``, ``.pdf``,
      images, and charts MUST NOT be verified with ``files_read_text``.
      Use the creator tool's returned ``size`` / count fields and, when
      needed, ``files_list_dir`` on the parent directory to confirm the
      file exists and has a non-trivial byte size.  Raise on missing or
      tiny files; do not return warning notes as success.
    - Generate ONLY the Python code — no markdown fences, no narration.
""")
