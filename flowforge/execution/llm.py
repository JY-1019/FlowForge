"""LLM call helpers for StepContext.call_llm().

Prompt templating
-----------------
``render_prompt(template, input_data)`` replaces ``{field_name}`` placeholders
with values from the step input (Pydantic model or dict).

Tool reference parsing
----------------------
``parse_tool_refs(prompt)`` extracts ``<tool_name>`` markers from the prompt
text and returns (cleaned_prompt, tool_names).

LLM API call
------------
``call_llm_api(...)`` dispatches to the configured provider (Anthropic, OpenAI,
Google) using the ``LLMConfig``.  Tool configs are converted to the provider's
tool-use format before the call.

When the LLM returns ``tool_use`` blocks, the tool-use loop executes each
tool call (MCP, FunctionTool, HTTPTool) and feeds the results back to the
LLM until it produces a final text or structured-output response.
"""
from __future__ import annotations

import json as _json
import logging
import re
import textwrap
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.types import LLMConfig, ToolConfig
    from flowforge.tools.registry import ToolRegistry
    from flowforge.execution.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

# Maximum number of tool-use round-trips before forcing a final answer.
_MAX_TOOL_ROUNDS = 25


# ---------------------------------------------------------------------------
# Prompt templating
# ---------------------------------------------------------------------------

def render_prompt(template: str, input_data: Any) -> str:
    """Replace ``{field_name}`` placeholders with values from *input_data*.

    Supports both Pydantic models (attribute access) and dicts.  Missing
    fields are left as-is (``{unknown}`` remains literal).

    Triple-quoted strings are handled automatically: leading/trailing blank
    lines are stripped and common indentation is removed via
    ``textwrap.dedent``, so you can write::

        await ctx.call_llm(\"\"\"
            Search for {query} in {language}.

            Use <web_search> to find results.
        \"\"\")

    and get a clean, dedented prompt.

    Parameters
    ----------
    template:
        The prompt string containing ``{field}`` placeholders.
    input_data:
        The step input — a Pydantic ``BaseModel`` instance or a ``dict``.

    Returns
    -------
    str
        The rendered prompt with placeholders replaced.
    """
    # Auto-dedent: strip common leading whitespace from triple-quoted strings.
    template = textwrap.dedent(template).strip()

    if input_data is None:
        return template

    def _get_value(match: re.Match[str]) -> str:
        field = match.group(1)
        if isinstance(input_data, dict):
            val = input_data.get(field)
        else:
            val = getattr(input_data, field, None)
        if val is None:
            return match.group(0)  # leave placeholder as-is
        return str(val)

    return re.sub(r"\{(\w+)\}", _get_value, template)


# ---------------------------------------------------------------------------
# Tool reference parsing
# ---------------------------------------------------------------------------

def parse_tool_refs(prompt: str) -> tuple[str, list[str]]:
    """Extract ``<tool_name>`` markers from *prompt*.

    Each ``<tool_name>`` marker indicates that the named tool should be
    included in the LLM call's ``tools`` parameter.  The markers are
    removed from the returned prompt text.

    Parameters
    ----------
    prompt:
        Raw prompt possibly containing ``<tool_name>`` references.

    Returns
    -------
    tuple[str, list[str]]
        ``(cleaned_prompt, tool_names)`` — the prompt without markers and
        the list of tool names extracted (in order of appearance, deduplicated).
    """
    tool_names: list[str] = []
    seen: set[str] = set()

    def _collect(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in seen:
            tool_names.append(name)
            seen.add(name)
        return ""  # remove marker from prompt

    # Match <word_chars> but not HTML-like tags (e.g. </close> or <br/>).
    cleaned = re.sub(r"<(\w+)>", _collect, prompt)
    # Clean up extra whitespace left by marker removal.
    cleaned = re.sub(r"  +", " ", cleaned).strip()

    return cleaned, tool_names


# ---------------------------------------------------------------------------
# LLM API call
# ---------------------------------------------------------------------------

def _tool_config_to_anthropic(tool_config: Any) -> dict[str, Any]:
    """Convert a ``ToolConfig`` to an Anthropic tool-use schema dict."""
    from flowforge.types import MCPServer, FunctionTool, HTTPTool

    if isinstance(tool_config, FunctionTool):
        import inspect
        name = tool_config.name or tool_config.func.__name__
        desc = tool_config.description or (inspect.getdoc(tool_config.func) or f"Call {name}")
        return {
            "name": name,
            "description": desc,
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    elif isinstance(tool_config, MCPServer):
        return {
            "name": tool_config.name or "mcp_tool",
            "description": tool_config.description or f"MCP server at {tool_config.url}",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    elif isinstance(tool_config, HTTPTool):
        return {
            "name": tool_config.name or "http_tool",
            "description": tool_config.description or f"HTTP {tool_config.method} {tool_config.url}",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    # Fallback for ToolAdapter instances from registry
    if hasattr(tool_config, "to_anthropic_tool"):
        return tool_config.to_anthropic_tool()
    return {}


def _pydantic_to_json_schema(schema: type) -> dict[str, Any]:
    """Convert a Pydantic BaseModel class to a JSON Schema dict for tool_use.

    Strips Pydantic-internal keys (``title``, ``$defs``) so the schema is
    clean for LLM tool definitions.
    """
    raw = schema.model_json_schema()

    # Inline $defs references for a flat, self-contained schema.
    defs = raw.pop("$defs", None) or raw.pop("definitions", None)
    if defs:
        import json
        raw_str = json.dumps(raw)
        for def_name, def_schema in defs.items():
            ref = f'{{"$ref": "#/$defs/{def_name}"}}'
            raw_str = raw_str.replace(ref, json.dumps(def_schema))
            ref_alt = f'{{"$ref": "#/definitions/{def_name}"}}'
            raw_str = raw_str.replace(ref_alt, json.dumps(def_schema))
        raw = json.loads(raw_str)

    raw.pop("title", None)
    return raw


async def _build_executor(
    tool_configs: list[ToolConfig] | None,
) -> tuple[ToolExecutor | None, dict[str, dict[str, Any]]]:
    """Create a ``ToolExecutor`` and fetch MCP schemas if needed.

    Returns ``(executor, mcp_schemas)`` where *mcp_schemas* is a mapping
    from tool name to an Anthropic-format tool schema with the real
    ``input_schema`` obtained from the MCP server.
    """
    if not tool_configs:
        return None, {}

    from flowforge.execution.tool_executor import ToolExecutor

    executor = ToolExecutor(tool_configs)
    mcp_schemas: dict[str, dict[str, Any]] = {}

    # Fetch real schemas from MCP servers (with proper inputSchema).
    from flowforge.types import MCPServer
    has_mcp = any(isinstance(tc, MCPServer) for tc in tool_configs)
    if has_mcp:
        try:
            mcp_schemas = await executor.fetch_mcp_tool_schemas()
            logger.info(
                "MCP tool schemas fetched: %s",
                list(mcp_schemas.keys()) or "none",
            )
        except Exception as e:
            logger.warning("Failed to fetch MCP schemas: %s", e)

    return executor, mcp_schemas


def _get_tool_name(tc: Any) -> str:
    """Extract a tool name from a ``ToolConfig`` instance."""
    from flowforge.types import MCPServer, FunctionTool, HTTPTool

    if isinstance(tc, MCPServer):
        return tc.name
    elif isinstance(tc, FunctionTool):
        return tc.name or (tc.func.__name__ if hasattr(tc.func, "__name__") else "")
    elif isinstance(tc, HTTPTool):
        return tc.name or "http_tool"
    return ""


async def call_llm_api(
    *,
    system_prompt: str,
    user_prompt: str,
    llm_config: LLMConfig,
    tool_configs: list[ToolConfig] | None = None,
    tool_registry: ToolRegistry | None = None,
    output_schema: type | None = None,
) -> Any:
    """Call the LLM using the configured provider.

    When tools are provided (MCP, FunctionTool, HTTPTool), a **tool-use loop**
    is executed: the LLM may call tools, the framework executes them and feeds
    results back, repeating until the LLM produces a final answer.

    Parameters
    ----------
    system_prompt:
        System-level instruction (from the ``@step`` annotation prompt).
    user_prompt:
        User/task prompt (from inside the step function, already rendered).
    llm_config:
        LLM configuration (provider, model, temperature, etc.).
    tool_configs:
        Tool configs resolved from ``<tool_name>`` references in the prompt.
    tool_registry:
        Global tool registry (used to look up ToolAdapter instances).
    output_schema:
        Optional Pydantic ``BaseModel`` subclass.  When provided the LLM is
        forced to return structured output matching this schema via tool_use
        (Anthropic) or structured outputs / function calling (OpenAI/Google).

    Returns
    -------
    Any
        The LLM response — a ``dict`` when *output_schema* is set (parsed
        from the tool call), otherwise a plain text string.
    """
    # Build executor and fetch real MCP schemas.
    executor, mcp_schemas = await _build_executor(tool_configs)

    # Build tools payload — prefer real MCP schemas over stubs.
    tools_payload: list[dict[str, Any]] = []
    for tc in (tool_configs or []):
        name = _get_tool_name(tc)
        if name and name in mcp_schemas:
            tools_payload.append(mcp_schemas[name])
        else:
            schema = _tool_config_to_anthropic(tc)
            if schema:
                tools_payload.append(schema)

    # ── Request logging ──────────────────────────────────────────────
    tool_names = [t.get("name", "?") for t in tools_payload]
    schema_name = output_schema.__name__ if output_schema else None
    logger.info(
        "LLM request  provider=%s model=%s tools=%s output_schema=%s",
        llm_config.provider, llm_config.model, tool_names or None, schema_name,
    )
    logger.debug("LLM system_prompt:\n%s", system_prompt)
    logger.debug("LLM user_prompt:\n%s", user_prompt)
    if output_schema:
        logger.debug(
            "LLM structured_output schema: %s",
            _json.dumps(_pydantic_to_json_schema(output_schema), ensure_ascii=False, indent=2),
        )

    t0 = time.monotonic()

    if llm_config.provider == "anthropic":
        result = await _call_anthropic(
            system_prompt, user_prompt, llm_config, tools_payload,
            executor=executor, output_schema=output_schema,
        )
    elif llm_config.provider == "openai":
        result = await _call_openai(
            system_prompt, user_prompt, llm_config, tools_payload,
            executor=executor, output_schema=output_schema,
        )
    elif llm_config.provider == "google":
        result = await _call_google(
            system_prompt, user_prompt, llm_config, tools_payload,
            output_schema=output_schema,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    elapsed = (time.monotonic() - t0) * 1000

    # ── Response logging ─────────────────────────────────────────────
    if isinstance(result, dict):
        result_preview = _json.dumps(result, ensure_ascii=False, default=str)
        if len(result_preview) > 500:
            result_preview = result_preview[:500] + "..."
        logger.info("LLM response  %.0fms  type=dict  preview=%s", elapsed, result_preview)
    elif isinstance(result, str):
        preview = result[:300].replace("\n", "\\n")
        if len(result) > 300:
            preview += "..."
        logger.info("LLM response  %.0fms  type=str  len=%d  preview=%s", elapsed, len(result), preview)
    else:
        logger.info("LLM response  %.0fms  type=%s", elapsed, type(result).__name__)

    return result


# ---------------------------------------------------------------------------
# Anthropic provider — with tool-use loop
# ---------------------------------------------------------------------------

async def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    tools: list[dict[str, Any]],
    executor: ToolExecutor | None = None,
    output_schema: type | None = None,
) -> Any:
    """Call the Anthropic Messages API with a tool-use loop.

    When the model returns ``tool_use`` blocks the executor runs each tool
    and the results are fed back as ``tool_result`` messages.  The loop
    continues until the model produces a final text answer, calls the
    ``structured_output`` tool, or ``_MAX_TOOL_ROUNDS`` is reached.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=config.api_key or None,
        base_url=config.base_url or None,
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system": system_prompt,
    }

    # -- Structured output via synthetic tool --------------------------------
    _structured_tool_name = "structured_output"
    all_tools = list(tools)

    if output_schema is not None:
        json_schema = _pydantic_to_json_schema(output_schema)
        structured_tool = {
            "name": _structured_tool_name,
            "description": (
                f"Return the result as structured JSON matching the "
                f"{output_schema.__name__} schema. You MUST call this tool "
                f"with your final answer."
            ),
            "input_schema": json_schema,
        }
        all_tools = [structured_tool] + all_tools

    if all_tools:
        kwargs["tools"] = all_tools

    # If no executable tools exist, force structured_output directly.
    has_executable = executor is not None and any(
        executor.has_tool(t.get("name", "")) for t in tools
    )
    if output_schema and not has_executable:
        kwargs["tool_choice"] = {"type": "tool", "name": _structured_tool_name}

    # -- Tool-use loop -------------------------------------------------------
    for round_idx in range(_MAX_TOOL_ROUNDS):
        kwargs["messages"] = messages
        response = await client.messages.create(**kwargs)

        # Log raw response
        logger.debug(
            "Anthropic response  round=%d stop=%s usage=%s",
            round_idx, response.stop_reason, getattr(response, "usage", None),
        )
        for i, block in enumerate(response.content):
            btype = getattr(block, "type", "?")
            if btype == "text":
                logger.debug("  content[%d] text=%s", i, block.text[:200].replace("\n", "\\n"))
            elif btype == "tool_use":
                logger.debug("  content[%d] tool_use=%s  input=%s", i, block.name, str(block.input)[:200])

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        # Check for structured_output call → return immediately.
        if output_schema is not None:
            for block in tool_uses:
                if block.name == _structured_tool_name:
                    return block.input

        # Separate executable tool calls from unknown ones.
        executable = [b for b in tool_uses if executor and executor.has_tool(b.name)]
        unknown = [b for b in tool_uses if b not in executable]

        if not executable:
            # No more tools to execute — return text.
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            if text_parts:
                return "\n".join(text_parts)
            # No text either — might be the last round, break.
            break

        # Serialize assistant content blocks for message history.
        assistant_content: list[dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", "?")
            if btype == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool and collect results.
        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            if block in executable:
                logger.info("Tool call  round=%d  %s(%s)", round_idx, block.name, str(block.input)[:200])
                result_text = await executor.execute(block.name, block.input)
                preview = result_text[:500].replace("\n", "\\n") if result_text else ""
                logger.info("Tool result  %s → %s", block.name, preview)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            else:
                # Unknown tool — return error so LLM can self-correct.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: Tool '{block.name}' not found.",
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

        # After feeding results, remove forced tool_choice so the LLM can
        # decide freely on the next turn (call more tools or finish).
        kwargs.pop("tool_choice", None)

    # -- Loop exhausted: force structured output if needed -------------------
    if output_schema is not None:
        logger.info("Tool loop exhausted (%d rounds), forcing structured_output", _MAX_TOOL_ROUNDS)
        kwargs["messages"] = messages
        kwargs["tool_choice"] = {"type": "tool", "name": _structured_tool_name}
        response = await client.messages.create(**kwargs)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _structured_tool_name:
                return block.input

    # Fallback: extract text from last response.
    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    return "\n".join(text_parts) if text_parts else ""


# ---------------------------------------------------------------------------
# OpenAI provider — with tool-use loop
# ---------------------------------------------------------------------------

async def _call_openai(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    tools: list[dict[str, Any]],
    executor: ToolExecutor | None = None,
    output_schema: type | None = None,
) -> Any:
    """Call the OpenAI Chat Completions API with a tool-use loop."""
    import openai

    client = openai.AsyncOpenAI(
        api_key=config.api_key or None,
        base_url=config.base_url or None,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    _structured_fn_name = "structured_output"
    openai_tools: list[dict[str, Any]] = []

    if output_schema is not None:
        json_schema = _pydantic_to_json_schema(output_schema)
        openai_tools.append({
            "type": "function",
            "function": {
                "name": _structured_fn_name,
                "description": f"Return structured JSON matching {output_schema.__name__}.",
                "parameters": json_schema,
            },
        })

    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })

    if openai_tools:
        kwargs["tools"] = openai_tools

    has_executable = executor is not None and any(
        executor.has_tool(t.get("name", "")) for t in tools
    )
    if output_schema and not has_executable:
        kwargs["tool_choice"] = {"type": "function", "function": {"name": _structured_fn_name}}

    for round_idx in range(_MAX_TOOL_ROUNDS):
        kwargs["messages"] = messages
        response = await client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        # Check for structured_output.
        if output_schema is not None:
            for tc in msg.tool_calls:
                if tc.function.name == _structured_fn_name:
                    return _json.loads(tc.function.arguments)

        # Separate executable from unknown.
        executable_ids = set()
        if executor:
            for tc in msg.tool_calls:
                if executor.has_tool(tc.function.name):
                    executable_ids.add(tc.id)

        if not executable_ids:
            return msg.content

        messages.append(msg.model_dump(exclude_unset=True))

        for tc in msg.tool_calls:
            if tc.id in executable_ids:
                args = _json.loads(tc.function.arguments)
                logger.info("Tool call  round=%d  %s(%s)", round_idx, tc.function.name, str(args)[:200])
                result_text = await executor.execute(tc.function.name, args)
                logger.info("Tool result  %s → %s", tc.function.name, result_text[:500])
            else:
                result_text = f"Error: Tool '{tc.function.name}' not found."

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

        kwargs.pop("tool_choice", None)

    # Force structured output if loop exhausted.
    if output_schema is not None:
        kwargs["messages"] = messages
        kwargs["tool_choice"] = {"type": "function", "function": {"name": _structured_fn_name}}
        response = await client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == _structured_fn_name:
                    return _json.loads(tc.function.arguments)

    return msg.content if msg else ""


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

async def _call_google(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    tools: list[dict[str, Any]],
    output_schema: type | None = None,
) -> Any:
    """Call the Google Gemini API."""
    import google.generativeai as genai

    if config.api_key:
        genai.configure(api_key=config.api_key)

    model = genai.GenerativeModel(
        model_name=config.model,
        system_instruction=system_prompt,
    )

    gen_config_kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
    }
    if output_schema is not None:
        gen_config_kwargs["response_mime_type"] = "application/json"
        gen_config_kwargs["response_schema"] = output_schema

    response = await model.generate_content_async(
        user_prompt,
        generation_config=genai.types.GenerationConfig(**gen_config_kwargs),
    )

    # Google with response_schema returns JSON text — parse it.
    if output_schema is not None:
        import json
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, ValueError):
            pass

    return response.text
