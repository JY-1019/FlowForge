"""Provider-agnostic LLM tool-call helper.

All LLM calls in FlowForge (doc generation, planning) go through
``call_with_tool()``.  It converts the Anthropic-native tool schema format
to each provider's expected format and normalises the response back to a
plain ``dict``.

Supported providers
-------------------
* ``"anthropic"`` — Anthropic Messages API (``anthropic>=0.40``).
  Installed by default.
* ``"openai"``    — OpenAI Chat Completions API (``openai>=1.50``).
  Install: ``pip install flowforge[openai]``
* ``"google"``    — Google Generative AI / Gemini (``google-generativeai>=0.8``).
  Install: ``pip install flowforge[google]``

Tool schema format (Anthropic-native, used as the canonical input)
-------------------------------------------------------------------
::

    {
        "name": "my_tool",
        "description": "What this tool does",
        "input_schema": {
            "type": "object",
            "properties": {"field": {"type": "string", ...}},
            "required": ["field"],
        },
    }

This format is automatically translated before calling OpenAI or Google.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.types import LLMConfig

logger = logging.getLogger(__name__)


def _get_ssl_verify(llm_config: LLMConfig | None = None) -> bool:
    """Return ``False`` when the user has opted out of SSL verification.

    Checks two sources (either one disables verification):

    1. ``LLMConfig.verify_ssl = False``
    2. Environment variable ``FLOWFORGE_SSL_VERIFY=0`` (or ``false``)

    Corporate proxies (e.g. LG CNS, ZScaler) often inject self-signed CA
    certificates whose ``Basic Constraints`` extension is not marked
    *critical*.  Modern OpenSSL (3.x) rejects these, breaking API calls.
    """
    # LLMConfig option takes precedence
    if llm_config is not None and not llm_config.verify_ssl:
        return False
    # Environment variable fallback
    val = os.environ.get("FLOWFORGE_SSL_VERIFY", "1").lower()
    return val not in ("0", "false", "no", "off")


async def call_with_tool(
    *,
    prompt: str,
    tool_schema: dict[str, Any],
    llm_config: LLMConfig,
    system_prompt: str = "",
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Call an LLM with a single forced tool invocation and return the tool arguments.

    The function forces the model to call exactly the tool named in
    ``tool_schema["name"]`` (using each provider's equivalent of
    ``tool_choice``).  The raw tool-call arguments are returned as a plain
    ``dict``.

    Parameters
    ----------
    prompt:
        The user-turn content sent to the model.
    tool_schema:
        Anthropic-native tool definition (see module docstring).
    llm_config:
        Provider, model, API key, and generation settings.
    system_prompt:
        Optional system / developer message prepended to the conversation.
    max_tokens:
        Maximum number of tokens the model may generate.

    Returns
    -------
    dict[str, Any]
        The parsed tool-call arguments dict.

    Raises
    ------
    ImportError
        If the required provider SDK is not installed.
    RuntimeError
        If the model did not call the expected tool.
    """
    provider = llm_config.provider
    logger.debug("call_with_tool provider=%s model=%s", provider, llm_config.model)

    if provider == "anthropic":
        return await _call_anthropic(
            prompt=prompt,
            tool_schema=tool_schema,
            llm_config=llm_config,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        return await _call_openai(
            prompt=prompt,
            tool_schema=tool_schema,
            llm_config=llm_config,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    elif provider == "google":
        return await _call_google(
            prompt=prompt,
            tool_schema=tool_schema,
            llm_config=llm_config,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            "Must be one of: 'anthropic', 'openai', 'google'."
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

async def _call_anthropic(
    *,
    prompt: str,
    tool_schema: dict[str, Any],
    llm_config: LLMConfig,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Dispatch to the Anthropic Messages API using tool_use."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The 'anthropic' package is required for provider='anthropic'. "
            "It is a core FlowForge dependency — run: pip install flowforge"
        ) from e

    api_key = llm_config.api_key
    kwargs: dict[str, Any] = {}
    if llm_config.base_url:
        kwargs["base_url"] = llm_config.base_url

    if not _get_ssl_verify(llm_config):
        import httpx
        kwargs["http_client"] = httpx.Client(verify=False)

    client = anthropic.Anthropic(api_key=api_key, **kwargs) if api_key else anthropic.Anthropic(**kwargs)

    tool_name = tool_schema["name"]
    create_kwargs: dict[str, Any] = dict(
        model=llm_config.model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": tool_name},
    )
    if system_prompt:
        create_kwargs["system"] = system_prompt

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(**create_kwargs),
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input  # type: ignore[return-value]

    raise RuntimeError(
        f"Anthropic model did not call tool '{tool_name}'. "
        f"Stop reason: {response.stop_reason}"
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

async def _call_openai(
    *,
    prompt: str,
    tool_schema: dict[str, Any],
    llm_config: LLMConfig,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Dispatch to the OpenAI Chat Completions API using function calling.

    Translates the Anthropic-native tool schema to OpenAI's
    ``{"type": "function", "function": {...}}`` format automatically.
    """
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "The 'openai' package is required for provider='openai'. "
            "Install it with: pip install flowforge[openai]"
        ) from e

    api_key   = llm_config.api_key
    base_url  = llm_config.base_url
    init_kwargs: dict[str, Any] = {}
    if api_key:
        init_kwargs["api_key"] = api_key
    if base_url:
        init_kwargs["base_url"] = base_url

    if not _get_ssl_verify(llm_config):
        import httpx
        init_kwargs["http_client"] = httpx.AsyncClient(verify=False)

    client = openai.AsyncOpenAI(**init_kwargs)

    tool_name = tool_schema["name"]

    # Convert Anthropic tool schema → OpenAI function calling schema.
    openai_tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_schema.get("description", ""),
            "parameters": tool_schema.get("input_schema", {}),
        },
    }

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=llm_config.model,
        max_tokens=max_tokens,
        messages=messages,
        tools=[openai_tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=llm_config.temperature,
    )

    choice = response.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    for tc in tool_calls:
        if tc.function.name == tool_name:
            return json.loads(tc.function.arguments)

    raise RuntimeError(
        f"OpenAI model did not call function '{tool_name}'. "
        f"Finish reason: {choice.finish_reason}"
    )


# ---------------------------------------------------------------------------
# Google Gemini  (uses the new google-genai SDK >= 1.0)
# ---------------------------------------------------------------------------

# Fields that Gemini's Schema proto does not accept.
_GEMINI_UNSUPPORTED = frozenset({
    "additionalProperties", "default", "$schema", "title",
    "$defs", "definitions", "examples", "const",
})

_TYPE_MAP = {
    "object":  "OBJECT",
    "array":   "ARRAY",
    "string":  "STRING",
    "number":  "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
}


def _json_schema_to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema dict to the format the Gemini SDK expects.

    - Uppercases type names (``"object"`` → ``"OBJECT"``)
    - Strips fields that Gemini does not support (``additionalProperties``, etc.)
    - Recurses into ``properties`` and ``items``
    """
    result = {k: v for k, v in schema.items() if k not in _GEMINI_UNSUPPORTED}

    if "type" in result:
        result["type"] = _TYPE_MAP.get(result["type"].lower(), result["type"].upper())

    if "properties" in result:
        result["properties"] = {
            k: _json_schema_to_gemini_schema(v)
            for k, v in result["properties"].items()
        }

    if "items" in result:
        result["items"] = _json_schema_to_gemini_schema(result["items"])

    return result


async def _call_google(
    *,
    prompt: str,
    tool_schema: dict[str, Any],
    llm_config: LLMConfig,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Dispatch to the Gemini API using the new ``google-genai`` SDK.

    * Translates Anthropic-native tool schema to Gemini's FunctionDeclaration.
    * Strips unsupported JSON Schema fields (``additionalProperties``, etc.).
    * Retries automatically on 429 rate-limit errors (up to 3 times, 15 / 30 / 60 s).

    Requires ``pip install flowforge[google]`` (``google-genai>=1.0``).
    """
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as e:
        raise ImportError(
            "The 'google-genai' package is required for provider='google'. "
            "Install it with: pip install flowforge[google]"
        ) from e

    api_key = llm_config.api_key
    genai_kwargs: dict[str, Any] = {}
    if api_key:
        genai_kwargs["api_key"] = api_key
    if not _get_ssl_verify(llm_config):
        import httpx
        genai_kwargs["http_options"] = {"client": httpx.AsyncClient(verify=False)}
    client = genai.Client(**genai_kwargs)

    tool_name   = tool_schema["name"]
    input_schema = _json_schema_to_gemini_schema(tool_schema.get("input_schema", {}))

    func_decl = gtypes.FunctionDeclaration(
        name=tool_name,
        description=tool_schema.get("description", ""),
        parameters=input_schema,
    )

    config_kwargs: dict[str, Any] = dict(
        tools=[gtypes.Tool(function_declarations=[func_decl])],
        tool_config=gtypes.ToolConfig(
            function_calling_config=gtypes.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[tool_name],
            )
        ),
        temperature=llm_config.temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    gen_config = gtypes.GenerateContentConfig(**config_kwargs)

    # Retry on 429 / quota errors with exponential backoff.
    _MAX_RETRIES = 3
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.aio.models.generate_content(
                model=llm_config.model,
                contents=prompt,
                config=gen_config,
            )
            break
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "quota" in err.lower() or "rate" in err.lower()
            if is_rate_limit and attempt < _MAX_RETRIES:
                wait = 15 * (2 ** attempt)   # 15 s → 30 s → 60 s
                logger.warning(
                    "Gemini rate limit (attempt %d/%d) — retrying in %d s …",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise

    # Extract the function-call arguments from the response.
    for candidate in response.candidates:
        for part in candidate.content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name == tool_name:
                return dict(fc.args)

    raise RuntimeError(
        f"Gemini model did not call function '{tool_name}'. "
        f"Response: {response!r}"
    )
