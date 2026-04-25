"""Core type definitions for FlowForge."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration for a flow, task, or global context.

    Supports three LLM providers out of the box:

    * **Anthropic** (default) — Claude models.
      Required package: ``anthropic>=0.40`` (already a core dependency).
    * **OpenAI** — ChatGPT / GPT-4 models.
      Required package: ``openai>=1.50`` (install with ``pip install flowforge[openai]``).
    * **Google** — Gemini models.
      Required package: ``google-generativeai>=0.8``
      (install with ``pip install flowforge[google]``).

    Factory helpers
    ---------------
    Use the classmethods for a one-liner setup::

        LLMConfig.for_claude()   # claude-sonnet-4-6, temperature 0.3
        LLMConfig.for_openai()   # gpt-4o, temperature 0.3
        LLMConfig.for_gemini()   # gemini-2.0-flash, temperature 0.3

    Parameters
    ----------
    provider:
        Which LLM backend to use: ``"anthropic"``, ``"openai"``, or ``"google"``.
    model:
        Model identifier passed directly to the provider SDK.
        Defaults to ``"claude-sonnet-4-6"`` for the ``"anthropic"`` provider.
    temperature:
        Sampling temperature (0 = deterministic, 1 = creative).
    max_tokens:
        Maximum tokens in the model response.
    api_key:
        Provider API key.  When ``None`` the SDK reads from the appropriate
        environment variable (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, or
        ``GOOGLE_API_KEY``).
    base_url:
        Custom base URL for the HTTP client.  Useful for OpenAI-compatible
        proxy endpoints or local model servers.
    """

    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    verify_ssl: bool = True

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Factory classmethods — convenience constructors per provider
    # ------------------------------------------------------------------

    @classmethod
    def for_claude(
        cls,
        model: str = "claude-sonnet-4-6",
        **kwargs: Any,
    ) -> "LLMConfig":
        """Return an ``LLMConfig`` targeting Anthropic Claude.

        Parameters
        ----------
        model:
            Claude model ID (e.g. ``"claude-opus-4-6"``,
            ``"claude-haiku-4-5-20251001"``).
        **kwargs:
            Any other ``LLMConfig`` fields (``api_key``, ``temperature``, …).
        """
        return cls(provider="anthropic", model=model, **kwargs)

    @classmethod
    def for_openai(
        cls,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> "LLMConfig":
        """Return an ``LLMConfig`` targeting OpenAI ChatGPT / GPT-4.

        Requires ``pip install flowforge[openai]``.

        Parameters
        ----------
        model:
            OpenAI model ID (e.g. ``"gpt-4o"``, ``"gpt-4-turbo"``,
            ``"o3"``).
        **kwargs:
            Any other ``LLMConfig`` fields (``api_key``, ``base_url``, …).
        """
        return cls(provider="openai", model=model, **kwargs)

    @classmethod
    def for_gemini(
        cls,
        model: str = "gemini-2.0-flash",
        **kwargs: Any,
    ) -> "LLMConfig":
        """Return an ``LLMConfig`` targeting Google Gemini.

        Requires ``pip install flowforge[google]``.

        Parameters
        ----------
        model:
            Gemini model ID (e.g. ``"gemini-2.0-flash"``,
            ``"gemini-1.5-pro"``).
        **kwargs:
            Any other ``LLMConfig`` fields (``api_key``, …).
        """
        return cls(provider="google", model=model, **kwargs)


class BranchCondition(BaseModel):
    """Condition for a branch decorator."""

    field: str
    enum: list[str]


class MCPServer(BaseModel):
    """MCP server tool configuration."""

    url: str
    name: str = ""
    description: str = ""


class FunctionTool(BaseModel):
    """Python function tool configuration."""

    func: Any
    name: str = ""
    description: str = ""

    model_config = {"arbitrary_types_allowed": True}


class HTTPTool(BaseModel):
    """HTTP API tool configuration."""

    url: str
    method: str = "POST"
    name: str = ""
    description: str = ""
    headers: dict[str, str] = Field(default_factory=dict)


ToolConfig = MCPServer | FunctionTool | HTTPTool


class DependencyPolicy(BaseModel):
    """Policy for dependencies requested by dynamically generated tools.

    Dynamic generation can discover that a missing capability needs an
    additional package.  The policy is intentionally explicit: generation can
    describe dependencies freely, but installation is gated by this model.
    """

    allow_install: bool = False
    allowed_managers: list[str] = Field(
        default_factory=lambda: ["pip", "uv", "npm", "pnpm", "yarn"]
    )
    allowed_packages: list[str] = Field(default_factory=list)
    denied_packages: list[str] = Field(default_factory=list)


class DynamicRunOptions(BaseModel):
    """Runtime options for dynamic flow/tool generation.

    These options are accepted by ``FlowForge.compile()`` and ``engine.run()``.
    ``@global_config(dynamic_flow=True)`` still declares that the agent is
    allowed to use dynamic generation; this object controls where generated
    code lives and what extra capabilities the dynamic generator may use.
    """

    enabled: bool = True
    project_root: str | None = None
    generated_dir: str = "flowforge/generated"
    persist_generated: bool = True
    auto_load_generated: bool = True
    include_builtin_tools: bool = True
    allow_tool_generation: bool = False
    allow_codegen_tool_use: bool = False
    allowed_shell_modes: list[
        Literal["readonly", "workspace_write", "project_exec", "install_dependency"]
    ] = Field(default_factory=lambda: ["readonly", "project_exec"])
    shell_timeout_seconds: int = 60
    shell_output_max_chars: int = 4000
    mcp_server_commands: dict[str, list[str]] = Field(default_factory=dict)
    mcp_start_timeout_seconds: int = 15
    project_context_max_chars: int = 4000
    max_requirements: int = 8
    dependency_policy: DependencyPolicy = Field(default_factory=DependencyPolicy)
