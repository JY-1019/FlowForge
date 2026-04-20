"""Pydantic models for AI-generated documentation.

These docs are consumed by the AI Planner for route selection.  Field types
are intentionally loose (``Any``) for LLM-generated content — the planner
only needs the information to be present, not strictly typed.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class StepDoc(BaseModel):
    summary: str
    input_schema_desc: Any = None
    output_schema_desc: Any = None
    preconditions: Any = Field(default_factory=list)
    capabilities: Any = Field(default_factory=list)


class BranchDoc(BaseModel):
    summary: str
    input_schema_desc: Any = None
    output_schema_desc: Any = None
    preconditions: Any = Field(default_factory=list)
    capabilities: Any = Field(default_factory=list)
    routing_hints: Any = None


class TaskDoc(BaseModel):
    summary: str
    input_schema_desc: Any = None
    output_schema_desc: Any = None
    preconditions: Any = Field(default_factory=list)
    capabilities: Any = Field(default_factory=list)
    children_overview: str = ""


class FlowDoc(BaseModel):
    summary: str
    input_schema_desc: Any = None
    output_schema_desc: Any = None
    preconditions: Any = Field(default_factory=list)
    capabilities: Any = Field(default_factory=list)
    children_overview: str = ""


class GlobalDoc(BaseModel):
    summary: str
    capabilities: Any = Field(default_factory=list)
    agent_overview: str = ""


AnyDoc = StepDoc | BranchDoc | TaskDoc | FlowDoc | GlobalDoc
