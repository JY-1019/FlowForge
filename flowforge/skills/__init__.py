"""Vendored Agent Skills bundled with FlowForge.

Anthropic publishes official Claude Skills as ``SKILL.md`` files in
``github.com/anthropics/skills``.  Those skills are open-format Markdown — they
are *not* part of the Anthropic Messages API hosted-skills catalog (which only
accepts a small set of names like ``pdf`` or ``pptx``), so they must be loaded
locally as ``AgentSkill`` instances.

FlowForge ships a curated subset of those upstream skills under
``flowforge/skills/anthropic/<name>/`` (see
:data:`flowforge.skills.sync.DEFAULT_BUNDLED_SKILLS`).  Any other upstream
skill can be vendored on demand via ``flowforge skills sync <name>`` or the
:func:`flowforge.skills.sync.sync_skill` helper.

    from flowforge import AgentSkill
    from flowforge.skills import bundled_skill_path

    AgentSkill(path=bundled_skill_path("frontend-design"), name="frontend-design")
"""
from __future__ import annotations

from pathlib import Path

from flowforge.skills.sync import (
    DEFAULT_BUNDLED_SKILLS,
    sync_default_skills,
    sync_skill,
    validate_skill_name,
)

_BUNDLED_ROOT = Path(__file__).resolve().parent / "anthropic"
_NATIVE_ROOT = Path(__file__).resolve().parent / "native"


def native_skill_path(name: str) -> Path:
    """Return the local path to a FlowForge-authored native Agent Skill.

    Native skills live under ``flowforge/skills/native/<name>/`` and ship with
    FlowForge itself (they are not vendored from the upstream Anthropic
    repository).  Raises ``FileNotFoundError`` if *name* is unknown.
    """
    name = validate_skill_name(name)
    path = _NATIVE_ROOT / name
    if not (path / "SKILL.md").is_file():
        available = ", ".join(native_skill_names()) or "(none)"
        raise FileNotFoundError(
            f"No native FlowForge skill named {name!r}. Available: {available}."
        )
    return path


def native_skill_names() -> list[str]:
    """Return the names of all FlowForge-authored native Agent Skills."""
    if not _NATIVE_ROOT.is_dir():
        return []
    return sorted(
        child.name
        for child in _NATIVE_ROOT.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )


def skill_path(name: str) -> Path:
    """Resolve *name* to a skill directory, native skills taking precedence.

    Checks FlowForge native skills first, then vendored Anthropic skills.
    Raises ``FileNotFoundError`` if neither has *name*.
    """
    name = validate_skill_name(name)
    if (_NATIVE_ROOT / name / "SKILL.md").is_file():
        return _NATIVE_ROOT / name
    return bundled_skill_path(name)


def bundled_skill_path(name: str) -> Path:
    """Return the local path to a vendored Anthropic Agent Skill.

    Raises ``FileNotFoundError`` with an actionable hint if *name* is not
    yet vendored.
    """
    name = validate_skill_name(name)
    path = _BUNDLED_ROOT / name
    if not (path / "SKILL.md").is_file():
        available = ", ".join(bundled_skill_names()) or "(none)"
        raise FileNotFoundError(
            f"No bundled Anthropic skill named {name!r}. "
            f"Available: {available}.  Install it with "
            f"`flowforge skills sync {name}`."
        )
    return path


def bundled_skill_names() -> list[str]:
    """Return the names of all vendored Anthropic Agent Skills."""
    if not _BUNDLED_ROOT.is_dir():
        return []
    return sorted(
        child.name
        for child in _BUNDLED_ROOT.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )


__all__ = [
    "DEFAULT_BUNDLED_SKILLS",
    "bundled_skill_names",
    "bundled_skill_path",
    "native_skill_names",
    "native_skill_path",
    "skill_path",
    "sync_default_skills",
    "sync_skill",
    "validate_skill_name",
]
