"""Download Anthropic Agent Skills from ``github.com/anthropics/skills``.

The upstream repository publishes skills as ``SKILL.md`` folders, sometimes
with supporting ``scripts/``/``references/``/``assets/`` subdirectories.
FlowForge ships a curated default bundle and exposes ``sync_skill`` so users
can fetch any other upstream skill on demand.

Usage
-----
    from flowforge.skills.sync import sync_skill, sync_default_skills

    sync_skill("pdf")            # vendors flowforge/skills/anthropic/pdf/
    sync_default_skills()        # refresh the curated bundle in-place
"""
from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "anthropics/skills"
UPSTREAM_BRANCH = "main"
UPSTREAM_PATH = "skills"

# Curated subset of upstream Anthropic skills bundled with FlowForge by
# default.  Skills with heavy supporting ``scripts/``/``references/`` folders
# (pdf, pptx, xlsx, docx, claude-api, skill-creator, ...) are intentionally
# omitted — they are installed on demand via ``flowforge skills sync <name>``.
DEFAULT_BUNDLED_SKILLS: tuple[str, ...] = (
    "frontend-design",
    "brand-guidelines",
    "internal-comms",
    "theme-factory",
    "web-artifacts-builder",
    "webapp-testing",
    "mcp-builder",
)

# Default-bundle skills are intentionally chosen to be ``SKILL.md`` + license
# only — no ``scripts/``/``references/`` subdirectories.  Hard-coding the file
# list lets the default sync hit the raw CDN directly and bypass the
# unauthenticated GitHub contents-API rate limit (60 req/hour per IP).
_DEFAULT_BUNDLE_FILES: dict[str, tuple[str, ...]] = {
    name: ("SKILL.md", "LICENSE.txt") for name in DEFAULT_BUNDLED_SKILLS
}
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_skill_name(name: str) -> str:
    """Validate and normalize an upstream Anthropic skill folder name."""
    cleaned = str(name).strip()
    if not _SKILL_NAME_RE.fullmatch(cleaned):
        raise ValueError(
            "Skill names must be simple upstream folder names containing only "
            f"letters, numbers, underscores, or hyphens: {name!r}"
        )
    return cleaned


def _ssl_context(verify_ssl: bool) -> ssl.SSLContext | None:
    if verify_ssl:
        return None
    return ssl._create_unverified_context()


def _resolve_verify_ssl(verify_ssl: bool | None) -> bool:
    if verify_ssl is not None:
        return verify_ssl
    flag = os.getenv("FLOWFORGE_VERIFY_SSL", "1").strip().lower()
    return flag not in {"0", "false", "no"}


def _http_get_json(url: str, *, verify_ssl: bool) -> object:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "flowforge-skills-sync",
        },
    )
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, context=_ssl_context(verify_ssl), timeout=30) as r:
        return json.loads(r.read())


def _http_get_bytes(url: str, *, verify_ssl: bool) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "flowforge-skills-sync"}
    )
    with urllib.request.urlopen(req, context=_ssl_context(verify_ssl), timeout=60) as r:
        return r.read()


def _list_skill_files(name: str, *, verify_ssl: bool) -> list[str]:
    """Return every blob path under ``skills/<name>/`` in the upstream repo."""
    name = validate_skill_name(name)
    api_root = (
        f"https://api.github.com/repos/{UPSTREAM_REPO}/contents/"
        f"{UPSTREAM_PATH}/{name}?ref={UPSTREAM_BRANCH}"
    )
    try:
        entries = _http_get_json(api_root, verify_ssl=verify_ssl)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 429}:
            raise RuntimeError(
                "GitHub API rate limit reached while listing upstream "
                f"skill {name!r}. Set GITHUB_TOKEN or GH_TOKEN and retry."
            ) from exc
        raise
    if isinstance(entries, dict) and entries.get("message"):
        raise FileNotFoundError(
            f"Upstream skill {name!r} not found at {api_root}: "
            f"{entries.get('message')}"
        )
    if not isinstance(entries, list):
        raise FileNotFoundError(f"Unexpected response for {name!r}: {entries!r}")

    paths: list[str] = []
    stack: list[dict] = list(entries)
    while stack:
        entry = stack.pop()
        etype = entry.get("type")
        if etype == "file":
            paths.append(entry["path"])
        elif etype == "dir":
            url = (
                f"https://api.github.com/repos/{UPSTREAM_REPO}/contents/"
                f"{entry['path']}?ref={UPSTREAM_BRANCH}"
            )
            try:
                sub = _http_get_json(url, verify_ssl=verify_ssl)
            except urllib.error.HTTPError as exc:
                if exc.code in {403, 429}:
                    raise RuntimeError(
                        "GitHub API rate limit reached while listing upstream "
                        f"skill {name!r}. Set GITHUB_TOKEN or GH_TOKEN and retry."
                    ) from exc
                raise
            if isinstance(sub, list):
                stack.extend(sub)
    return paths


def _bundled_root() -> Path:
    return Path(__file__).resolve().parent / "anthropic"


def sync_skill(
    name: str,
    *,
    dest: Path | None = None,
    force: bool = False,
    verify_ssl: bool | None = None,
) -> Path:
    """Download one upstream Anthropic skill folder and vendor it locally.

    Parameters
    ----------
    name:
        Folder name under ``skills/`` in ``anthropics/skills`` (e.g. ``"pdf"``).
    dest:
        Destination directory.  Defaults to ``flowforge/skills/anthropic/<name>``.
    force:
        Re-download even when ``SKILL.md`` is already present locally.
    verify_ssl:
        Override TLS verification.  Defaults to the ``FLOWFORGE_VERIFY_SSL``
        env var (``1`` by default).  Set ``False`` for the LG CNS proxy.

    Returns
    -------
    Path
        The local skill directory.
    """
    name = validate_skill_name(name)
    verify = _resolve_verify_ssl(verify_ssl)
    target = dest if dest is not None else _bundled_root() / name

    skill_md = target / "SKILL.md"
    if skill_md.is_file() and not force:
        logger.info("skill %r already vendored at %s", name, target)
        return target

    base = f"{UPSTREAM_PATH}/{name}/"
    known = _DEFAULT_BUNDLE_FILES.get(name)
    if known is not None:
        # Default-bundle skill: skip the contents API entirely.  Pull each
        # known file straight from raw.githubusercontent.com and tolerate
        # missing optional files (LICENSE.txt may legitimately be absent).
        paths = [base + rel for rel in known]
    else:
        paths = _list_skill_files(name, verify_ssl=verify)
        if not paths:
            raise FileNotFoundError(f"No files found for upstream skill {name!r}")

    target.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in paths:
        if not path.startswith(base):
            continue
        rel = path[len(base):]
        local = target / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
            f"{UPSTREAM_BRANCH}/{path}"
        )
        try:
            payload = _http_get_bytes(url, verify_ssl=verify)
        except urllib.error.HTTPError as exc:
            if known is not None and exc.code == 404:
                # Optional file missing for a default-bundle skill — skip.
                logger.debug("optional file missing: %s", url)
                continue
            raise
        local.write_bytes(payload)
        written += 1
        logger.info("  wrote %s", local)

    if written == 0:
        raise FileNotFoundError(
            f"Upstream listing for {name!r} returned no files inside {base}"
        )
    return target


def sync_default_skills(
    *, force: bool = False, verify_ssl: bool | None = None
) -> list[Path]:
    """Vendor every skill in :data:`DEFAULT_BUNDLED_SKILLS`."""
    return [
        sync_skill(name, force=force, verify_ssl=verify_ssl)
        for name in DEFAULT_BUNDLED_SKILLS
    ]


__all__ = [
    "DEFAULT_BUNDLED_SKILLS",
    "UPSTREAM_REPO",
    "UPSTREAM_BRANCH",
    "validate_skill_name",
    "sync_skill",
    "sync_default_skills",
]
