"""Tests for Anthropic Agent Skill bundling and upstream sync helpers."""
from __future__ import annotations

import urllib.error

import pytest


def test_validate_skill_name_rejects_path_traversal():
    from flowforge.skills.sync import validate_skill_name

    for bad in ["../pdf", "pdf/../../x", "pdf\\x", "", ".hidden"]:
        with pytest.raises(ValueError):
            validate_skill_name(bad)


def test_bundled_skill_path_rejects_path_traversal():
    from flowforge.skills import bundled_skill_path

    with pytest.raises(ValueError):
        bundled_skill_path("../frontend-design")


def test_sync_default_skill_uses_raw_files_and_tolerates_missing_license(
    tmp_path,
    monkeypatch,
):
    import flowforge.skills.sync as sync_mod

    requested_urls: list[str] = []

    def fake_get_bytes(url: str, *, verify_ssl: bool) -> bytes:
        requested_urls.append(url)
        if url.endswith("/LICENSE.txt"):
            raise urllib.error.HTTPError(url, 404, "not found", None, None)
        assert url.endswith("/skills/frontend-design/SKILL.md")
        return b"---\nname: frontend-design\ndescription: Test skill.\n---\n\nBody\n"

    monkeypatch.setattr(sync_mod, "_http_get_bytes", fake_get_bytes)
    monkeypatch.setattr(
        sync_mod,
        "_http_get_json",
        lambda *args, **kwargs: pytest.fail("default skills should skip GitHub API"),
    )

    target = sync_mod.sync_skill(
        "frontend-design",
        dest=tmp_path / "frontend-design",
        force=True,
        verify_ssl=True,
    )

    assert (target / "SKILL.md").read_text(encoding="utf-8").startswith("---")
    assert any(url.endswith("/SKILL.md") for url in requested_urls)
    assert any(url.endswith("/LICENSE.txt") for url in requested_urls)


def test_sync_non_default_skill_fetches_recursive_github_listing(
    tmp_path,
    monkeypatch,
):
    import flowforge.skills.sync as sync_mod

    def fake_get_json(url: str, *, verify_ssl: bool):
        if url.endswith("/skills/custom-skill?ref=main"):
            return [
                {
                    "type": "file",
                    "path": "skills/custom-skill/SKILL.md",
                },
                {
                    "type": "dir",
                    "path": "skills/custom-skill/scripts",
                },
            ]
        if url.endswith("/skills/custom-skill/scripts?ref=main"):
            return [
                {
                    "type": "file",
                    "path": "skills/custom-skill/scripts/run.py",
                }
            ]
        return pytest.fail(f"unexpected GitHub API URL: {url}")

    def fake_get_bytes(url: str, *, verify_ssl: bool) -> bytes:
        if url.endswith("/SKILL.md"):
            return b"---\nname: custom-skill\ndescription: Custom.\n---\n\nBody\n"
        if url.endswith("/scripts/run.py"):
            return b"print('ok')\n"
        return pytest.fail(f"unexpected raw URL: {url}")

    monkeypatch.setattr(sync_mod, "_http_get_json", fake_get_json)
    monkeypatch.setattr(sync_mod, "_http_get_bytes", fake_get_bytes)

    target = sync_mod.sync_skill(
        "custom-skill",
        dest=tmp_path / "custom-skill",
        force=True,
        verify_ssl=True,
    )

    assert (target / "SKILL.md").is_file()
    assert (target / "scripts" / "run.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_sync_non_default_skill_rate_limit_error_mentions_token(
    tmp_path,
    monkeypatch,
):
    import flowforge.skills.sync as sync_mod

    def fake_get_json(url: str, *, verify_ssl: bool):
        raise urllib.error.HTTPError(url, 403, "rate limit exceeded", None, None)

    monkeypatch.setattr(sync_mod, "_http_get_json", fake_get_json)

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        sync_mod.sync_skill(
            "custom-skill",
            dest=tmp_path / "custom-skill",
            force=True,
            verify_ssl=True,
        )
