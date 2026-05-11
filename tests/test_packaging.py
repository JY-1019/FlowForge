from __future__ import annotations

import importlib.metadata
from pathlib import Path

import flowforge
from flowforge.skills import DEFAULT_BUNDLED_SKILLS, bundled_skill_path


def test_import_metadata_version_matches_public_version() -> None:
    assert importlib.metadata.version("flowforge") == flowforge.__version__


def test_runtime_package_data_is_available() -> None:
    package_root = Path(flowforge.__file__).resolve().parent

    assert (package_root / "py.typed").is_file()
    assert (package_root / "mkdocs.yml").is_file()
    assert (package_root / "_docs" / "index.md").is_file()


def test_default_bundled_skills_are_available() -> None:
    for name in DEFAULT_BUNDLED_SKILLS:
        skill_dir = bundled_skill_path(name)
        assert (skill_dir / "SKILL.md").is_file()
        assert (skill_dir / "LICENSE.txt").is_file()

