"""SHA-based prompt cache for generated docs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_DEFAULT_CACHE_DIR = Path(".flowforge_cache")


class DocCache:
    """File-based cache keyed by SHA-256 of the prompt string."""

    def __init__(self, cache_dir: Path | str = _DEFAULT_CACHE_DIR) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, prompt: str) -> dict[str, Any] | None:
        p = self._path(self._key(prompt))
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def set(self, prompt: str, doc: dict[str, Any]) -> None:
        p = self._path(self._key(prompt))
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2))

    def invalidate(self, prompt: str) -> None:
        p = self._path(self._key(prompt))
        if p.exists():
            p.unlink()

    def clear(self) -> None:
        for f in self._dir.glob("*.json"):
            f.unlink()
