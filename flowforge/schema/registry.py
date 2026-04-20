"""SchemaRegistry singleton — single source of truth for all metadata."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.annotations.metadata import GlobalMeta


class SchemaRegistry:
    """Singleton registry holding the compiled DAG and metadata."""

    _instance: SchemaRegistry | None = None

    def __new__(cls) -> SchemaRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dag = None
            cls._instance._global_meta = None
        return cls._instance

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def dag(self) -> FlowForgeDAG | None:
        return self._dag

    @dag.setter
    def dag(self, value: FlowForgeDAG) -> None:
        self._dag = value

    @property
    def global_meta(self) -> GlobalMeta | None:
        return self._global_meta

    @global_meta.setter
    def global_meta(self, value: GlobalMeta) -> None:
        self._global_meta = value

    def reset(self) -> None:
        """Reset the registry (useful for testing)."""
        self._dag = None
        self._global_meta = None

    @classmethod
    def get_instance(cls) -> SchemaRegistry:
        return cls()
