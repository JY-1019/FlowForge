"""Ledger runtime orchestration.

:class:`LedgerConfig` is the immutable settings object attached to every
flow/task/step meta by the ``@ledger`` decorator.  :class:`Ledger` is the
per-run runtime that owns the SQLite store + failure detector and is invoked
by the runners through :meth:`guard_step` (contract gate + trajectory +
decision + failure recording).

The whole feature is inert unless a node's ``meta.ledger`` is set, so non-
ledger agents pay nothing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from flowforge.errors import FlowForgeError
from flowforge.ledger.contract import check_contracts
from flowforge.ledger.detector import FailureDetector
from flowforge.ledger.store import LedgerStore

logger = logging.getLogger(__name__)


class ContractViolation(FlowForgeError):
    """Raised before a node runs when it would breach a ledger contract."""


@dataclass
class LedgerConfig:
    """Settings shared by every node under one ``@ledger`` flow."""

    contracts: dict[str, str] = field(default_factory=dict)
    db_path: str = "./ledger.db"
    failure_threshold: int = 3


def apply_ledger(flow_meta: Any, config: LedgerConfig) -> None:
    """Recursively stamp *config* onto a FlowMeta and all its descendants."""
    flow_meta.ledger = config
    for tm in getattr(flow_meta, "tasks", []) or []:
        _apply_task(tm, config)
    for cf in getattr(flow_meta, "child_flows", []) or []:
        apply_ledger(cf, config)
    for bm in (getattr(flow_meta, "branches", {}) or {}).values():
        apply_ledger(bm, config)
    if getattr(flow_meta, "fallback", None) is not None:
        apply_ledger(flow_meta.fallback, config)


def _apply_task(task_meta: Any, config: LedgerConfig) -> None:
    task_meta.ledger = config
    for sm in getattr(task_meta, "steps", []) or []:
        sm.ledger = config
    for ct in getattr(task_meta, "child_tasks", []) or []:
        _apply_task(ct, config)
    for bm in (getattr(task_meta, "branches", {}) or {}).values():
        _apply_task(bm, config)
    if getattr(task_meta, "fallback", None) is not None:
        _apply_task(task_meta.fallback, config)


class Ledger:
    """Per-run ledger runtime: store + detector + contract gate."""

    def __init__(self, config: LedgerConfig, llm_config: Any = None) -> None:
        self.config = config
        self.llm_config = llm_config
        self.store = LedgerStore(config.db_path)
        self.detector = FailureDetector(
            self.store, llm_config=llm_config, threshold=config.failure_threshold
        )
        self.run_id: str | None = None

    # ------------------------------------------------------------------
    # Run lifecycle (called by the engine)
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, input_repr: str) -> None:
        self.run_id = run_id
        self.store.start_run(run_id, input_repr)

    def finish_run(self, output_repr: str, error: str | None) -> None:
        if self.run_id is not None:
            self.store.finish_run(self.run_id, output_repr, error)

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------------
    # Node guard (called by the runners)
    # ------------------------------------------------------------------

    async def guard_step(
        self,
        *,
        meta: Any,
        node_id: str,
        node_type: str,
        step_name: str,
        input_data: Any,
        tracer: Any,
        thunk: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Contract-gate, run, then record trajectory/decision/failure."""
        await self._check_contracts(meta, node_id, node_type, input_data)

        try:
            output = await thunk()
        except ContractViolation:
            raise
        except Exception as exc:
            self._sync_node(node_id, tracer)
            await self.detector.observe(
                run_id=self.run_id or "?",
                node_id=node_id,
                step_name=step_name,
                exc=exc,
                func=getattr(meta, "func", None),
            )
            raise

        self._sync_node(node_id, tracer)
        return output

    async def _check_contracts(
        self, meta: Any, node_id: str, node_type: str, input_data: Any
    ) -> None:
        node_contract = getattr(meta, "contract", None)
        # Flow-level contracts are evaluated once, at step granularity, to avoid
        # redundant LLM calls at the flow/task/step layers for the same input.
        contracts = self.config.contracts if node_type == "step" else {}
        if not node_contract and not contracts:
            return

        context_text = _safe_text(input_data)
        violations = await check_contracts(
            node_id=node_id,
            node_contract=node_contract,
            contracts=contracts,
            context_text=context_text,
            llm_config=self.llm_config,
        )
        if not violations:
            return

        for v in violations:
            self.store.record_violation(
                self.run_id or "?", node_id, v.contract, v.reason
            )
        detail = "; ".join(f"{v.contract} → {v.reason}" for v in violations)
        raise ContractViolation(
            f"[Ledger] '{node_id}' 실행 차단 — 계약 위반: {detail}"
        )

    def _sync_node(self, node_id: str, tracer: Any) -> None:
        """Copy the tracer's NodeTrace for *node_id* into the SQLite store."""
        if tracer is None:
            return
        nt = tracer.trace.get_node_trace(node_id)
        if nt is None:
            return
        status = "error" if nt.error else ("ok" if nt.finished_at else "running")
        self.store.record_node(
            run_id=self.run_id or "?",
            node_id=nt.node_id,
            node_type=nt.node_type,
            name=nt.name,
            execution_order=nt.execution_order,
            started_at=nt.started_at,
            finished_at=nt.finished_at,
            duration_ms=nt.duration_ms,
            input_repr=nt.input_repr,
            output_repr=nt.output_repr,
            status=status,
            error=nt.error,
        )
        if nt.selected_branch:
            self.store.record_decision(
                self.run_id or "?",
                node_id,
                kind="branch",
                selected=nt.selected_branch,
            )


def _safe_text(value: Any, limit: int = 800) -> str:
    try:
        if hasattr(value, "model_dump"):
            text = str(value.model_dump())
        else:
            text = repr(value)
    except Exception:  # noqa: BLE001
        text = "<unrepresentable>"
    return text[:limit]
