"""Harness Ledger — one decorator, everything automatic.

Put ``@ledger`` above ``@flow`` and the flow's whole subtree gains, with zero
further wiring:

* **Trajectory Store** — every step execution persisted to SQLite (``./ledger.db``).
* **Decision Log** — branch selections recorded.
* **Contract Check** — natural-language rules verified before a node runs;
  violations block execution.
* **Failure Detector** — repeated failures surface a console warning + a stored
  repair suggestion (apply via the ``ledger`` CLI).

Usage::

    @ledger
    @flow(name="my_flow", prompt="…")
    class MyFlow:
        ...

    @ledger(contracts={
        "결제 금액 100만원 초과": "human_approval 요청 필수",
        "환불 + 30일 초과": "정책 문서 참조 후 응답",
    })
    @flow(name="payment", prompt="…")
    class PaymentFlow:
        ...
"""
from __future__ import annotations

from typing import Any, Callable, overload

from flowforge.annotations.decorators import _FLOW_ATTR
from flowforge.errors import CompileError
from flowforge.ledger.core import (
    ContractViolation,
    Ledger,
    LedgerConfig,
    apply_ledger,
)

__all__ = ["ledger", "Ledger", "LedgerConfig", "ContractViolation"]


@overload
def ledger(flow_cls: type) -> type: ...
@overload
def ledger(
    *,
    contracts: dict[str, str] | None = ...,
    db_path: str = ...,
    failure_threshold: int = ...,
) -> Callable[[type], type]: ...


def ledger(
    flow_cls: type | None = None,
    *,
    contracts: dict[str, str] | None = None,
    db_path: str = "./ledger.db",
    failure_threshold: int = 3,
) -> Any:
    """Activate the Harness Ledger for the decorated ``@flow`` and its subtree.

    Works both bare (``@ledger``) and called (``@ledger(contracts={...})``).
    Must sit **above** ``@flow``; raises :class:`CompileError` otherwise.
    """
    config = LedgerConfig(
        contracts=dict(contracts or {}),
        db_path=db_path,
        failure_threshold=failure_threshold,
    )

    def wrap(target_cls: type) -> type:
        flow_meta = getattr(target_cls, _FLOW_ATTR, None)
        if flow_meta is None:
            raise CompileError(
                "@ledger must be applied directly above @flow "
                f"(got {target_cls!r} without flow metadata)."
            )
        apply_ledger(flow_meta, config)
        return target_cls

    if flow_cls is not None:
        return wrap(flow_cls)
    return wrap
