"""Natural-language contract checking.

Contracts are plain Korean/English sentences — either a per-node *must-not*
rule (``@step(contract="승인 없이 삭제 금지")``) or a flow-level
``{situation: rule}`` map passed to ``@ledger(contracts=...)``.  Before a
guarded node runs, every applicable contract is matched against the current
execution context by a single LLM call; any violation blocks execution.

The check **fails open**: when no LLM is reachable (e.g. offline, no API key)
it logs a warning and allows execution rather than blocking the whole agent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """A single contract the current context would breach."""

    contract: str
    reason: str


class _Verdict(BaseModel):
    violations: list["_VerdictItem"]


class _VerdictItem(BaseModel):
    contract: str
    reason: str


_Verdict.model_rebuild()


_SYSTEM = (
    "You are a strict compliance gate for an AI agent runtime. "
    "You are given the execution context for a node that is about to run and "
    "a list of contracts (rules). For EACH contract, decide whether proceeding "
    "with this node under the given context would VIOLATE the rule. "
    "A rule is violated when the situation it describes applies AND the required "
    "guard (approval, policy reference, prohibition, …) is not satisfied by the "
    "context. Only report contracts that are actually violated. When in doubt and "
    "the situation clearly does not apply, do NOT report it. "
    "Return your answer via the structured_output tool."
)


def _build_contract_list(
    node_contract: str | None, contracts: dict[str, str] | None
) -> list[str]:
    items: list[str] = []
    if node_contract:
        items.append(f"[MUST NOT] {node_contract}")
    for situation, rule in (contracts or {}).items():
        items.append(f"[IF] {situation} [THEN] {rule}")
    return items


async def check_contracts(
    *,
    node_id: str,
    node_contract: str | None,
    contracts: dict[str, str] | None,
    context_text: str,
    llm_config: Any,
) -> list[Violation]:
    """Return the list of contracts the current context violates (possibly empty)."""
    rules = _build_contract_list(node_contract, contracts)
    if not rules:
        return []

    from flowforge.execution.llm import call_llm_api

    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
    user_prompt = (
        f"Node about to execute: {node_id}\n\n"
        f"Execution context:\n{context_text}\n\n"
        f"Contracts to evaluate:\n{numbered}\n\n"
        "List every contract that would be violated. Copy the contract text "
        "verbatim into the `contract` field and give a one-sentence `reason`."
    )

    try:
        result = await call_llm_api(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            llm_config=llm_config,
            output_schema=_Verdict,
        )
    except Exception as exc:  # noqa: BLE001 - fail open, never block on infra error
        logger.warning(
            "[Ledger] contract check skipped for %s (LLM unavailable: %s)",
            node_id, exc,
        )
        return []

    raw = result.get("violations", []) if isinstance(result, dict) else []
    violations: list[Violation] = []
    for item in raw:
        if isinstance(item, dict) and item.get("contract"):
            violations.append(
                Violation(
                    contract=str(item["contract"]),
                    reason=str(item.get("reason", "")),
                )
            )
    return violations
