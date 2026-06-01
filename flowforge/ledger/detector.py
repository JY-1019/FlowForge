"""Failure-pattern detection and repair-suggestion generation.

Every step failure is recorded with a *signature* (step name + exception
type).  When the same signature recurs ``threshold`` times (default 3) the
detector emits a console warning and stores a repair suggestion.  When an LLM
is reachable and the failing function's source can be located, the suggestion
also carries a concrete replacement for that function's source so
``ledger apply`` can patch the file (and ``ledger rollback`` can undo it).
"""
from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 3


class _Suggestion(BaseModel):
    cause: str
    suggestion: str
    new_source: str | None = None


def _signature(step_name: str, exc: BaseException) -> str:
    """Stable key grouping 'the same' failure across runs."""
    msg = re.sub(r"\d+", "#", str(exc))[:80]
    return f"{step_name}:{type(exc).__name__}:{msg}"


def _func_source(func: Any) -> tuple[str | None, str | None]:
    """Return ``(source_file, source_text)`` for *func*, or ``(None, None)``."""
    try:
        src_file = inspect.getsourcefile(func)
        src_text = inspect.getsource(func)
        return src_file, src_text
    except (OSError, TypeError):
        return None, None


_SYSTEM = (
    "You are a senior engineer triaging a repeatedly-failing step in an AI "
    "agent workflow. Given the step's source code and the recurring error, "
    "identify the root cause in one sentence, give a concrete fix suggestion, "
    "and — if you can fix it by editing this function — return the COMPLETE "
    "corrected function source in `new_source` (same signature/decorator, valid "
    "Python, no surrounding prose). If you cannot safely fix it from this "
    "function alone, leave `new_source` null. Reply via structured_output."
)


class FailureDetector:
    """Accumulates failures and surfaces suggestions at the threshold."""

    def __init__(
        self, store: Any, llm_config: Any = None, threshold: int = _DEFAULT_THRESHOLD
    ) -> None:
        self._store = store
        self._llm_config = llm_config
        self._threshold = threshold
        # Signatures already surfaced this process — avoid duplicate warnings.
        self._warned: set[str] = set()

    async def observe(
        self,
        *,
        run_id: str,
        node_id: str,
        step_name: str,
        exc: BaseException,
        func: Any = None,
    ) -> None:
        sig = _signature(step_name, exc)
        count = self._store.record_failure(
            run_id, node_id, step_name, sig, str(exc)
        )
        if count < self._threshold or sig in self._warned:
            return
        self._warned.add(sig)

        cause, suggestion, new_source = await self._diagnose(step_name, exc, func)
        src_file, src_text = _func_source(func) if func is not None else (None, None)

        self._store.upsert_suggestion(
            signature=sig,
            step_name=step_name,
            cause=cause,
            suggestion=suggestion,
            source_file=src_file,
            func_name=getattr(func, "__name__", step_name),
            old_source=src_text,
            new_source=new_source,
        )
        self._print_warning(step_name, cause, suggestion)

    async def _diagnose(
        self, step_name: str, exc: BaseException, func: Any
    ) -> tuple[str, str, str | None]:
        """Return ``(cause, suggestion, new_source)``; LLM-backed when possible."""
        _, src_text = _func_source(func) if func is not None else (None, None)
        fallback_cause = f"{type(exc).__name__}: {exc}"
        fallback_suggestion = (
            f"'{step_name}' 가 동일한 {type(exc).__name__} 로 반복 실패합니다. "
            "입력 검증/예외 처리를 추가하거나 호출 대상을 점검하세요."
        )
        if self._llm_config is None or src_text is None:
            return fallback_cause, fallback_suggestion, None

        from flowforge.execution.llm import call_llm_api

        user_prompt = (
            f"Step: {step_name}\n"
            f"Recurring error: {type(exc).__name__}: {exc}\n\n"
            f"Current function source:\n```python\n{src_text}\n```"
        )
        try:
            result = await call_llm_api(
                system_prompt=_SYSTEM,
                user_prompt=user_prompt,
                llm_config=self._llm_config,
                output_schema=_Suggestion,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[Ledger] suggestion generation failed: %s", e)
            return fallback_cause, fallback_suggestion, None

        if not isinstance(result, dict):
            return fallback_cause, fallback_suggestion, None
        return (
            str(result.get("cause") or fallback_cause),
            str(result.get("suggestion") or fallback_suggestion),
            result.get("new_source") or None,
        )

    @staticmethod
    def _print_warning(step_name: str, cause: str, suggestion: str) -> None:
        # Spec-mandated console format.
        print(f"\n[Ledger] 실패 패턴 감지: {step_name} - {cause}")
        print(f"수정 제안: {suggestion}")
        print("적용하려면: ledger apply\n")
