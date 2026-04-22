"""Tests for @step(pass_criteria=...) — LLM judge + retry with feedback.

Covers:
- pass_criteria triggers LLM judge evaluation
- Failed criteria causes step re-run with feedback
- Max retries exhausted → last result accepted
- Passed on first attempt → no retries
- Feedback history is accessible via ctx.pass_criteria_feedback
- Judge failure (exception) → result accepted gracefully
"""
import pytest
from unittest.mock import AsyncMock, patch

from flowforge import global_config, flow, task, step, FlowForge
from flowforge.execution.runner import _parse_judge_response


# ---------------------------------------------------------------------------
# _parse_judge_response tests
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_dict_pass(self):
        assert _parse_judge_response({"pass": True, "feedback": ""})["pass"] is True

    def test_dict_fail(self):
        r = _parse_judge_response({"pass": False, "feedback": "too short"})
        assert r["pass"] is False
        assert r["feedback"] == "too short"

    def test_json_string_pass(self):
        r = _parse_judge_response('{"pass": true, "feedback": ""}')
        assert r["pass"] is True

    def test_json_string_fail(self):
        r = _parse_judge_response('{"pass": false, "feedback": "missing field"}')
        assert r["pass"] is False

    def test_markdown_fenced_json(self):
        text = '```json\n{"pass": true, "feedback": "ok"}\n```'
        assert _parse_judge_response(text)["pass"] is True

    def test_fallback_no_json(self):
        r = _parse_judge_response("this output is not valid")
        assert r["pass"] is False


# ---------------------------------------------------------------------------
# Integration tests — pass_criteria with mocked LLM judge
# ---------------------------------------------------------------------------

# Counter tracking how many times the step runs.
_criteria_call_count = 0


def _reset():
    global _criteria_call_count
    _criteria_call_count = 0


# Agent with pass_criteria: step returns "bad" first 2 times, "good" on 3rd.
@global_config(prompt="criteria test agent")
class CriteriaAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(
                order=1,
                prompt="produce a valid result",
                pass_criteria="결과에 반드시 'valid' 필드가 True여야 한다",
                pass_criteria_max_retries=3,
            )
            async def produce(ctx):
                global _criteria_call_count
                _criteria_call_count += 1

                # On retries, ctx.pass_criteria_feedback contains previous feedback
                feedback = ctx.pass_criteria_feedback

                if _criteria_call_count >= 3:
                    return {"valid": True, "value": "success", "attempt": _criteria_call_count,
                            "used_feedback": len(feedback) > 0}
                return {"valid": False, "value": "not_yet", "attempt": _criteria_call_count}


class TestPassCriteriaIntegration:
    @pytest.mark.asyncio
    async def test_retry_until_criteria_met(self):
        """Step is re-run when pass_criteria judge says fail."""
        _reset()

        # Mock the LLM judge: fail first 2 times, pass on 3rd
        judge_responses = [
            '{"pass": false, "feedback": "valid 필드가 False입니다"}',
            '{"pass": false, "feedback": "여전히 valid=False, 수정 필요"}',
            '{"pass": true, "feedback": ""}',
        ]
        call_idx = {"i": 0}

        async def mock_judge(**kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return judge_responses[idx]

        with patch("flowforge.execution.llm.call_llm_api", side_effect=mock_judge):
            engine = FlowForge.compile(CriteriaAgent)
            result = await engine.run(None)

        assert result["valid"] is True
        assert result["attempt"] == 3
        assert result["used_feedback"] is True  # had feedback from previous fails
        assert _criteria_call_count == 3

    @pytest.mark.asyncio
    async def test_pass_on_first_attempt(self):
        """No retries when judge says pass on first evaluation."""
        _reset()

        async def mock_judge(**kwargs):
            return '{"pass": true, "feedback": ""}'

        with patch("flowforge.execution.llm.call_llm_api", side_effect=mock_judge):
            engine = FlowForge.compile(CriteriaAgent)
            result = await engine.run(None)

        assert _criteria_call_count == 1  # only ran once
        assert result["attempt"] == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """After max retries, last result is used regardless."""
        _reset()

        async def mock_judge(**kwargs):
            return '{"pass": false, "feedback": "always failing"}'

        with patch("flowforge.execution.llm.call_llm_api", side_effect=mock_judge):
            engine = FlowForge.compile(CriteriaAgent)
            result = await engine.run(None)

        # max_retries=3 → initial + 3 retries = 4 judge calls, 4 step calls
        # (initial run + 3 retries)
        assert _criteria_call_count == 4
        # Last result is returned even though criteria failed
        assert result["attempt"] == 4

    @pytest.mark.asyncio
    async def test_judge_exception_accepts_result(self):
        """If judge LLM call fails, the result is accepted gracefully."""
        _reset()

        async def mock_judge(**kwargs):
            raise RuntimeError("LLM unavailable")

        with patch("flowforge.execution.llm.call_llm_api", side_effect=mock_judge):
            engine = FlowForge.compile(CriteriaAgent)
            result = await engine.run(None)

        assert _criteria_call_count == 1  # only ran once, judge failed → accepted
        assert result["attempt"] == 1


# Agent without pass_criteria — should work normally.
@global_config(prompt="no criteria agent")
class NoCriteriaAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(order=1, prompt="just run")
            async def run_it(ctx):
                return {"done": True}


class TestNoCriteria:
    @pytest.mark.asyncio
    async def test_no_criteria_runs_once(self):
        """Without pass_criteria, step runs normally without judge."""
        engine = FlowForge.compile(NoCriteriaAgent)
        result = await engine.run(None)
        assert result["done"] is True


# Agent with pass_criteria and output_schema — schema validated after criteria pass.
_schema_call_count = 0


@global_config(prompt="schema criteria agent")
class SchemaCriteriaAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(
                order=1,
                prompt="produce result with schema",
                pass_criteria="score must be >= 80",
                pass_criteria_max_retries=2,
            )
            async def produce(ctx):
                global _schema_call_count
                _schema_call_count += 1
                if _schema_call_count >= 2:
                    return {"score": 90, "label": "pass"}
                return {"score": 50, "label": "fail"}


class TestPassCriteriaWithTrace:
    @pytest.mark.asyncio
    async def test_criteria_with_trace(self):
        """pass_criteria retries are reflected in the trace."""
        global _schema_call_count
        _schema_call_count = 0

        judge_responses = [
            '{"pass": false, "feedback": "score 50 < 80"}',
            '{"pass": true, "feedback": ""}',
        ]
        call_idx = {"i": 0}

        async def mock_judge(**kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return judge_responses[idx]

        with patch("flowforge.execution.llm.call_llm_api", side_effect=mock_judge):
            engine = FlowForge.compile(SchemaCriteriaAgent)
            result, trace = await engine.run_traced(None)

        assert result["score"] == 90
        assert trace.succeeded


# Test feedback accumulation across retries.
_feedback_log = []


@global_config(prompt="feedback agent")
class FeedbackAgent:
    @flow(name="f", prompt="f")
    class F:
        @task(name="t", prompt="t")
        class T:
            @step(
                order=1,
                prompt="produce and track feedback",
                pass_criteria="must contain greeting",
                pass_criteria_max_retries=3,
            )
            async def produce(ctx):
                global _feedback_log
                _feedback_log.append(list(ctx.pass_criteria_feedback))
                return {"text": f"attempt {len(ctx.pass_criteria_feedback) + 1}"}


class TestFeedbackAccumulation:
    @pytest.mark.asyncio
    async def test_feedback_grows_each_retry(self):
        """Each retry receives accumulated feedback from all previous failures."""
        global _feedback_log
        _feedback_log = []

        judge_responses = [
            '{"pass": false, "feedback": "no greeting found"}',
            '{"pass": false, "feedback": "still no greeting"}',
            '{"pass": true, "feedback": ""}',
        ]
        call_idx = {"i": 0}

        async def mock_judge(**kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return judge_responses[idx]

        with patch("flowforge.execution.llm.call_llm_api", side_effect=mock_judge):
            engine = FlowForge.compile(FeedbackAgent)
            await engine.run(None)

        # 1st call: no feedback yet
        assert _feedback_log[0] == []
        # 2nd call (1st retry): one feedback
        assert _feedback_log[1] == ["no greeting found"]
        # 3rd call (2nd retry): two feedbacks accumulated
        assert _feedback_log[2] == ["no greeting found", "still no greeting"]
