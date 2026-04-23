"""Tests for session memory and step history."""
import pytest
from flowforge import global_config, flow, task, step, FlowForge


# ── Agent definition (module-level) ────────────────────────────────────

@task(name="greet", prompt="greet the user")
class GreetTask:
    @step(order=1, prompt="say hello")
    async def say_hello(ctx):
        return {"message": "hello", "trail": ["greet.say_hello"]}


@flow(name="hello_flow", prompt="a simple hello flow")
class HelloFlow:
    GreetTask = GreetTask


@global_config(prompt="test agent for memory")
class MemoryAgent:
    HelloFlow = HelloFlow


# ── Unit tests for SessionMemory ────────────────────────────────────────

class TestSessionMemory:
    def test_empty_memory(self):
        from flowforge.execution.memory import SessionMemory
        mem = SessionMemory()
        assert len(mem) == 0
        assert not mem
        assert mem.to_prompt_block() == ""

    def test_record_and_retrieve(self):
        from flowforge.execution.memory import SessionMemory
        mem = SessionMemory()
        mem.record_run("input1", {"result": "ok"}, route="hello_flow")
        assert len(mem) == 1
        assert mem
        block = mem.to_prompt_block()
        assert "Previous Runs" in block
        assert "input1" in block
        assert "hello_flow" in block

    def test_max_entries_eviction(self):
        from flowforge.execution.memory import SessionMemory
        mem = SessionMemory(max_entries=3)
        for i in range(5):
            mem.record_run(f"input_{i}", f"output_{i}")
        assert len(mem) == 3
        # Oldest entries (0, 1) should be evicted
        block = mem.to_prompt_block()
        assert "input_0" not in block
        assert "input_1" not in block
        assert "input_4" in block

    def test_clear(self):
        from flowforge.execution.memory import SessionMemory
        mem = SessionMemory()
        mem.record_run("a", "b")
        mem.record_run("c", "d")
        assert len(mem) == 2
        mem.clear()
        assert len(mem) == 0
        assert mem.to_prompt_block() == ""

    def test_summarize_truncation(self):
        from flowforge.execution.memory import SessionMemory
        mem = SessionMemory(max_entry_chars=20)
        long_input = "x" * 100
        mem.record_run(long_input, "short")
        entry = mem.entries[0]
        assert len(entry.input_summary) <= 20

    def test_pydantic_model_summary(self):
        from flowforge.execution.memory import SessionMemory
        from pydantic import BaseModel

        class Query(BaseModel):
            text: str

        mem = SessionMemory()
        mem.record_run(Query(text="hello"), {"done": True})
        block = mem.to_prompt_block()
        assert "hello" in block


# ── Unit tests for StepHistory ────────────────────────────────────────

class TestStepHistory:
    def test_empty_history(self):
        from flowforge.execution.memory import StepHistory
        hist = StepHistory()
        assert len(hist) == 0
        assert not hist
        assert hist.to_prompt_block() == ""

    def test_record_and_retrieve(self):
        from flowforge.execution.memory import StepHistory
        hist = StepHistory()
        hist.record_step("validate", 1, "check input", "input is valid")
        assert len(hist) == 1
        block = hist.to_prompt_block()
        assert "Previous Steps" in block
        assert "validate" in block
        assert "input is valid" in block

    def test_response_truncation(self):
        from flowforge.execution.memory import StepHistory
        hist = StepHistory(max_step_chars=30)
        hist.record_step("step1", 1, "prompt", "x" * 100)
        entry = hist.entries[0]
        assert len(entry.response_summary) <= 30

    def test_clear(self):
        from flowforge.execution.memory import StepHistory
        hist = StepHistory()
        hist.record_step("a", 1, "p", "r")
        hist.record_step("b", 2, "p", "r")
        hist.clear()
        assert len(hist) == 0


# ── Integration: engine.memory persists across runs ─────────────────

@pytest.mark.asyncio
async def test_engine_memory_persists_across_runs():
    engine = FlowForge.compile(MemoryAgent)

    # Memory should be empty initially.
    assert len(engine.memory) == 0

    # First run.
    await engine.run(None)
    assert len(engine.memory) == 1

    # Second run.
    await engine.run({"query": "hi"})
    assert len(engine.memory) == 2

    # Block should contain both runs.
    block = engine.memory.to_prompt_block()
    assert "Run 1" in block
    assert "Run 2" in block


@pytest.mark.asyncio
async def test_engine_memory_clear():
    engine = FlowForge.compile(MemoryAgent)
    await engine.run(None)
    await engine.run(None)
    assert len(engine.memory) == 2

    engine.memory.clear()
    assert len(engine.memory) == 0
    assert engine.memory.to_prompt_block() == ""


@pytest.mark.asyncio
async def test_step_history_created_per_task():
    """StepHistory is created fresh for each task invocation."""
    engine = FlowForge.compile(MemoryAgent)
    await engine.run(None)
    # No assertion on step_history content (would need LLM call),
    # but verify the engine doesn't crash and memory is recorded.
    assert len(engine.memory) == 1
