"""Tests for AgentSession and create_session() isolation."""
import pytest
from flowforge import global_config, flow, task, step, FlowForge, AgentSession


# ── Agent definition (module-level) ────────────────────────────────────

@task(name="echo", prompt="echo the input")
class EchoTask:
    @step(order=1, prompt="return input as-is")
    async def do_echo(ctx):
        return {"echoed": True}


@flow(name="echo_flow", prompt="a simple echo flow")
class EchoFlow:
    EchoTask = EchoTask


@global_config(prompt="test agent for sessions")
class SessionAgent:
    EchoFlow = EchoFlow


# ── Tests ────────────────────────────────────────────────────────────────

class TestCreateSession:
    def test_create_session_returns_agent_session(self):
        agent = FlowForge.compile(SessionAgent)
        session = agent.create_session()
        assert isinstance(session, AgentSession)

    def test_sessions_have_independent_memory(self):
        agent = FlowForge.compile(SessionAgent)
        s1 = agent.create_session()
        s2 = agent.create_session()

        s1.memory.record_run("input_a", "output_a")
        assert len(s1.memory) == 1
        assert len(s2.memory) == 0  # s2 unaffected

    def test_sessions_share_dag(self):
        agent = FlowForge.compile(SessionAgent)
        s1 = agent.create_session()
        s2 = agent.create_session()
        assert s1._dag is s2._dag  # same object, zero-copy

    def test_sessions_share_docs(self):
        agent = FlowForge.compile(SessionAgent)
        s1 = agent.create_session()
        s2 = agent.create_session()
        assert s1._docs is s2._docs

    def test_session_independent_from_compiled_agent(self):
        agent = FlowForge.compile(SessionAgent)
        session = agent.create_session()

        agent.memory.record_run("agent_input", "agent_output")
        assert len(agent.memory) == 1
        assert len(session.memory) == 0  # session unaffected


@pytest.mark.asyncio
async def test_session_run_records_memory():
    agent = FlowForge.compile(SessionAgent)
    session = agent.create_session()

    await session.run(None)
    assert len(session.memory) == 1

    await session.run({"data": "hello"})
    assert len(session.memory) == 2


@pytest.mark.asyncio
async def test_session_run_traced():
    agent = FlowForge.compile(SessionAgent)
    session = agent.create_session()

    result, trace = await session.run_traced(None)
    assert trace is not None
    assert session.last_trace is trace
    assert trace.succeeded


@pytest.mark.asyncio
async def test_sessions_isolated_traces():
    agent = FlowForge.compile(SessionAgent)
    s1 = agent.create_session()
    s2 = agent.create_session()

    await s1.run(None)
    assert s1.last_trace is not None
    assert s2.last_trace is None  # s2 never ran


@pytest.mark.asyncio
async def test_session_memory_clear():
    agent = FlowForge.compile(SessionAgent)
    session = agent.create_session()

    await session.run(None)
    await session.run(None)
    assert len(session.memory) == 2

    session.memory.clear()
    assert len(session.memory) == 0
