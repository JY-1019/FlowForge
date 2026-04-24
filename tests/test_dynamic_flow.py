"""Tests for the dynamic flow generation feature.

Covers:
- @global_config(dynamic_flow=True) flag
- CompiledAgent.add_flow() — dynamic flow injection
- CompiledAgent.recompile() — full DAG rebuild
- DynamicFlowGenerator.compile_flow_code() — code → FlowMeta
- Meta-flow DAG injection when dynamic_flow=True
- add_flow_to_dag() — incremental compilation
"""
import pytest

from flowforge import (
    global_config, flow, task, step,
    FlowForge, CompileError,
)
from flowforge.annotations.decorators import _GLOBAL_ATTR, _FLOW_ATTR
from flowforge.schema.compiler import add_flow_to_dag
from flowforge.schema.dag import NodeType


# ---------------------------------------------------------------------------
# Fixtures: simple agent definitions at module level
# ---------------------------------------------------------------------------

@flow(name="alpha", prompt="first flow")
class AlphaFlow:
    @task(name="a_task", prompt="alpha task")
    class ATask:
        @step(order=1, prompt="alpha step")
        async def a_step(ctx):
            return {"result": "alpha"}


@flow(name="beta", prompt="second flow")
class BetaFlow:
    @task(name="b_task", prompt="beta task")
    class BTask:
        @step(order=1, prompt="beta step")
        async def b_step(ctx):
            return {"result": "beta"}


# Extra flow used to test add_flow()
@flow(name="gamma", prompt="third flow added dynamically")
class GammaFlow:
    @task(name="g_task", prompt="gamma task")
    class GTask:
        @step(order=1, prompt="gamma step")
        async def g_step(ctx):
            return {"result": "gamma"}


@global_config(prompt="test agent")
class BasicAgent:
    AlphaFlow = AlphaFlow
    BetaFlow = BetaFlow


@global_config(prompt="dynamic agent", dynamic_flow=True)
class DynamicAgent:
    AlphaFlow = AlphaFlow


# Custom _dynamic_generator flow for precedence test
@flow(name="_dynamic_generator", prompt="my custom generator")
class CustomGeneratorFlow:
    @task(name="custom_gen", prompt="custom generation")
    class CustomGenTask:
        @step(order=1, prompt="custom step")
        async def custom_step(ctx):
            return {"custom": True}


@global_config(prompt="agent with custom gen", dynamic_flow=True)
class CustomAgent:
    AlphaFlow = AlphaFlow
    CustomGeneratorFlow = CustomGeneratorFlow


def _fresh_basic():
    """Create a fresh BasicAgent to avoid test-to-test state leakage.

    add_flow() mutates global_meta.flows, so each test that calls it
    must use its own agent instance.
    """
    @global_config(prompt="test agent")
    class _Agent:
        AlphaFlow = AlphaFlow
        BetaFlow = BetaFlow
    return _Agent


# ---------------------------------------------------------------------------
# Test: dynamic_flow flag on GlobalMeta
# ---------------------------------------------------------------------------

class TestDynamicFlowFlag:

    def test_default_is_false(self):
        meta = getattr(BasicAgent, _GLOBAL_ATTR)
        assert meta.dynamic_flow is False

    def test_dynamic_flow_true(self):
        meta = getattr(DynamicAgent, _GLOBAL_ATTR)
        assert meta.dynamic_flow is True


# ---------------------------------------------------------------------------
# Test: CompiledAgent.add_flow()
# ---------------------------------------------------------------------------

class TestAddFlow:

    def test_add_flow_creates_new_node(self):
        engine = FlowForge.compile(_fresh_basic())
        initial_count = len(engine.dag.get_all_nodes())

        node_id = engine.add_flow(GammaFlow)

        assert node_id == "global.gamma"
        assert engine.dag.get_node("global.gamma") is not None
        assert len(engine.dag.get_all_nodes()) > initial_count

    def test_add_flow_subtree_is_complete(self):
        engine = FlowForge.compile(_fresh_basic())
        engine.add_flow(GammaFlow)

        # Should have: global.gamma, global.gamma.g_task,
        # global.gamma.g_task.g_step[1]
        assert engine.dag.get_node("global.gamma") is not None
        assert engine.dag.get_node("global.gamma.g_task") is not None
        assert engine.dag.get_node("global.gamma.g_task.g_step[1]") is not None

    def test_add_flow_duplicate_raises(self):
        engine = FlowForge.compile(_fresh_basic())
        with pytest.raises(CompileError, match="already exists"):
            engine.add_flow(AlphaFlow)

    def test_add_flow_not_decorated_raises(self):
        class Plain:
            pass

        engine = FlowForge.compile(_fresh_basic())
        with pytest.raises(CompileError, match="not decorated"):
            engine.add_flow(Plain)

    def test_add_flow_preserves_existing_nodes(self):
        engine = FlowForge.compile(_fresh_basic())
        # Capture existing nodes.
        original_ids = {n.id for n in engine.dag.get_all_nodes()}

        engine.add_flow(GammaFlow)

        current_ids = {n.id for n in engine.dag.get_all_nodes()}
        assert original_ids.issubset(current_ids)


# ---------------------------------------------------------------------------
# Test: CompiledAgent.recompile()
# ---------------------------------------------------------------------------

class TestRecompile:

    def test_recompile_rebuilds_dag(self):
        engine = FlowForge.compile(_fresh_basic())
        engine.add_flow(GammaFlow)
        count_after_add = len(engine.dag.get_all_nodes())

        engine.recompile()
        count_after_recompile = len(engine.dag.get_all_nodes())

        # After recompile, gamma is still present (it was added to global_meta.flows)
        assert count_after_recompile == count_after_add


# ---------------------------------------------------------------------------
# Test: add_flow_to_dag() — incremental compilation
# ---------------------------------------------------------------------------

class TestAddFlowToDag:

    def test_add_flow_to_existing_dag(self):
        engine = FlowForge.compile(_fresh_basic())
        dag = engine.dag

        meta = getattr(GammaFlow, _FLOW_ATTR)
        node_id = add_flow_to_dag(dag, meta)

        assert node_id == "global.gamma"
        node = dag.get_node(node_id)
        assert node is not None
        assert node.type == NodeType.FLOW

    def test_add_flow_duplicate_raises(self):
        engine = FlowForge.compile(_fresh_basic())
        dag = engine.dag

        meta = getattr(AlphaFlow, _FLOW_ATTR)
        with pytest.raises(CompileError, match="already exists"):
            add_flow_to_dag(dag, meta)


# ---------------------------------------------------------------------------
# Test: Meta-flow injection with dynamic_flow=True
# ---------------------------------------------------------------------------

class TestMetaFlowInjection:

    def test_dynamic_agent_has_meta_flow(self):
        engine = FlowForge.compile(DynamicAgent)

        # The _dynamic_generator flow should be present.
        node = engine.dag.get_node("global._dynamic_generator")
        assert node is not None
        assert node.type == NodeType.FLOW

    def test_dynamic_agent_has_meta_flow_steps(self):
        engine = FlowForge.compile(DynamicAgent)

        # Check the task exists.
        task_node = engine.dag.get_node(
            "global._dynamic_generator._generate_and_run"
        )
        assert task_node is not None

        # Check steps exist.
        step_ids = [
            n.id for n in engine.dag.get_all_nodes()
            if n.id.startswith("global._dynamic_generator._generate_and_run.")
            and n.type == NodeType.STEP
        ]
        assert len(step_ids) == 4  # analyse_gap, generate_code, compile_and_inject, execute_new_flow

    def test_basic_agent_has_no_meta_flow(self):
        engine = FlowForge.compile(BasicAgent)
        assert engine.dag.get_node("global._dynamic_generator") is None

    def test_compile_idempotent(self):
        """Calling compile() twice does not duplicate the meta-flow."""
        engine1 = FlowForge.compile(DynamicAgent)
        engine2 = FlowForge.compile(DynamicAgent)

        gen_nodes_1 = [
            n for n in engine1.dag.get_all_nodes()
            if "_dynamic_generator" in n.id
        ]
        gen_nodes_2 = [
            n for n in engine2.dag.get_all_nodes()
            if "_dynamic_generator" in n.id
        ]
        assert len(gen_nodes_1) == len(gen_nodes_2)

    def test_user_custom_dynamic_generator_takes_precedence(self):
        """If user defines their own _dynamic_generator flow, use it."""
        engine = FlowForge.compile(CustomAgent)

        # The _dynamic_generator node should exist
        node = engine.dag.get_node("global._dynamic_generator")
        assert node is not None

        # It should be the USER's version (has "custom_gen" task,
        # not "_generate_and_run" from the built-in)
        task_node = engine.dag.get_node(
            "global._dynamic_generator.custom_gen"
        )
        assert task_node is not None

        # The built-in "_generate_and_run" task should NOT exist
        builtin_task = engine.dag.get_node(
            "global._dynamic_generator._generate_and_run"
        )
        assert builtin_task is None


# ---------------------------------------------------------------------------
# Test: DynamicFlowGenerator.compile_flow_code()
# ---------------------------------------------------------------------------

class TestCompileFlowCode:

    def test_compile_valid_code(self):
        from flowforge.dynamic.generator import DynamicFlowGenerator
        from flowforge.types import LLMConfig

        engine = FlowForge.compile(BasicAgent)
        generator = DynamicFlowGenerator(
            llm_config=LLMConfig(),
            dag=engine.dag,
        )

        code = '''
from flowforge import flow, task, step

@flow(name="hello_flow", prompt="says hello")
class HelloFlow:
    @task(name="greet", prompt="greet the user")
    class GreetTask:
        @step(order=1, prompt="say hello")
        async def say_hello(ctx):
            return {"message": "hello!"}
'''
        meta = generator.compile_flow_code(code)
        assert meta.name == "hello_flow"
        assert len(meta.tasks) == 1
        assert meta.tasks[0].name == "greet"

    def test_compile_invalid_code_raises(self):
        from flowforge.dynamic.generator import DynamicFlowGenerator
        from flowforge.types import LLMConfig

        engine = FlowForge.compile(BasicAgent)
        generator = DynamicFlowGenerator(
            llm_config=LLMConfig(),
            dag=engine.dag,
        )

        with pytest.raises(CompileError):
            generator.compile_flow_code("this is not python code!!!")

    def test_compile_code_without_flow_raises(self):
        from flowforge.dynamic.generator import DynamicFlowGenerator
        from flowforge.types import LLMConfig

        engine = FlowForge.compile(BasicAgent)
        generator = DynamicFlowGenerator(
            llm_config=LLMConfig(),
            dag=engine.dag,
        )

        code = '''
class JustAClass:
    pass
'''
        with pytest.raises(CompileError, match="does not contain"):
            generator.compile_flow_code(code)


# ---------------------------------------------------------------------------
# Test: helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_strip_markdown_fences(self):
        from flowforge.dynamic.generator import _strip_markdown_fences

        assert _strip_markdown_fences("```python\ncode\n```") == "code"
        assert _strip_markdown_fences("```\ncode\n```") == "code"
        assert _strip_markdown_fences("code") == "code"

    def test_sanitise_name(self):
        from flowforge.dynamic.generator import _sanitise_name

        assert _sanitise_name("hello-world") == "hello_world"
        assert _sanitise_name("123abc") == "flow_123abc"
        assert _sanitise_name("My Flow!") == "my_flow"
        assert _sanitise_name("") == "dynamic_flow"
        assert _sanitise_name("valid_name") == "valid_name"


# ---------------------------------------------------------------------------
# Test: end-to-end add_flow + run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_flow_and_run():
    """After add_flow(), the new flow is runnable via route."""
    engine = FlowForge.compile(_fresh_basic())
    engine.add_flow(GammaFlow)

    result = await engine.run({"input": "test"}, route="gamma")
    assert result == {"result": "gamma"}


@pytest.mark.asyncio
async def test_original_flows_still_work_after_add():
    """Adding a flow does not break existing flows."""
    engine = FlowForge.compile(_fresh_basic())
    engine.add_flow(GammaFlow)

    result = await engine.run({"input": "test"}, route="alpha")
    assert result == {"result": "alpha"}
