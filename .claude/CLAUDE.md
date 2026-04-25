# FlowForge — Design Specification v1.1

> Annotation-Based AI Agent Framework  
> Version 1.1 | April 2026

---

## Claude Code — Working Notes

> Read this section first. It covers everything that is NOT obvious from the code.

### Project State
- Python 3.12, pip package at `/Users/jongyeon/workspace/flowforge/`
- **311 tests pass** — always verify with `python -m pytest tests/ -x -q` after any change
- Public API: `@global_config`, `@flow`, `@task`, `@step`, `FlowForge.compile()`
- **No `@branch` decorator** — removed. Branch dispatching is a parameter of `@step`/`@task`/`@flow`
- **Dynamic flow generation**: `@global_config(dynamic_flow=True)` + `DynamicRunOptions`

### Key Files

| File | Role |
|------|------|
| `flowforge/annotations/decorators.py` | All four decorators — primary entry point for annotation changes |
| `flowforge/annotations/metadata.py` | `StepMeta`, `TaskMeta`, `FlowMeta`, `GlobalMeta` dataclasses |
| `flowforge/annotations/validators.py` | Compile-time validators (run at `@task` decoration time) |
| `flowforge/execution/runner.py` | `StepRunner`, `TaskRunner`, `FlowRunner` — runtime execution logic |
| `flowforge/schema/compiler.py` | Annotation → DAG nodes/edges |
| `flowforge/schema/dag.py` | `FlowForgeDAG`, `DAGNode`, `NodeType` (GLOBAL/FLOW/TASK/STEP only) |
| `flowforge/errors.py` | All exception types |
| `flowforge/types.py` | `LLMConfig`, `BranchCondition`, `MCPServer`, `ToolConfig`, `DynamicRunOptions`, `DependencyPolicy` |
| `flowforge/dynamic/__init__.py` | Dynamic flow generation public API |
| `flowforge/dynamic/generator.py` | `DynamicFlowGenerator` — LLM code gen, AST safety, compile, retry |
| `flowforge/dynamic/meta_flow.py` | Built-in `_dynamic_generator` meta-flow (3-step: analyse → prepare → generate+inject) |
| `flowforge/dynamic/manifest.py` | `DynamicManifest`, `GeneratedFlowRecord`, `GeneratedToolRecord`, file-locked persistence |
| `flowforge/tools/builtin.py` | Builtin tool pack: shell tools + utility tools (web, json, files) |
| `flowforge/planner/llm_planner.py` | LLM planner with requirement decomposition and gap detection |
| `flowforge/planner/prompt_builder.py` | Planner prompt assembly (flow-level only) |
| `tests/test_annotations.py` | Decorator unit tests |
| `tests/test_dag_compile.py` | DAG compilation tests |
| `tests/test_execution.py` | End-to-end execution tests |
| `tests/test_run_trace.py` | Run trace and Mermaid visualisation tests |
| `tests/test_validators.py` | Validator unit tests |
| `tests/test_tools_and_llm.py` | Hierarchical tools & `ctx.call_llm()` tests |
| `tests/test_route_and_loop.py` | Route filtering & task loop tests |
| `tests/test_dynamic_flow.py` | Dynamic flow generation: AST safety, manifest, contract, partial gap, utility tools |
| `tests/test_planner.py` | Planner prompt building and gap metadata round-trip |

### Critical: Python Scoping in Tests

Agent classes used in async tests **must be defined at module level**, not inside test functions. Python class bodies cannot reference local variables from enclosing function scopes.

```python
# BAD — NameError at runtime when MyFlow tries to reference MyTask
async def test_foo():
    @task(name="t", prompt="t")
    class MyTask: ...

    @flow(name="f", prompt="f")
    class MyFlow:
        MyTask = MyTask  # NameError — MyTask is a local var

# GOOD — define everything at module level, reference by name in test
@task(name="t", prompt="t")
class MyTask: ...

@flow(name="f", prompt="f")
class MyFlow:
    MyTask = MyTask  # OK — module-level name

@global_config(prompt="test agent")
class MyAgent:
    MyFlow = MyFlow

async def test_foo():
    engine = FlowForge.compile(MyAgent)
    ...
```

### Decorator Processing Order (Inner Before Outer)

Python processes inner class decorators before outer ones:

```
@step     → sets __flowforge_step_meta__ on the function
@task     → scans cls.__dict__ for step/task attrs → builds TaskMeta
@flow     → scans for flow/task attrs → builds FlowMeta
@global_config → scans for flow attrs → builds GlobalMeta
FlowForge.compile() → reads GlobalMeta → builds DAG + validates
```

### Order, Parallelism & Uniqueness (v1.1)

| Scenario | Behaviour |
|----------|-----------|
| `order=None` (default on `@flow`/`@task`) | Auto-sequential: each node gets its own slot, insertion order preserved |
| Same explicit `order` integer on siblings | **Parallel execution** — each node receives the same input; last result forwarded |
| `unique=True` on one node in a same-order group | Only that node runs; siblings are skipped |
| Multiple `unique=True` at same order | `OrderConflictError` at decoration time |

### Branch Dispatching API

```python
# Step-level: branches are async callables
@step(order=2, prompt="route",
      condition=BranchCondition(field="source", enum=["web","db"]),
      branches={"web": web_handler, "db": db_handler},
      fallback=web_handler)
async def route(ctx): ...

# Task-level: branches are @task-decorated classes
@task(name="dispatch", prompt="...",
      condition=BranchCondition(field="mode", enum=["fast","slow"]),
      branches={"fast": FastTask, "slow": SlowTask})
class DispatchTask: ...

# Flow-level: branches are @flow-decorated classes
@flow(name="dispatch", prompt="...",
      condition=BranchCondition(field="type", enum=["a","b"]),
      branches={"a": FlowA, "b": FlowB})
class DispatchFlow: ...
```

`StepContext.selected_branch` and `.condition_value` are populated by the runner before any branch handler is called.

### Route Execution

Execute only specific flows or tasks by passing `route` to `engine.run()`:

```python
engine = FlowForge.compile(MyAgent)

# Run only the "beta" flow
result = await engine.run(data, route="beta")

# Run only a specific task within a flow
result = await engine.run(data, route="alpha.a1")

# Run multiple routes
result = await engine.run(data, route=["alpha", "gamma"])
```

- `dag.resolve_route("flow_name")` returns all ancestor + descendant node IDs
- `dag.resolve_route("flow.task")` returns the task branch + ancestors
- Invalid route segments raise `ValueError`
- Route overrides `planning_mode` when set

### Task Loop (Retry with Condition)

Tasks can loop until a condition is met using `max_loops` and `loop_condition`:

```python
@task(
    name="retry_task",
    prompt="keep trying until valid",
    max_loops=5,
    loop_condition=lambda output: output.get("valid", False),
)
class RetryTask:
    @step(order=1, prompt="produce result")
    async def produce(ctx):
        # loop_condition(output) → True = accept, stop looping
        # loop_condition(output) → False = discard, re-run steps
        return {"valid": some_check(), "data": result}
```

- `loop_condition` receives the task output dict; returns `True` to accept
- When `loop_condition` returns `False`, the task re-runs its step chain
- After `max_loops` exhausted, the last result is returned regardless
- `max_loops` without `loop_condition` has no effect (runs once)
- `loop_condition` without `max_loops` defaults to `max_loops=1` (no loop)

### DAG Node ID Format

```
global.<flow_name>.<task_name>.<step_func_name>[<order>]
# e.g. global.search.execute_search.optimize_query[1]
```

This format is used by both the compiler and the runner — keep them in sync when changing either.

### What to Avoid

- **Never** add `@branch` back — it was intentionally removed. Use `@step(condition=..., branches={...})`.
- **Never** import `BranchContext` or `BranchRunner` — they do not exist.
- **Never** reference `NodeType.BRANCH` — only GLOBAL/FLOW/TASK/STEP exist.
- **Never** use `T = T` syntax inside a class body to reference a locally-scoped class.
- **Never** skip `python -m pytest tests/ -x -q` after changes.
- **Never** call `sys.modules.pop()` on dynamically generated modules — it breaks `__module__` references needed at runtime.
- **Never** bypass AST safety validation (`_validate_generated_ast`) when executing generated code.
- **Never** write to manifest.json without acquiring `_manifest_lock` — concurrent corruption risk.

---

## 1. Overview

FlowForge는 Python pip 모듈로, LangChain/LangGraph 같은 복잡한 프레임워크 없이 **Python 데코레이터(어노테이션)만으로 AI Agent 시스템을 구축**할 수 있게 해준다.

유저는 `@flow`, `@task`, `@step`, `@branch`, `@global` 어노테이션으로 에이전트 구조를 정의하고, 프레임워크가 이를 **DAG(Directed Acyclic Graph) 스키마**로 컴파일한다. AI Planner는 이 스키마를 "지도"로 삼아 유저 요청에 최적화된 **실행 경로(서브트리)**를 선택하거나 자율적으로 구성한다.

### 1.1 Design Principles

- **Annotation-First**: 모든 에이전트 구조가 Python 데코레이터로 정의됨. 그래프 구성 코드 불필요.
- **Type-Safe Data Flow**: Pydantic 모델이 모든 어노테이션 경계에서 I/O 계약을 강제.
- **DAG-Native**: 어노테이션 계층이 DAG로 컴파일되어 위상 정렬, 순환 감지 가능.
- **Dual-Prompt System**: 각 어노테이션은 유저 작성 `prompt`와 AI 자동 생성 `doc`을 보유.
- **Recursive Nesting**: Flow 안에 Flow, Task 안에 Task(트리 구조) 가능.
- **Tool-Agnostic Integration**: MCP, HTTP, Python 함수를 통합 인터페이스로 연결.

---

## 2. Dual-Prompt 시스템: `prompt` vs `doc`

모든 어노테이션(@global, @flow, @task, @step, @branch)은 **두 가지 텍스트**를 가진다.

### 2.1 `prompt` — 유저가 작성

유저가 직접 작성하는 자연어 설명. 해당 어노테이션이 "무엇을 해야 하는지"를 기술한다.

```python
@step(order=1, prompt="유저가 입력한 문서의 포맷을 감지하고 유효성을 검증한다")
async def validate_input(ctx): ...
```

### 2.2 `doc` — AI가 자동 생성

**최초 컴파일 시** AI가 모든 어노테이션의 `prompt`를 분석하여 자동으로 생성하는 구조화된 문서. 각 어노테이션마다 개별 `doc`이 생성된다.

`doc`에 포함되는 정보:

| 항목 | 설명 |
|------|------|
| `summary` | 해당 어노테이션의 역할을 1~2문장으로 요약 |
| `input_schema_desc` | 입력 Pydantic 모델의 필드별 의미 설명 |
| `output_schema_desc` | 출력 Pydantic 모델의 필드별 의미 설명 |
| `preconditions` | 이 노드가 실행되기 위해 충족해야 할 조건 |
| `capabilities` | 이 노드가 수행할 수 있는 능력 목록 |
| `children_overview` | (Flow/Task만) 하위 노드 구조 요약 |
| `routing_hints` | (Branch만) 어떤 상황에서 어떤 분기를 선택해야 하는지 |

#### `doc` 생성 예시

유저가 아래처럼 작성하면:

```python
@flow(name="data_pipeline", prompt="다양한 포맷의 데이터를 수집, 정제, 변환하는 파이프라인")
class DataPipeline:
    @task(name="ingest", prompt="외부 소스에서 원시 데이터를 가져온다")
    class IngestTask: ...

    @task(name="transform", prompt="정제된 데이터를 목적에 맞게 변환한다")
    class TransformTask: ...
```

AI가 컴파일 시 생성하는 `doc`:

```json
{
  "node": "flow:data_pipeline",
  "doc": {
    "summary": "CSV, JSON, XML 등 다양한 외부 데이터를 수집·정제·변환하는 파이프라인",
    "input_schema_desc": {
      "source_url": "데이터를 가져올 외부 소스 URL",
      "format_hint": "예상 데이터 포맷 (optional)"
    },
    "output_schema_desc": {
      "records": "변환 완료된 레코드 리스트",
      "metadata": "처리 통계 (건수, 소요시간 등)"
    },
    "preconditions": ["유효한 source_url이 제공되어야 함"],
    "capabilities": ["CSV/JSON/XML 파싱", "스키마 정규화", "결측치 처리"],
    "children_overview": "ingest(수집) → transform(변환) 순서로 2개 Task 실행"
  }
}
```

### 2.3 `doc`이 AI 경로 결정에 사용되는 방식

```
유저 요청
  → AI Planner가 모든 노드의 doc을 스캔
  → doc.summary + doc.capabilities로 관련 노드 필터링
  → doc.preconditions로 실행 가능 여부 판단
  → doc.routing_hints로 Branch 분기 결정
  → 최적 서브트리(실행 경로) 확정
```

AI Planner는 `doc`을 통해 각 노드가 무엇을 할 수 있는지 파악하고, 유저 요청을 처리하기 위해 **어떤 Flow의 어떤 Task까지 실행해야 하는지** 결정한다. `prompt`는 실제 실행 시 LLM에게 전달되는 instruction이고, `doc`은 경로 선택 시 참조되는 메타데이터다.

---

## 3. Technology Stack

| Category | Library / Version | Purpose |
|----------|-------------------|---------|
| Language | Python 3.11+ | async/await, type hints, decorator protocol |
| Type System | Pydantic v2.7+ | I/O 스키마 검증, JSON Schema 생성 |
| LLM Client | anthropic 0.40+ / openai 1.50+ | AI Planner 호출, structured output (tool_use) |
| Async Runtime | asyncio + anyio 4.x | 병렬 Flow 실행, TaskGroup 관리 |
| DAG Engine | networkx 3.3+ | DAG 구축, 위상 정렬, 순환 감지 |
| MCP | mcp 1.x (Anthropic SDK) | 외부 도구 연동 |
| Visualization | graphviz 0.20+ / mermaid-py | DAG 렌더링, 실행 흐름 시각화 |
| Logging | structlog 24.x | 구조화된 JSON 로깅 |
| Testing | pytest 8.x + pytest-asyncio | 비동기 테스트, 어노테이션 fixture |
| CLI | typer 0.12+ | `flowforge viz`, `flowforge validate`, `flowforge run` |

---

## 4. Annotation Hierarchy & Nesting Rules

### 4.1 전체 계층 구조

```
@global
 └─ @flow (root)
     ├─ @flow (child)              ← Flow 안에 Flow 중첩 가능
     │   ├─ @flow (grandchild)     ← 재귀적 무한 중첩
     │   └─ @task (leaf)
     ├─ @task (parent)             ← Task도 트리 구조 가능
     │   ├─ @task (child, leaf)
     │   │   ├─ @step  (order=1)
     │   │   ├─ @branch(order=2)
     │   │   └─ @step  (order=3)
     │   └─ @task (child, leaf)
     │       ├─ @step  (order=1)
     │       └─ @step  (order=2)
     └─ @flow (sibling)
         └─ @task (leaf)
             ├─ @step  (order=1)
             ├─ @branch(order=2)   ← Branch에도 order 필수
             └─ @step  (order=3)
```

### 4.2 Nesting Rules 요약

| 부모 | 허용되는 자식 | 제약 |
|------|-------------|------|
| `@global` | `@flow` (root-level) | 하나의 Global 인스턴스만 존재 |
| `@flow` | `@flow`, `@task` | 자식 `@flow`와 `@task`를 동시에 가질 수 있음. 하위 flow 처리 후 task가 마무리하는 패턴 가능 |
| `@task` | `@task`, `@step`, `@branch` | 자식 `@task`가 있으면 컨테이너. **리프 task만** `@step`/`@branch`를 직접 포함 |
| `@step` | (없음) | 리프 실행 단위 |
| `@branch` | (핸들러 참조) | 각 분기가 실행할 handler 함수를 참조. 핸들러는 별도 정의 |

### 4.3 핵심 규칙: Task 내 `order` 유일성

**하나의 리프 Task 내에서 모든 @step과 @branch의 `order` 번호는 유일해야 한다.**

```python
@task(name="process")
class ProcessTask:
    @step(order=1, ...)          # ✅ order=1
    async def validate(ctx): ...

    @branch(order=2, ...)        # ✅ order=2 (branch도 order를 가짐)
    async def route(ctx): ...

    @step(order=3, ...)          # ✅ order=3
    async def transform(ctx): ...

    @step(order=3, ...)          # ❌ COMPILE ERROR: order=3 중복!
    async def finalize(ctx): ...
```

order는 실행 순서를 결정하며, Step과 Branch가 섞여 있어도 **order 번호 기준으로 정렬**되어 순차 실행된다.

---

## 5. Annotation 상세 명세

### 5.1 @global

환경 전체에 적용되는 규칙과 설정을 정의한다.

```python
@global_config(
    prompt="너는 데이터 처리 전문 에이전트이다. 항상 한국어로 응답한다.",
    llm_config=LLMConfig(model="claude-sonnet-4-20250514", temperature=0.3),
    tools=[MCPServer("https://api.example.com/mcp")]
)
class MyAgent: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | 전역 시스템 프롬프트 |
| `llm_config` | `LLMConfig` | 기본 LLM 설정 (모델, temperature 등) |
| `tools` | `list[ToolConfig]` | 전역 도구 등록 (MCP, HTTP 등) |
| `doc` | `GlobalDoc` (auto) | AI 생성: 전체 에이전트 역할 요약 + 전체 구조 개요 |

**Design Pattern**: Singleton + Registry

---

### 5.2 @flow

최상위 실행 단위. Flow끼리 DAG 구조로 연결되며, **Flow 안에 또 다른 Flow를 중첩**할 수 있다.

```python
@flow(
    name="data_pipeline",
    prompt="다양한 포맷의 데이터를 수집, 정제, 변환하는 파이프라인",
    input_schema=RawDataInput,
    output_schema=ProcessedData,
    depends_on=["auth_flow"],
    parallel=False
)
class DataPipelineFlow:

    # 자식 Flow (중첩)
    @flow(name="ingestion", prompt="외부 소스에서 데이터를 수집한다")
    class IngestionSubFlow:
        @task(name="fetch", prompt="HTTP/FTP로 원시 데이터를 가져온다")
        class FetchTask: ...

    # 같은 레벨의 또 다른 자식 Flow
    @flow(name="validation", prompt="수집된 데이터의 무결성을 검증한다")
    class ValidationSubFlow:
        @task(name="check", prompt="스키마 및 값 범위 검증")
        class CheckTask: ...

    # 리프 Task (하위 Flow 처리 후 실행)
    @task(name="transform", prompt="정제된 데이터를 변환한다")
    class TransformTask: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | 고유 식별자 |
| `prompt` | `str` | (required) | 이 Flow의 역할 설명 (유저 작성) |
| `doc` | `FlowDoc` | (auto) | AI 생성: 역할 요약, I/O 설명, 하위 구조 개요, 실행 조건 |
| `input_schema` | `Type[BaseModel]` | `None` | 입력 Pydantic 모델 |
| `output_schema` | `Type[BaseModel]` | `None` | 출력 Pydantic 모델 |
| `depends_on` | `list[str]` | `[]` | 선행 Flow 이름 (DAG 엣지 생성) |
| `parallel` | `bool` | `False` | 자식 노드 병렬 실행 여부 |
| `max_retries` | `int` | `3` | 실패 시 재시도 횟수 |

**Design Pattern**: Composite Pattern

**Flow-Flow 트리 구조**:
```
FlowA (root)
├─ FlowB (child)
│   ├─ FlowD (grandchild → has tasks)
│   └─ FlowE (grandchild → has tasks)
├─ FlowC (child → has tasks)
└─ depends_on으로 FlowB → FlowC 실행 순서 강제 가능
```

---

### 5.3 @task

Flow가 실행하는 실행 단위. **Task도 트리 구조를 가질 수 있다.** 자식 Task를 가진 Task는 컨테이너, 리프 Task만 @step/@branch를 직접 포함한다.

```python
@task(name="document_processing", prompt="문서를 분석하고 처리한다")
class DocumentProcessingTask:

    # 자식 Task (트리 구조)
    @task(name="parse", prompt="문서 포맷을 파싱한다")
    class ParseTask:
        @step(order=1, prompt="포맷 감지")
        async def detect_format(ctx): ...

        @branch(order=2, condition=BranchCondition(
            field="format", enum=["pdf","docx","html"]))
        async def route_parser(ctx): ...

        @step(order=3, prompt="파싱 결과 정규화")
        async def normalize(ctx): ...

    @task(name="analyze", prompt="파싱된 문서의 내용을 분석한다")
    class AnalyzeTask:
        @step(order=1, prompt="키워드 추출")
        async def extract_keywords(ctx): ...

        @step(order=2, prompt="요약 생성")
        async def summarize(ctx): ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | 고유 식별자 |
| `prompt` | `str` | (required) | Task 역할 설명 (유저 작성) |
| `doc` | `TaskDoc` | (auto) | AI 생성: 역할 요약, 하위 step/branch 구조 설명, I/O 계약 |
| `input_schema` | `Type[BaseModel]` | `None` | 입력 모델 |
| `output_schema` | `Type[BaseModel]` | `None` | 출력 모델 |

**Design Pattern**: Composite(트리) + Chain of Responsibility(내부 step 체인)

**Task 트리 구조**:
```
TaskA (container)
├─ TaskB (leaf) → [step(1), branch(2), step(3)]
└─ TaskC (leaf) → [step(1), step(2)]
```
컨테이너 Task는 자식 Task를 순차 실행하며, 자식 간 output→input 타입이 연결된다.

---

### 5.4 @step

Task 내에서 **구체적인 동작 단계**를 정의한다. 동일 Task 내에서 order 기준 순차 실행.

```python
@step(
    order=1,
    prompt="입력 문서의 스키마를 검증한다",
    input_schema=RawDoc,
    output_schema=ValidatedDoc,
    tool_mode=False,
    timeout_seconds=30
)
async def validate_schema(ctx: StepContext) -> ValidatedDoc:
    ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | `int` | (required) | 실행 순서. **Task 내에서 유일** (step/branch 통합) |
| `prompt` | `str` | (required) | 이 Step이 수행할 동작 설명 (유저 작성) |
| `doc` | `StepDoc` | (auto) | AI 생성: 동작 요약, I/O 필드 설명, 실행 조건 |
| `input_schema` | `Type[BaseModel]` | `None` | 입력 모델 (None이면 이전 order 노드 output 자동 바인딩) |
| `output_schema` | `Type[BaseModel]` | `None` | 출력 모델 |
| `tool_mode` | `bool` | `False` | True면 LLM tool_use로 등록 (Agent Tool처럼 동작) |
| `timeout_seconds` | `int` | `60` | 실행 제한 시간 |

**Design Pattern**: Chain of Responsibility

---

### 5.5 @branch

Task 내에서 **조건부 분기**를 정의한다. Branch도 `order`를 가지며 Step과 동일한 순서 체계에 참여한다.

```python
@branch(
    order=2,
    name="format_router",
    prompt="문서 포맷에 따라 적절한 파서를 선택한다",
    condition=BranchCondition(
        field="doc_type",
        enum=["csv", "json", "xml"]
    ),
    branches={
        "csv":  csv_handler,
        "json": json_handler,
        "xml":  xml_handler,
    },
    fallback=default_handler
)
async def route_by_format(ctx: BranchContext):
    ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | `int` | (required) | 실행 순서. **Task 내에서 유일** (step과 통합 번호 체계) |
| `name` | `str` | (required) | 분기점 식별자 |
| `prompt` | `str` | (required) | 분기 기준 설명 (유저 작성) |
| `doc` | `BranchDoc` | (auto) | AI 생성: 분기 조건 설명, 각 경로별 역할, routing_hints |
| `condition` | `BranchCondition` | (required) | 분기 조건 (discriminator field + enum) |
| `branches` | `dict[str, Callable]` | (required) | 분기값 → 핸들러 매핑 |
| `fallback` | `Callable` | `None` | 조건 불일치 시 기본 핸들러 |

**Design Pattern**: Strategy Pattern + Discriminated Union

**핵심**: 모든 분기 핸들러의 output_schema는 **동일 타입**이어야 한다. 다음 order 노드의 input_schema와 호환 보장.

---

## 6. Order 체계와 실행 순서

### 6.1 통합 Order 규칙

하나의 **리프 Task** 내에서 @step과 @branch는 **같은 order 번호 공간을 공유**한다.

```
Task "process_document"
│
├─ order=1  @step   "validate_input"      in:RawDoc      → out:ValidDoc
├─ order=2  @branch "route_by_format"     in:ValidDoc    → out:ParsedDoc
├─ order=3  @step   "enrich_metadata"     in:ParsedDoc   → out:EnrichedDoc
└─ order=4  @step   "save_result"         in:EnrichedDoc → out:SaveResult
```

**규칙 정리**:

1. order 번호는 Task 내에서 **정수, 양수, 유일**
2. Step과 Branch가 같은 번호 공간 → `order=2`가 step이든 branch이든 하나만 존재
3. 실행은 order 오름차순으로 **순차적**
4. order가 연속적일 필요는 없음 (1, 3, 7, 10 가능)
5. 각 노드의 output_schema가 다음 order 노드의 input_schema와 일치해야 함

### 6.2 컴파일 시 검증

Schema Compiler가 수행하는 order 관련 검증:

```
[CHECK] Task 내 order 중복 검사 → OrderConflictError
[CHECK] order 기준 정렬 후 연속 노드 간 I/O 타입 호환성 → IOBindingError
[CHECK] Branch의 모든 핸들러 output이 동일 타입인지 → BranchOutputMismatchError
[CHECK] Branch 핸들러 output → 다음 order 노드 input 호환 → IOBindingError
[CHECK] 첫 번째 order 노드 input ↔ Task의 input_schema 호환
[CHECK] 마지막 order 노드 output ↔ Task의 output_schema 호환
```

---

## 7. Data Flow Architecture

### 7.1 Context 계층 구조

```
GlobalContext
 │  ├─ llm_config, global_prompt, tool_registry, env_vars
 │  └─ all_docs: dict[str, Doc]  ← 모든 노드의 doc 저장
 │
 └─▸ FlowContext
      │  ├─ global_ctx (참조)
      │  ├─ flow_prompt, flow_doc
      │  ├─ parent_flow_output (상위 Flow 결과, 중첩 시)
      │  └─ flow_state: dict (Flow 범위 상태)
      │
      └─▸ TaskContext
           │  ├─ flow_ctx (참조)
           │  ├─ task_prompt, task_doc
           │  ├─ step_results: OrderedDict[int, Any]  ← order별 결과 누적
           │  └─ parent_task_output (상위 Task 결과, 트리 구조 시)
           │
           ├─▸ StepContext
           │    ├─ task_ctx (참조)
           │    ├─ step_prompt, step_doc
           │    ├─ input: BaseModel (이전 order 노드의 output)
           │    ├─ tools: ToolAccessor
           │    └─ previous_results: dict
           │
           └─▸ BranchContext
                ├─ task_ctx (참조)
                ├─ branch_prompt, branch_doc
                ├─ input: BaseModel
                ├─ condition_value: Any
                └─ selected_branch: str
```

### 7.2 데이터 전달 메커니즘

어노테이션 경계를 넘을 때마다:

```
이전 노드 return → Pydantic model_validate() → dict 직렬화 → 다음 노드 input 주입
```

### 7.3 구체적 흐름 예시

```
Flow A (input: UserQuery)
│
├─ [FlowContext 생성: GlobalContext + UserQuery 주입]
│
├─▸ Child Flow B (input: Flow A의 input을 상속 또는 변환)
│    │
│    └─▸ Task X (input: Flow B에서 전달)
│         │
│         ├─ order=1 Step: validate
│         │   input  ← TaskContext.input (= Flow B의 input)
│         │   output → ValidatedData
│         │
│         ├─ order=2 Branch: route
│         │   input  ← order=1의 output (ValidatedData)
│         │   condition_value = input.data_type
│         │   selected = "csv"
│         │   output → ParsedData
│         │
│         └─ order=3 Step: transform
│             input  ← order=2의 output (ParsedData)
│             output → TransformedData → Task X 최종 output
│
│    Task X output → Flow B output
│
├─▸ Task Y (input: Flow B output 또는 Flow A input)
│    └─ ...
│
└─ Flow A output = 마지막 자식의 output
```

### 7.4 I/O Binding Rules

| 연결 | 타입 규칙 |
|------|----------|
| Flow → 자식 Flow | 부모 input 또는 이전 형제 Flow output이 자식 input과 호환 |
| Flow → Task | Flow의 resolved input → Task input. Task output → Flow output |
| Task → 자식 Task | 부모 input → 첫 자식 input. 마지막 자식 output → 부모 output |
| Step[n] → Step[n+1] | `output_schema == input_schema` (정확 일치) |
| Step → Branch | Step output이 Branch condition의 discriminator field 포함 필수 |
| Branch → Step | 모든 핸들러 output이 동일 타입이고, 다음 Step input과 호환 |

### 7.5 tool_mode에서의 I/O

`tool_mode=True`인 Step/Branch는 고정 순서 대신 **LLM이 동적으로 호출**한다:

- input: LLM tool_use 호출 시 전달하는 JSON (input_schema 기반)
- output: 함수 반환값이 Pydantic 검증 후 LLM context로 반환
- tool_mode 노드는 order 체인에서 **빠지며**, AI Planner가 필요 시 호출

---

## 8. AI Planner Design

### 8.1 Schema-to-Prompt 컴파일

Schema Compiler가 어노테이션 DAG를 구조화된 프롬프트로 변환한다. 포함 정보:

1. **전체 DAG 구조**: 노드 목록, 엣지 관계, 각 노드의 `doc`
2. **가용 경로**: 가능한 서브트리 목록과 각 경로의 I/O 계약
3. **현재 실행 상태**: 완료된 노드, 누적 context
4. **도구 목록**: MCP/함수 도구의 스키마
5. **유저 요청**: 원본 입력

### 8.2 경로 선택 모드

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Deterministic** | 사전 정의 서브트리만 선택. `doc.preconditions` + `capabilities` 매칭. | 프로덕션, 예측 가능성 |
| **Autonomous** | DAG에서 새 경로 자율 구성. 노드 건너뛰기, 병렬 재정렬 가능. | 탐색, 프로토타이핑 |
| **Hybrid** | 사전 정의 경로 + 범위 내 이탈 허용 (tool 선택, branch 자율 판단). | 균형 잡힌 제어 |

### 8.3 Prompt Assembly Pipeline

```
1. @global prompt + global doc
2. 현재 Flow의 prompt + doc + 가용 하위 경로
3. 현재 Task/Step의 prompt + doc + I/O schema (JSON Schema)
4. 이전 실행 결과 + 누적 context
5. 도구 목록 + tool_use schema
6. 유저 원본 요청
```

---

## 9. Design Patterns by Component

| Component | Pattern | Rationale |
|-----------|---------|-----------|
| **SchemaRegistry** | Singleton + Registry | 모든 메타데이터의 단일 진실 소스 |
| **Flow 계층** | Composite | 컨테이너(자식 Flow)와 리프(Task 보유)를 통합 인터페이스로 |
| **Task 계층** | Composite | 컨테이너 Task(자식 Task)와 리프 Task를 통합 |
| **Step 체인** | Chain of Responsibility | order 순차 실행 + 타입 핸드오프 |
| **Branch 분기** | Strategy | discriminator 기반 런타임 경로 선택 |
| **Doc 생성** | Template Method | 고정 구조 + 노드별 커스텀 로직 |
| **Tool 통합** | Adapter + Plugin | MCP/함수/HTTP를 ToolAdapter로 통합 |
| **Context 전달** | Context Object | 계층적 context로 상태 전파 |
| **DAG Builder** | Builder | 분산 어노테이션에서 점진적 DAG 구축 |
| **AI Planner** | Template Method | 프롬프트 조립 고정 + 경로 선택 전략 교체 |
| **Execution Engine** | Visitor | 노드 타입별 핸들러 디스패치 |
| **Output Adapters** | Adapter | 다양한 싱크로 라우팅 |
| **Error Handling** | Circuit Breaker + Retry | 노드별 재시도 + 캐스케이드 방지 |

---

## 10. Module Structure

```
flowforge/
├── __init__.py                # Public API: @flow, @task, @step, @branch, @global_config
├── annotations/
│   ├── __init__.py            # 데코레이터 re-export
│   ├── decorators.py          # 데코레이터 구현
│   ├── metadata.py            # FlowMeta, TaskMeta, StepMeta, BranchMeta
│   └── validators.py          # order 유일성, I/O 호환성 검증
├── schema/
│   ├── registry.py            # SchemaRegistry (singleton)
│   ├── compiler.py            # 어노테이션 → DAG + doc 생성 트리거
│   ├── dag.py                 # DAG 노드/엣지 모델, networkx 래퍼
│   └── resolver.py            # 의존성 해석, 위상 정렬, 순환 감지
├── dynamic/
│   ├── __init__.py            # Dynamic flow generation public API
│   ├── generator.py           # DynamicFlowGenerator: LLM codegen, AST safety, compile, retry
│   ├── meta_flow.py           # _dynamic_generator 메타플로우 (analyse→prepare→generate+inject)
│   └── manifest.py            # DynamicManifest, file-locked persistence (flows + tools)
├── doc/
│   ├── generator.py           # LLM 기반 doc 자동 생성
│   ├── models.py              # GlobalDoc, FlowDoc, TaskDoc, StepDoc, BranchDoc
│   └── cache.py               # doc 캐시 (prompt 변경 시만 재생성)
├── planner/
│   ├── base.py                # AbstractPlanner
│   ├── llm_planner.py         # LLM planner: 경로 선택, requirement decomposition, gap detection
│   ├── prompt_builder.py      # Schema + doc → 프롬프트 조립 (flow-level)
│   └── path_selector.py       # Deterministic/Autonomous/Hybrid
├── execution/
│   ├── engine.py              # DAG 실행기 (Visitor) + dynamic flow orchestration
│   ├── runner.py              # Flow/Task/Step/Branch 개별 실행기
│   ├── parallel.py            # anyio TaskGroup 병렬 실행
│   ├── context.py             # Context 계층 구현
│   └── llm.py                 # ctx.call_llm() 구현
├── tools/
│   ├── __init__.py            # Tool registration + builtin tool injection
│   ├── builtin.py             # Builtin tool pack: shell tools (mode-gated) + utility tools
│   ├── registry.py            # ToolRegistry
│   ├── mcp_adapter.py         # MCP 어댑터
│   ├── function_tool.py       # 함수 → tool 래퍼
│   ├── http_adapter.py        # HTTP API 어댑터
│   └── base.py                # ToolAdapter 추상 베이스
├── output/
│   ├── adapters.py            # Output 싱크 (API, file, webhook)
│   └── hooks.py               # Pre/post 콜백
├── viz/
│   ├── renderer.py            # DAG 렌더링
│   └── trace.py               # 실시간 트레이스
├── cli/
│   └── main.py                # flowforge viz | validate | run | doc-generate
└── errors.py                  # OrderConflictError, IOBindingError 등
```

---

## 11. Execution Lifecycle

### Phase 1: Compile

```
데코레이터 → SchemaRegistry 등록
           → order 유일성 검증, I/O 호환성 검증
           → DAG 구축 (networkx), 위상 정렬, 순환 감지
           → 출력: FlowSchema (immutable)
```

### Phase 2: Doc Generation

```
FlowSchema + 모든 prompt → LLM 호출
                         → 노드별 doc 생성
                         → 캐시 저장
                         → 출력: DocRegistry
```

doc은 **최초 컴파일 시 1회 생성**, prompt 변경 없으면 캐시 재사용.

### Phase 3: Plan

```
유저 요청 + FlowSchema + DocRegistry → 프롬프트 조립
                                     → LLM 호출 (경로 선택)
                                     → 출력: ExecutionPlan (서브트리)
```

### Phase 4: Execute

```
ExecutionPlan → Visitor로 DAG 순회
             → 각 노드: Context 생성 → Input 검증 → 실행 → Output 검증 → 전달
             → 병렬 Flow: anyio TaskGroup
             → 출력: FlowOutput
```

---

## 12. Error Handling & Resilience

| Mechanism | Scope | Configuration |
|-----------|-------|---------------|
| Retry + backoff | Step/Task/Flow | `max_retries`, `backoff_factor` |
| Circuit breaker | 외부 tool 호출 | `failure_threshold`, `recovery_timeout` |
| Fallback branch | Branch | `@branch(fallback=handler)` |
| Checkpoint | Flow | Task 완료 후 자동 체크포인트 |
| Timeout | Step/Task | `timeout_seconds` |
| Validation error | I/O 경계 | Pydantic error → LLM 보정 재시도 |
| Dead letter | Flow | 최종 실패 → dead letter queue |

---

## 13. Complete Usage Example

```python
from flowforge import global_config, flow, task, step, branch
from flowforge.tools import MCPServer
from flowforge.types import BranchCondition, LLMConfig
from pydantic import BaseModel

# ─── Schemas ───
class UserQuery(BaseModel):
    query: str
    language: str = "ko"

class AnalyzedQuery(BaseModel):
    intent: str
    keywords: list[str]
    source_preference: str  # "web" | "db" | "api"

class SearchResult(BaseModel):
    results: list[dict]
    source: str

class FormattedAnswer(BaseModel):
    answer: str
    citations: list[str]


# ─── Agent Definition ───
@global_config(
    prompt="다국어 리서치 어시스턴트. 정확한 출처와 함께 답변한다.",
    llm_config=LLMConfig(model="claude-sonnet-4-20250514", temperature=0.3),
    tools=[MCPServer("https://search.example.com/mcp")]
)
class ResearchAgent:

    @flow(
        name="research",
        prompt="유저 질문을 분석 → 최적 소스 검색 → 답변 생성",
        input_schema=UserQuery,
        output_schema=FormattedAnswer
    )
    class ResearchFlow:

        # ─── 자식 Flow: 검색 (Flow 안에 Flow) ───
        @flow(
            name="search",
            prompt="분석된 쿼리 기반으로 적절한 소스에서 검색 수행",
            input_schema=AnalyzedQuery,
            output_schema=SearchResult
        )
        class SearchSubFlow:

            @task(name="execute_search", prompt="소스별 검색 실행")
            class ExecuteSearchTask:

                @step(order=1, prompt="검색 쿼리 최적화")
                async def optimize_query(ctx): ...

                @branch(
                    order=2,
                    name="source_select",
                    prompt="쿼리 분석에 따라 검색 소스 선택",
                    condition=BranchCondition(
                        field="source_preference",
                        enum=["web", "db", "api"]
                    ),
                    branches={
                        "web": web_search_handler,
                        "db":  db_search_handler,
                        "api": api_search_handler,
                    },
                    fallback=web_search_handler
                )
                async def route_source(ctx): ...

                @step(order=3, prompt="검색 결과 정제 및 중복 제거")
                async def deduplicate(ctx): ...

        # ─── 메인 Task: 분석 + 포맷 (Task 트리 구조) ───
        @task(name="analyze_and_format",
              prompt="쿼리 분석 및 최종 답변 포맷팅")
        class AnalyzeAndFormatTask:

            # 자식 Task
            @task(name="analyze",
                  prompt="유저 쿼리의 의도와 키워드를 분석",
                  output_schema=AnalyzedQuery)
            class AnalyzeTask:
                @step(order=1, prompt="쿼리 의도 분류")
                async def classify_intent(ctx): ...

            # 자식 Task
            @task(name="format",
                  prompt="검색 결과를 최종 답변으로 포맷팅",
                  output_schema=FormattedAnswer)
            class FormatTask:
                @step(order=1, prompt="답변 초안 생성")
                async def draft_answer(ctx): ...

                @step(order=2, prompt="출처 인용 추가",
                      tool_mode=True)
                async def add_citations(ctx): ...


# ─── Run ───
import asyncio
from flowforge import FlowForge

async def main():
    # Compile + Doc Generation
    engine = FlowForge.compile(ResearchAgent)

    # 생성된 doc 확인
    for node_id, doc in engine.docs.items():
        print(f"[{node_id}] {doc.summary}")

    # Plan + Execute
    result = await engine.run(
        UserQuery(query="2026년 AI 에이전트 프레임워크 트렌드")
    )
    print(result.answer)

    # DAG 시각화
    engine.visualize("research_flow.svg")

asyncio.run(main())
```

---

## 14. CLI Commands

```bash
# DAG 구조 렌더링
flowforge viz ./my_agent.py --output dag.svg

# doc 포함 상세 뷰
flowforge viz ./my_agent.py --show-docs

# 실시간 실행 트레이스
flowforge run ./my_agent.py --trace --port 8080

# 컴파일 검증만 실행
flowforge validate ./my_agent.py

# doc 강제 재생성
flowforge doc-generate ./my_agent.py --force
```

---

## 15. Dynamic Flow Generation

> **Agent가 Agent를 만든다** — `@global_config(dynamic_flow=True)` 활성화 시, FlowForge는 실행 중 기존 DAG에 없는 기능을 LLM으로 코드 생성하여 자동 확장한다.

### 15.1 아키텍처 개요

```
유저 요청
  → AI Planner: 기존 DAG 스캔
  → gap_detected / requirements[].covered=False
  → Engine: _dynamic_generator 메타플로우 트리거 (requirement 당 1회)
  → Meta-flow 3-step:
      [1] analyse_gap     — 기존 flow로 커버 가능한지 확인
      [2] prepare_codegen — downstream contract, 도구 목록 등 brief 조립
      [3] generate_and_inject — LLM codegen → AST 검증 → compile → persist → inject
  → Engine: replan → 새 flow 포함하여 실행
```

### 15.2 DynamicRunOptions

`DynamicRunOptions`는 동적 생성의 모든 동작을 제어한다:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | 동적 생성 on/off |
| `project_root` | `str` | `""` | 프로젝트 루트 경로 |
| `generated_dir` | `str` | `"generated"` | 생성 파일 저장 디렉토리 (project_root 내부) |
| `persist_generated` | `bool` | `False` | manifest.json + 파일 영속화 |
| `auto_load_generated` | `bool` | `False` | 컴파일 시 기존 생성 파일 자동 로드 |
| `include_builtin_tools` | `bool` | `True` | 내장 도구(web, json, files) 활성화 |
| `allow_codegen_tool_use` | `bool` | `False` | 생성 코드 내 tool_use 허용 |
| `allowed_shell_modes` | `list[str]` | `["readonly"]` | 셸 도구 허용 모드 |
| `shell_output_max_chars` | `int` | `4000` | 셸 출력 최대 문자 수 |
| `project_context_max_chars` | `int` | `4000` | 프로젝트 컨텍스트 최대 문자 수 |

### 15.3 AST 안전 검증

생성된 코드는 `exec_module()` 전에 AST 스캔을 통과해야 한다:

- **차단 import**: `subprocess`, `shutil`, `ctypes`, `socket`
- **차단 호출**: `os.system`, `os.popen`, `subprocess.run`, `shutil.rmtree` 등
- **차단 builtin**: `__import__`, `exec`, `eval`, `compile`

`_validate_generated_ast(code)` → `None`이면 안전, 문자열이면 거부 사유.

### 15.4 Manifest 영속화

`persist_generated=True`일 때, 생성된 flow/tool은 파일로 저장되고 `manifest.json`에 등록된다:

```
generated/
├── manifest.json          ← DynamicManifest (flows + tools 레코드)
├── manifest.json.lock     ← fcntl.flock 동시성 보호
├── flows/
│   └── fetch_papers.py    ← 생성된 flow 코드
└── tools/
    └── custom_tool.py     ← 생성된 tool 코드
```

`GeneratedFlowRecord`의 `bridge` 필드:
- `"contract"`: downstream flow의 `input_schema`를 JSON Schema로 추출하여 codegen 프롬프트에 주입
- `"shared_data"`: 공유 데이터를 통한 연결
- `""`: 독립 실행

모든 manifest 읽기/쓰기는 `_manifest_lock` (fcntl.flock) 내에서 수행된다.

### 15.5 Contract-First Flow Chaining

downstream flow가 존재할 때:
1. `downstream_flow_route`로 대상 flow를 지정
2. 해당 flow의 `input_schema`를 JSON Schema로 추출
3. codegen 프롬프트에 "이 스키마와 호환되는 output을 생성하라" 지시
4. 컴파일 후 `check_contract_compatibility()`로 검증

### 15.6 Builtin Tool Pack

`include_builtin_tools=True`일 때 주입되는 도구:

| Tool | Description | 제약 |
|------|-------------|------|
| `pip_install` | Python 패키지 설치 | `DependencyPolicy(allow_install=True)` |
| `python_import_check` | 모듈 import 가능 여부 확인 | 항상 사용 가능 |
| `web_fetch_url` | URL에서 텍스트 가져오기 (urllib) | 항상 사용 가능 |
| `json_select_fields` | JSON에서 특정 필드 추출 | 항상 사용 가능 |
| `files_read_text` | 텍스트 파일 읽기 | project_root 내부만 |
| `files_write_text` | 텍스트 파일 쓰기 | project_root 내부만 |
| `files_list_dir` | 디렉토리 목록 | project_root 내부만 |
| `pdf_read_text` | PDF 텍스트 추출 | `pypdf` 필요, project_root 내부만 |
| `pptx_create` | PPTX 프레젠테이션 생성 | `python-pptx` 필요, project_root 내부만 |
| `csv_read` | CSV 파일 읽기 | 항상 사용 가능, project_root 내부만 |
| `csv_write` | CSV 파일 쓰기 | 항상 사용 가능, project_root 내부만 |
| `docx_create` | Word 문서 생성 | `python-docx` 필요, project_root 내부만 |
| `markdown_write` | Markdown 파일 쓰기 | 항상 사용 가능, project_root 내부만 |
| `chart_create` | 차트 이미지(PNG) 생성 | `matplotlib` 필요, project_root 내부만 |
| `shell_*` | 셸 명령 실행 | `allowed_shell_modes` 에 따름 |

파일 도구는 `_resolve_safe_path()`로 경로를 검증하여 project_root 바깥 접근을 차단한다.
외부 패키지가 필요한 도구(`pypdf`, `python-pptx`, `python-docx`, `matplotlib`)는 패키지 미설치 시 명확한 에러 메시지를 반환하며, `pip_install` 도구로 먼저 설치하도록 안내한다.

### 15.6.1 `ctx.call_tool()` — Step에서 직접 도구 호출

`StepContext.call_tool(tool_name, **kwargs)`로 Step 함수 내에서 등록된 도구를 직접 호출할 수 있다:

```python
@step(order=1, prompt="PPT 생성")
async def render(ctx):
    result = await ctx.call_tool("pptx_create", path="report.pptx", slides=json_str)
```

`call_tool`은 `merged_tools` (global → flow → task → step) 에서 이름이 일치하는 `FunctionTool`을 찾아 실행한다.

### 15.7 Planner Requirement Decomposition

autonomous 모드에서 Planner는 유저 요청을 `requirements[]` 배열로 분해:

```json
{
  "requirements": [
    {"description": "논문 검색", "covered": false, "needs_flow": true,
     "suggested_flow_name": "search_papers", "suggested_flow_prompt": "..."},
    {"description": "PDF 분석", "covered": true, "covered_by": "paper_report_pipeline"}
  ],
  "gap_detected": true
}
```

`covered=false`인 각 requirement마다 `_dynamic_generator`가 한 번씩 실행된다.

### 15.8 사용 예시

```python
from flowforge import FlowForge, global_config, DynamicRunOptions
from flowforge.types import LLMConfig, FunctionTool

@global_config(
    prompt="AI 리서치 에이전트",
    llm_config=LLMConfig.for_claude(model="claude-sonnet-4-6"),
    tools=[my_search_tool],
    dynamic_flow=True,
)
class MyAgent:
    ExistingPipeline = ExistingPipeline  # 기존 flow

options = DynamicRunOptions(
    project_root=".",
    generated_dir="generated",
    persist_generated=True,
    include_builtin_tools=True,
)

engine = FlowForge.compile(MyAgent, dynamic_options=options)
await engine.generate_docs(planning_only=True)

# planner가 gap을 감지하면 자동으로 flow를 생성하고 실행
result = await engine.run(
    "최근 수학 논문을 검색해서 요약해줘",
    planning_mode="autonomous",
    dynamic_options=options,
)
```

---

*End of Specification*