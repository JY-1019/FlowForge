# FlowForge — Input / Output Flow

이 문서는 `@global_config → @flow → @task → @step / @branch` 각 어노테이션 경계를 데이터가 어떻게 통과하는지, 코드 레벨에서 정확히 무슨 일이 일어나는지를 설명한다.

---

## 1. 전체 데이터 흐름 요약

```
engine.run(UserInput)
      │
      ▼
FlowRunner.run(FlowMeta, global_ctx, input=UserInput)
      │  flow_input 이 current_output 으로 설정
      │
      ├─► [child flows 가 있으면] 순차 또는 병렬로 각 child flow 실행
      │        이전 child flow 의 output → 다음 child flow 의 input
      │
      └─► [tasks 가 있으면] 순차 실행
               이전 task 의 output → 다음 task 의 input
                     │
                     ▼
              TaskRunner.run(TaskMeta, flow_ctx, input=current_output)
                     │
                     ├─► [is_leaf=True] 순차 step/branch 체인
                     │        order 1 output → order 2 input → order 3 input → …
                     │
                     └─► [is_leaf=False, container] 순차 child task 체인
                              child task 1 output → child task 2 input → …
```

---

## 2. 어노테이션별 I/O 계약

### 2.1 `@global_config` — 입력/출력 없음

`@global_config` 자체는 데이터를 받거나 내보내지 않는다.
`engine.run(input)` 으로 들어온 값이 **첫 번째 root flow** 의 `flow_input` 으로 그대로 전달된다.

```python
@global_config(prompt="...")
class MyAgent:
    @flow(name="first_flow", ...)
    class FirstFlow: ...
```

```
engine.run("hello")
  └─► FlowRunner.run(FirstFlow_meta, global_ctx, flow_input="hello")
```

---

### 2.2 `@flow` — flow_input → flow_output

Flow 는 하나의 **변환 단위**다. 들어온 값을 처리하고 하나의 값을 돌려준다.

#### I/O 결정 규칙

| 상황 | 이 flow 의 input | 이 flow 의 output |
|------|-----------------|-----------------|
| 루트 flow (global 의 직접 자식) | `engine.run()` 에 넘긴 값 | 마지막 child / task 의 output |
| child flow (다른 flow 안에 중첩) | 부모 flow 의 input 또는 직전 형제 flow 의 output | 마지막 child / task 의 output |
| `parallel=True` 인 부모의 child | 부모 flow 의 input | 결과 list 의 마지막 원소 |

#### 코드 추적

```python
# execution/runner.py — FlowRunner._run_once()

current_output = flow_input          # ← 이 flow 의 입력

# 1. child flows 직렬 실행 (parallel=False)
for child_flow in meta.child_flows:
    current_output = await self.run(child_flow, global_ctx, current_output)
    #                                                        ↑ 이전 child 출력이 다음 child 입력

# 2. tasks 실행
for task_meta in meta.tasks:
    current_output = await self._task_runner.run(task_meta, flow_ctx, current_output)
    #                                                                   ↑ 이전 task 출력이 다음 task 입력

return current_output                # ← 이 flow 의 출력
```

#### `input_schema` / `output_schema` 의 역할

`@flow` 의 스키마는 현재 **문서화 목적** (doc 생성 시 LLM에 전달) 과 **컴파일 검증 힌트** 로 사용된다. 런타임에 flow 경계에서 자동 validate 가 걸리지는 않는다 — Pydantic 검증은 step/branch 단위에서만 일어난다.

---

### 2.3 `@task` — task_input → task_output

Task 는 flow 안의 실행 단위로, 두 가지 형태가 있다.

#### Container Task (child_tasks 보유)

```
task_input
  └─► child task 1 (input=task_input)
        └─► output → child task 2 (input=output)
              └─► output → child task 3 (input=output)
                    └─► output = task_output
```

#### Leaf Task (steps/branches 보유)

```
task_input
  └─► step/branch order=1 (input=task_input)
        └─► output → step/branch order=2 (input=output)
              └─► output → step/branch order=3 (input=output)
                    └─► output = task_output
```

#### 코드 추적

```python
# execution/runner.py — TaskRunner._run_leaf_task()

sorted_steps = sorted(meta.steps, key=lambda s: s.order)
current_input = task_input           # ← task 의 입력 = 첫 번째 step 의 입력

for node_meta in sorted_steps:
    if isinstance(node_meta, StepMeta):
        current_input = await self._step_runner.run(node_meta, task_ctx, current_input)
    elif isinstance(node_meta, BranchMeta):
        current_input = await self._branch_runner.run(node_meta, task_ctx, current_input)
    # 각 step/branch 의 반환값이 다음 step/branch 의 input 이 된다

return current_input                 # ← task 의 출력 = 마지막 step 의 출력
```

---

### 2.4 `@step` — ctx.input → return value

Step 은 리프 실행 단위다. `ctx.input` 으로 이전 노드의 출력을 받고, 함수의 반환값을 다음 노드에 전달한다.

#### ctx 안에 있는 것들

```python
class StepContext:
    input: Any                          # 이전 step/branch 의 출력 (또는 task_input)
    step_prompt: str                    # 이 step 의 prompt
    order: int                          # 이 step 의 order 번호
    previous_results: dict[int, Any]    # 이 task 에서 지금까지 완료된 모든 step 결과
    tools: ToolRegistry                 # 등록된 도구 목록
    llm_config: LLMConfig               # LLM 설정
    # 상위 컨텍스트 접근
    task_ctx: TaskContext
    flow_ctx: FlowContext
    global_ctx: GlobalContext
```

#### 실행 순서

```python
# execution/runner.py — StepRunner.run()

# 1. input_schema 가 선언되어 있으면 먼저 Pydantic 으로 validate
if meta.input_schema is not None and step_input is not None:
    step_input = _to_validated(meta.input_schema, step_input)

# 2. StepContext 생성 — ctx.input = 검증된 step_input
ctx = StepContext(task_ctx=..., step_input=step_input, ...)

# 3. 함수 실행
result = await meta.func(ctx)

# 4. output_schema 가 선언되어 있으면 Pydantic 으로 validate
if meta.output_schema is not None and result is not None:
    result = _to_validated(meta.output_schema, result)

# 5. task_ctx 에 결과 기록 (이후 step 에서 previous_results 로 접근 가능)
task_ctx.step_results[meta.order] = result

return result   # ← 다음 step 의 input 이 된다
```

#### 스키마 검증 타이밍

```
step_input 진입
    │
    ▼  [input_schema 선언 시]
Pydantic model_validate()   ← 실패 시 ExecutionError (input validation failed)
    │
    ▼
async def my_step(ctx):     ← ctx.input = 검증된 값
    return SomeModel(...)
    │
    ▼  [output_schema 선언 시]
Pydantic model_validate()   ← 실패 시 ExecutionError
    │
    ▼
다음 노드의 input
```

---

### 2.5 `@branch` — ctx.input → selected handler → return value

Branch 는 조건에 따라 여러 핸들러 중 하나를 선택해 실행한다.

#### 분기 선택 로직

```python
# execution/runner.py — BranchRunner.run()

# 1. input_schema validate (step 과 동일)

# 2. condition.field 로 분기 값 추출
value = getattr(branch_input, condition.field, None)
#  예: branch_input.source_preference == "web"  →  value = "web"

# 3. 핸들러 선택
if value is not None and str(value) in meta.branches:
    handler = meta.branches[str(value)]    # 정확히 매칭
elif meta.fallback:
    handler = meta.fallback                # 매칭 없으면 fallback
else:
    handler = meta.func                    # fallback 도 없으면 branch 함수 자체 실행

# 4. 선택된 핸들러 실행
result = await handler(ctx)

# 5. output_schema validate
# 6. task_ctx.step_results 에 기록
```

#### `BranchContext` 안에 있는 것들

```python
class BranchContext:
    input: Any                   # 이전 step 의 출력
    condition_value: Any         # condition.field 에서 추출한 실제 값
    selected_branch: str         # 선택된 분기 키 (또는 "__fallback__")
    branch_prompt: str
    previous_results: dict[int, Any]
    task_ctx: TaskContext
    # ... 상위 컨텍스트 접근 동일
```

#### 중요 제약: 모든 핸들러는 같은 타입을 반환해야 한다

`@branch` 데코레이터 적용 시 `validate_branch_output_consistency()` 가 바로 실행되어, 타입 힌트가 있는 핸들러들이 서로 다른 반환 타입을 가지면 즉시 `BranchOutputMismatchError` 가 발생한다. 이는 다음 step 의 input 타입을 일관되게 보장하기 위해서다.

---

## 3. 레이어 간 데이터 흐름 다이어그램

```
engine.run(UserQuery)
│
│   UserQuery ──────────────────────────────────────────────────────────► flow_input
│                                                                                │
│                       ┌─────────────────────────────────────────┐             │
│                       │  FlowRunner: "research" flow             │◄────────────┘
│                       │                                         │
│                       │  flow_input ──► child flow "search"     │
│                       │                      │                  │
│                       │              SearchSubFlow output        │
│                       │                      │                  │
│                       │               ──► task "analyze_and_format"
│                       │                      │                  │
│                       │               task output = flow output │
│                       └─────────────────────────────────────────┘
│                                       │
│                                       │
│          ┌────────────────────────────▼──────────────────────────┐
│          │  TaskRunner: "analyze_and_format" (container task)     │
│          │                                                        │
│          │  task_input ──► child task "analyze"                  │
│          │                       │ AnalyzedQuery                 │
│          │                 ──► child task "format"               │
│          │                       │ FormattedAnswer               │
│          │                 = task output                         │
│          └────────────────────────────────────────────────────────┘
│                                   │
│          ┌────────────────────────▼──────────────────────────────┐
│          │  TaskRunner: "analyze" (leaf task)                     │
│          │                                                        │
│          │  task_input                                            │
│          │     │                                                  │
│          │     ▼  order=1                                         │
│          │  StepRunner: "classify_intent"                         │
│          │     ctx.input = task_input                             │
│          │     return AnalyzedQuery(...)                          │
│          │     │                                                  │
│          │     ▼  = task output = AnalyzedQuery                  │
│          └────────────────────────────────────────────────────────┘
│
│          ┌───────────────────────────────────────────────────────┐
│          │  TaskRunner: "execute_search" (leaf task)             │
│          │                                                        │
│          │  task_input (AnalyzedQuery)                           │
│          │     │                                                  │
│          │     ▼  order=1                                         │
│          │  StepRunner: "optimize_query"                          │
│          │     ctx.input = AnalyzedQuery                         │
│          │     return AnalyzedQuery  (pass-through or modified)   │
│          │     │                                                  │
│          │     ▼  order=2                                         │
│          │  BranchRunner: "source_select"                         │
│          │     ctx.input = AnalyzedQuery                         │
│          │     condition.field = "source_preference"              │
│          │     value = "web"  →  web_search_handler()            │
│          │     return SearchResult(source="web")                  │
│          │     │                                                  │
│          │     ▼  order=3                                         │
│          │  StepRunner: "deduplicate"                             │
│          │     ctx.input = SearchResult                           │
│          │     return SearchResult  (deduplicated)                │
│          │     │                                                  │
│          │     ▼  = task output = SearchResult                   │
│          └───────────────────────────────────────────────────────┘
```

---

## 4. 완전한 동작 예시 — Research Agent

아래 예시는 `examples/research_agent.py` 를 기준으로 각 단계에서 실제로 어떤 값이 흐르는지를 추적한다.

### 코드

```python
from pydantic import BaseModel
from flowforge import global_config, flow, task, step, branch, FlowForge, BranchCondition

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


async def web_search_handler(ctx):
    return SearchResult(results=[{"title": "Web result"}], source="web")

async def db_search_handler(ctx):
    return SearchResult(results=[{"title": "DB result"}], source="db")

async def api_search_handler(ctx):
    return SearchResult(results=[{"title": "API result"}], source="api")


@global_config(prompt="다국어 리서치 어시스턴트.")
class ResearchAgent:

    @flow(name="research", prompt="질문 분석 → 검색 → 답변")
    class ResearchFlow:

        @flow(name="search", prompt="적절한 소스에서 검색")
        class SearchSubFlow:

            @task(name="execute_search", prompt="소스별 검색 실행")
            class ExecuteSearchTask:

                @step(order=1, prompt="검색 쿼리 최적화")
                async def optimize_query(ctx):
                    # ctx.input = 이 task 에 들어온 값 (= ResearchFlow 의 flow_input)
                    # 여기서는 UserQuery 가 그대로 넘어온다
                    return ctx.input   # pass-through

                @branch(
                    order=2,
                    name="source_select",
                    prompt="검색 소스 선택",
                    condition=BranchCondition(field="source_preference", enum=["web","db","api"]),
                    branches={"web": web_search_handler, "db": db_search_handler, "api": api_search_handler},
                    fallback=web_search_handler,
                )
                async def route_source(ctx):
                    ...  # branches dict 로 자동 라우팅됨

                @step(order=3, prompt="중복 제거")
                async def deduplicate(ctx):
                    # ctx.input = order=2 branch 가 반환한 SearchResult
                    return ctx.input   # pass-through

        @task(name="analyze_and_format", prompt="분석 및 포맷팅")
        class AnalyzeAndFormatTask:

            @task(name="analyze", prompt="의도 분석")
            class AnalyzeTask:
                @step(order=1, prompt="의도 분류")
                async def classify_intent(ctx):
                    # ctx.input = AnalyzeAndFormatTask 의 task_input
                    return AnalyzedQuery(
                        intent="information_search",
                        keywords=["AI", "agent", "2026"],
                        source_preference="web",
                    )

            @task(name="format", prompt="답변 포맷팅")
            class FormatTask:
                @step(order=1, prompt="초안 작성")
                async def draft_answer(ctx):
                    # ctx.input = AnalyzeTask 의 output (AnalyzedQuery)
                    return FormattedAnswer(
                        answer=f"키워드: {ctx.input.keywords}",
                        citations=[],
                    )

                @step(order=2, prompt="출처 추가")
                async def add_citations(ctx):
                    # ctx.input = order=1 step 의 output (FormattedAnswer)
                    return FormattedAnswer(
                        answer=ctx.input.answer,
                        citations=["https://example.com"],
                    )
```

### 실행 추적

```python
import asyncio

async def trace():
    engine = FlowForge.compile(ResearchAgent)
    result = await engine.run(UserQuery(query="2026 AI 트렌드"))
    # result == FormattedAnswer(answer="...", citations=["..."])
```

```
engine.run( UserQuery(query="2026 AI 트렌드") )
│
│  ┌─ FlowRunner: "research" ──────────────────────────────────────────────────
│  │  flow_input = UserQuery(query="2026 AI 트렌드")
│  │
│  │  ┌─ FlowRunner: "search" (child flow) ─────────────────────────────────
│  │  │  flow_input = UserQuery(query="2026 AI 트렌드")   ← 부모 input 그대로
│  │  │
│  │  │  ┌─ TaskRunner: "execute_search" (leaf) ───────────────────────────
│  │  │  │  task_input = UserQuery(query="2026 AI 트렌드")
│  │  │  │
│  │  │  │  order=1  StepRunner: optimize_query
│  │  │  │    ctx.input  = UserQuery(query="2026 AI 트렌드")
│  │  │  │    return     = UserQuery(query="2026 AI 트렌드")  [pass-through]
│  │  │  │
│  │  │  │  order=2  BranchRunner: source_select
│  │  │  │    ctx.input         = UserQuery(...)
│  │  │  │    condition.field   = "source_preference"
│  │  │  │    value             = None  (UserQuery 에는 이 필드 없음)
│  │  │  │    → fallback 선택 = web_search_handler
│  │  │  │    return           = SearchResult(results=[...], source="web")
│  │  │  │
│  │  │  │  order=3  StepRunner: deduplicate
│  │  │  │    ctx.input  = SearchResult(results=[...], source="web")
│  │  │  │    return     = SearchResult(...)  [pass-through]
│  │  │  │
│  │  │  │  task output = SearchResult(results=[...], source="web")
│  │  │  └──────────────────────────────────────────────────────────────────
│  │  │
│  │  │  flow output = SearchResult(...)    ← "search" flow 의 output
│  │  └──────────────────────────────────────────────────────────────────────
│  │
│  │  current_output = SearchResult(...)   ← child flow 실행 후
│  │
│  │  ┌─ TaskRunner: "analyze_and_format" (container) ───────────────────────
│  │  │  task_input = SearchResult(...)   ← "search" flow output 이 바로 넘어옴
│  │  │
│  │  │  ┌─ TaskRunner: "analyze" (leaf) ──────────────────────────────────
│  │  │  │  task_input = SearchResult(...)
│  │  │  │
│  │  │  │  order=1  StepRunner: classify_intent
│  │  │  │    ctx.input  = SearchResult(...)
│  │  │  │    return     = AnalyzedQuery(intent="information_search",
│  │  │  │                              keywords=["AI","agent","2026"],
│  │  │  │                              source_preference="web")
│  │  │  │
│  │  │  │  task output = AnalyzedQuery(...)
│  │  │  └──────────────────────────────────────────────────────────────────
│  │  │
│  │  │  ┌─ TaskRunner: "format" (leaf) ───────────────────────────────────
│  │  │  │  task_input = AnalyzedQuery(...)   ← "analyze" task output
│  │  │  │
│  │  │  │  order=1  StepRunner: draft_answer
│  │  │  │    ctx.input  = AnalyzedQuery(...)
│  │  │  │    return     = FormattedAnswer(answer="키워드: ['AI','agent','2026']",
│  │  │  │                                citations=[])
│  │  │  │
│  │  │  │  order=2  StepRunner: add_citations
│  │  │  │    ctx.input  = FormattedAnswer(answer="...", citations=[])
│  │  │  │    return     = FormattedAnswer(answer="...", citations=["https://example.com"])
│  │  │  │
│  │  │  │  task output = FormattedAnswer(answer="...", citations=["https://example.com"])
│  │  │  └──────────────────────────────────────────────────────────────────
│  │  │
│  │  │  task output = FormattedAnswer(...)   ← "format" (마지막 child) output
│  │  └──────────────────────────────────────────────────────────────────────
│  │
│  │  flow output = FormattedAnswer(...)
│  └──────────────────────────────────────────────────────────────────────────
│
└─ engine.run() return = FormattedAnswer(answer="키워드: ['AI','agent','2026']",
                                         citations=["https://example.com"])
```

---

## 5. ctx.previous_results — 이전 step 결과 접근

같은 task 안의 이전 step 결과에 접근해야 할 때 `ctx.previous_results` 를 쓴다.

```python
@task(name="multi_step", prompt="여러 단계 처리")
class MultiStepTask:

    @step(order=1, prompt="데이터 수집")
    async def collect(ctx):
        return {"raw": "some data"}

    @step(order=2, prompt="데이터 검증")
    async def validate_data(ctx):
        # ctx.input = order=1 의 return (직전 step 결과)
        return {"validated": ctx.input["raw"]}

    @step(order=3, prompt="최종 처리")
    async def finalize(ctx):
        # ctx.input    = order=2 의 return (직전 step 결과)
        # ctx.previous_results = {1: {"raw": "some data"}, 2: {"validated": "some data"}}
        raw = ctx.previous_results[1]          # order=1 결과
        validated = ctx.previous_results[2]    # order=2 결과
        return {"final": validated["validated"], "original": raw["raw"]}
```

---

## 6. Pydantic 스키마 선언이 있을 때 vs 없을 때

### 스키마 없음 (기본값)

```python
@step(order=1, prompt="변환")
async def transform(ctx):
    # ctx.input = 이전 노드가 반환한 값 그대로
    # 타입 검증 없음 — 어떤 타입이든 통과
    return {"processed": True}
```

### input_schema 만 선언

```python
@step(order=2, prompt="처리", input_schema=MyInput)
async def process(ctx):
    # StepRunner 가 실행 전에 ctx.input 을 MyInput.model_validate() 로 검증
    # 실패 시 ExecutionError(input validation failed against MyInput: ...)
    val: MyInput = ctx.input   # 안전하게 타입 단언 가능
    return val.field * 2
```

### output_schema 만 선언

```python
@step(order=3, prompt="포맷", output_schema=MyOutput)
async def format_output(ctx):
    result = {"value": 42}
    # StepRunner 가 반환 후 MyOutput.model_validate(result) 실행
    # 실패 시 ExecutionError
    return result   # dict 도 OK — model_validate 가 처리
```

### 둘 다 선언 (권장)

```python
@step(order=1, prompt="변환", input_schema=RawData, output_schema=CleanData)
async def clean(ctx):
    data: RawData = ctx.input   # 이미 검증됨
    return CleanData(value=data.raw.strip())
```

---

## 7. Branch 핸들러 내부에서의 접근

```python
async def web_search_handler(ctx: BranchContext):
    # ctx.input           = 이전 step 의 output (BranchCondition 이 읽은 객체)
    # ctx.condition_value = BranchCondition.field 에서 추출된 값 ("web")
    # ctx.selected_branch = "web"
    # ctx.task_ctx        = 상위 TaskContext 접근
    # ctx.tools           = 등록된 도구 목록

    query = ctx.input.keywords if hasattr(ctx.input, "keywords") else str(ctx.input)
    # ... 실제 검색 로직
    return SearchResult(results=[...], source="web")
```

---

## 8. 병렬 Flow — I/O 처리 방식

`parallel=True` 인 flow 는 모든 child flow 가 **동일한 input** 을 받고 동시에 실행된다.
각 child 의 output 은 수집되며, **마지막 child 의 output** 이 다음 단계로 전달된다.

```python
@flow(name="parallel_analysis", prompt="병렬 분석", parallel=True)
class ParallelAnalysisFlow:

    @flow(name="sentiment", prompt="감성 분석")
    class SentimentFlow:   # input: UserQuery, output: SentimentResult
        ...

    @flow(name="summary", prompt="요약")
    class SummaryFlow:     # input: UserQuery, output: SummaryResult
        ...
```

```
flow_input = UserQuery(...)

병렬 실행:
  SentimentFlow(input=UserQuery) → SentimentResult
  SummaryFlow(input=UserQuery)   → SummaryResult

current_output = SummaryResult   ← results[-1]  (마지막 child 의 output)
```

> 여러 병렬 결과를 모두 사용하려면 마지막 flow 에서 이전 결과를 취합하는 step 을 추가하거나, container task 에서 `ctx.task_ctx.flow_ctx.flow_state` 를 공유 스토리지로 활용한다.

---

## 9. 요약 테이블

| 경계 | 무엇이 input | 무엇이 output | 검증 |
|------|-------------|---------------|------|
| `engine.run(x)` → root flow | `x` | 마지막 root flow output | 없음 |
| flow → child flow | 부모 flow input 또는 이전 형제 output | child 의 마지막 task/flow output | 없음 (문서화만) |
| flow → task | flow 의 current_output | task output | 없음 |
| container task → child task | 이전 child task output | 마지막 child task output | 없음 |
| leaf task → step[1] | task_input | step return value | input_schema 선언 시 Pydantic |
| step[n] → step[n+1] | step[n] return value | step[n+1] return value | 양쪽 스키마 선언 시 컴파일+런타임 검증 |
| step[n] → branch[n+1] | step[n] return value | selected handler return | 동일 |
| branch → step[n+1] | selected handler return | step return | output_schema 선언 시 Pydantic |
