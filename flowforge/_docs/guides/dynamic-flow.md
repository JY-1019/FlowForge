# Dynamic Flow Generation

> Agent가 Agent를 만든다 — 실행 중 기존 DAG에 없는 기능을 LLM이 코드를 생성하여 자동 확장한다.

---

## Overview

`@global_config(dynamic_flow=True)`를 설정하면, FlowForge 엔진은 autonomous/hybrid 모드에서 planner가 "gap"을 감지했을 때 자동으로 새 Flow를 생성한다.

```
유저 요청 → Planner → gap 감지 → _dynamic_generator 메타플로우 실행
                                   → LLM codegen → AST 검증 → compile → inject
                                → 새 DAG으로 replan → 실행
```

---

## Quick Start

```python
from flowforge import global_config, FlowForge, DynamicRunOptions
from flowforge.types import LLMConfig, FunctionTool

# 1. 도구 정의
async def search_papers(query: str) -> dict:
    ...

search_tool = FunctionTool(
    func=search_papers,
    name="search_papers",
    description="Search for academic papers",
)

# 2. Agent 정의 — dynamic_flow=True
@global_config(
    prompt="논문 검색 에이전트",
    llm_config=LLMConfig.for_claude(),
    tools=[search_tool],
    dynamic_flow=True,
)
class MyAgent:
    pass  # flow가 없어도 OK

# 3. DynamicRunOptions 설정
options = DynamicRunOptions(
    project_root=".",
    generated_dir="generated",
    persist_generated=True,
    include_builtin_tools=True,
)

# 4. Compile & Run
engine = FlowForge.compile(MyAgent, dynamic_options=options)
await engine.generate_docs(planning_only=True)

result = await engine.run(
    "최근 AI 논문 3개를 찾아 요약해줘",
    planning_mode="autonomous",
    dynamic_options=options,
)
```

---

## How It Works

### 1. Planner Requirement Decomposition

autonomous 모드에서 `LLMPlanner`는 유저 요청을 `requirements[]` 배열로 분해한다:

```json
{
  "routes": ["paper_report_pipeline"],
  "gap_detected": true,
  "requirements": [
    {
      "description": "논문 검색",
      "covered": false,
      "needs_flow": true,
      "suggested_flow_name": "search_papers",
      "suggested_flow_prompt": "arXiv에서 논문을 검색하고 결과를 반환한다"
    },
    {
      "description": "보고서 생성",
      "covered": true,
      "matched_route": "paper_report_pipeline"
    }
  ]
}
```

`covered=false`인 각 requirement마다 `_dynamic_generator`가 한 번씩 실행된다.

### 2. Meta-Flow 3-Step Pipeline

`_dynamic_generator`는 FlowForge의 자체 데코레이터로 정의된 내부 flow다:

```
@flow "_dynamic_generator"
└─ @task "_generate_and_run"
    ├─ step[1] analyse_gap        — 기존 flow로 커버 가능한지 확인
    ├─ step[2] prepare_codegen    — downstream contract, 도구 목록 등 brief 조립
    └─ step[3] generate_and_inject — LLM codegen → AST 검증 → compile → persist → inject
```

### 3. Code Generation & Safety

`DynamicFlowGenerator`가 LLM에게 FlowForge 데코레이터 코드를 생성하도록 요청한다.
생성된 코드는 실행 전에 반드시 AST 안전 검증을 통과해야 한다:

**차단 대상:**
- Import: `subprocess`, `shutil`, `ctypes`, `socket`
- 호출: `os.system`, `os.popen`, `subprocess.run`, `shutil.rmtree`
- Builtin: `__import__`, `exec`, `eval`, `compile`

컴파일 실패 시 에러를 LLM에 피드백하여 최대 3회 재시도한다.

### 4. Replan & Execute

새 flow가 DAG에 주입된 후, 엔진이 planner를 다시 호출하여 새 flow를 포함한 실행 계획을 수립한다.

---

## Two Usage Patterns

### Pattern 1: Partial Gap — Static Pipeline + Dynamic Upstream

기존 downstream pipeline은 유지하고, 누락된 upstream flow만 동적 생성:

```python
@flow(name="report_pipeline", prompt="보고서 생성 파이프라인")
class ReportPipeline:
    # ... 정적으로 정의된 task들

@global_config(
    prompt="...",
    tools=[search_tool],
    dynamic_flow=True,
)
class MyAgent:
    ReportPipeline = ReportPipeline  # downstream은 이미 있음
```

실행 시 planner가 "검색 flow가 없다"고 판단 → 자동 생성 → `search_flow → report_pipeline` 순서로 실행.

### Pattern 2: Empty Agent — 모든 Flow를 동적 생성

Flow를 하나도 정의하지 않고, 유저 요청에 따라 필요한 flow를 모두 생성:

```python
@global_config(
    prompt="다목적 에이전트",
    tools=[tool_a, tool_b],
    dynamic_flow=True,
)
class EmptyAgent:
    pass  # flow 없음
```

유저가 복합 요청을 하면 planner가 여러 requirement로 분해하고, 각각에 대해 flow를 생성한다.

---

## Contract-First Chaining

downstream flow가 존재할 때, 동적 생성되는 upstream flow의 output이 downstream의 input_schema와 호환되어야 한다.

```
1. planner가 downstream_flow_route 지정
2. 해당 flow의 input_schema를 JSON Schema로 추출
3. codegen 프롬프트: "이 스키마와 호환되는 output을 반환하라"
4. 컴파일 후 check_contract_compatibility()로 검증
```

---

## Manifest Persistence

`persist_generated=True`일 때 생성된 코드는 파일로 저장되고 `manifest.json`에 등록된다:

```
generated/
├── manifest.json          ← flow/tool 레코드
├── manifest.json.lock     ← fcntl.flock 동시성 보호
├── flows/
│   └── search_papers.py
└── tools/
    └── custom_tool.py
```

`auto_load_generated=True`로 컴파일하면 이전에 생성된 flow를 자동으로 로드한다.

---

## Builtin Tools

`include_builtin_tools=True`(기본값)일 때, 생성된 flow가 사용할 수 있는 도구:

| Tool | Description | Gate |
|------|-------------|------|
| `pip_install` | Python 패키지 설치 (`pip install`) | `DependencyPolicy(allow_install=True)` |
| `python_import_check` | Python 모듈 import 가능 여부 확인 | 항상 |
| `web_fetch_url` | URL에서 텍스트 가져오기 | 항상 |
| `json_select_fields` | JSON에서 특정 필드 추출 | 항상 |
| `files_read_text` | 텍스트 파일 읽기 (project_root 내부만) | 항상 |
| `files_write_text` | 텍스트 파일 쓰기 (project_root 내부만) | 항상 |
| `files_list_dir` | 디렉토리 목록 (project_root 내부만) | 항상 |
| `pdf_read_text` | PDF 텍스트 추출 | `pypdf` 필요 |
| `pptx_create` | PPTX 프레젠테이션 생성 | `python-pptx` 필요 |
| `csv_read` | CSV 파일 읽기 | 항상 |
| `csv_write` | CSV 파일 쓰기 | 항상 |
| `docx_create` | Word 문서 생성 | `python-docx` 필요 |
| `markdown_write` | Markdown 파일 쓰기 | 항상 |
| `chart_create` | 차트 이미지(PNG) 생성 | `matplotlib` 필요 |

외부 패키지가 필요한 도구는 미설치 시 안내 메시지와 함께 실패하며, `pip_install` 도구로 먼저 설치할 수 있다.

### Step에서 직접 도구 호출

`ctx.call_tool()`로 Step 함수 내에서 등록된 builtin 도구를 직접 호출할 수 있다:

```python
@step(order=1, prompt="PPT 렌더링")
async def render(ctx):
    result = await ctx.call_tool(
        "pptx_create",
        path="report.pptx",
        slides=json.dumps(slide_data),
    )
```

### pip_install 사용

`pip_install` 도구는 `DependencyPolicy`로 제어된다:

```python
from flowforge.types import DependencyPolicy

options = DynamicRunOptions(
    dependency_policy=DependencyPolicy(
        allow_install=True,              # 설치 허용
        allowed_packages=["httpx"],      # 화이트리스트 (비어있으면 전체 허용)
        denied_packages=["subprocess"],  # 블랙리스트
    ),
)
```

---

## DynamicRunOptions Reference

See [Types → DynamicRunOptions](../api/types.md#dynamicrunoptions) for the full parameter list.

---

## Examples

- `examples/dynamic_paper_report_agent.py` — Static pipeline + dynamic upstream (arXiv)
- `examples/dynamic_bare_agent.py` — Bare agent: zero flows, zero tools, everything generated at runtime
