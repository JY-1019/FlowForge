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

핵심 특징:
- **자동 도구 선택** — 생성된 flow가 builtin 도구 중 적합한 것을 자동으로 사용
- **파일 출력 자동 감지** — PPT, CSV, 차트 등 파일 생성 요청 시 적절한 도구를 자동 포함
- **Manifest 캐싱** — 이전에 생성된 flow를 자동 저장/로드하여 재생성 방지
- **Contract-First 체이닝** — downstream flow의 input_schema와 호환되는 output 자동 생성

---

## Quick Start

```python
from flowforge import global_config, FlowForge, DynamicRunOptions
from flowforge.types import LLMConfig

# 1. Agent 정의 — dynamic_flow=True
@global_config(
    prompt="다목적 AI 에이전트",
    llm_config=LLMConfig.for_claude(),
    dynamic_flow=True,
)
class MyAgent:
    pass  # flow가 없어도 OK — 필요한 flow를 자동 생성

# 2. DynamicRunOptions 설정
options = DynamicRunOptions(
    project_root=".",
    generated_dir="generated",
    persist_generated=True,       # 생성 코드를 파일로 저장
    auto_load_generated=True,     # 다음 실행 시 자동 로드
    include_builtin_tools=True,   # 내장 도구 활성화
)

# 3. Compile & Run
engine = FlowForge.compile(MyAgent, dynamic_options=options)
await engine.generate_docs(planning_only=True)

result = await engine.run(
    "세계에서 가장 높은 산 Top 5를 표로 정리해줘",
    planning_mode="autonomous",
    dynamic_options=options,
)
```

---

## DynamicRunOptions 상세 설명

`DynamicRunOptions`는 동적 생성의 모든 동작을 제어한다.
`FlowForge.compile()` 시점과 `engine.run()` 시점 모두에서 전달할 수 있다.

```python
from flowforge import DynamicRunOptions
from flowforge.types import DependencyPolicy

options = DynamicRunOptions(
    # ── 기본 설정 ──
    enabled=True,
    project_root=".",
    generated_dir="flowforge/generated",

    # ── 캐싱 / 영속화 ──
    persist_generated=True,
    auto_load_generated=True,

    # ── 도구 설정 ──
    include_builtin_tools=True,
    allow_tool_generation=False,
    allow_codegen_tool_use=False,

    # ── 셸 실행 제어 ──
    allowed_shell_modes=["readonly", "project_exec"],
    shell_timeout_seconds=60,
    shell_output_max_chars=4000,

    # ── MCP 도구 생성 보조 ──
    mcp_server_commands={},
    mcp_start_timeout_seconds=15,

    # ── 코드 생성 컨텍스트 ──
    project_context_max_chars=4000,
    max_requirements=8,

    # ── 의존성 정책 ──
    dependency_policy=DependencyPolicy(
        allow_install=False,
        allowed_packages=[],
        denied_packages=[],
    ),
)
```

### 전체 파라미터 레퍼런스

#### 기본 설정

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | 동적 생성 기능 on/off. `False`이면 gap이 감지되어도 생성하지 않음 |
| `project_root` | `str \| None` | `None` | 프로젝트 루트 경로. `None`이면 `cwd()` 사용. 파일 도구의 샌드박스 경계 역할 |
| `generated_dir` | `str` | `"flowforge/generated"` | 생성된 코드 저장 디렉토리. `project_root` 내부여야 함 (보안 제한) |

#### 캐싱 / 영속화 (Manifest)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `persist_generated` | `bool` | `True` | 생성된 flow/tool 코드를 `.py` 파일로 저장하고 `manifest.json`에 등록 |
| `auto_load_generated` | `bool` | `True` | `FlowForge.compile()` 시 `manifest.json`에서 이전 생성 flow를 자동 로드 |

**캐싱 동작 원리:**

```
첫 번째 실행:
  compile() → DAG에 flow 없음
  run() → planner가 gap 감지 → flow 생성 → manifest.json에 저장

두 번째 실행:
  compile() → auto_load_generated=True → manifest.json에서 로드 → DAG에 포함
  run() → planner가 해당 flow를 "covered"로 판정 → 재생성 안 함
```

- `persist_generated=True` + `auto_load_generated=True` (기본값): 생성된 flow가 프로세스 재시작 후에도 재사용됨
- 같은 세션 내에서도 DAG에 이미 inject된 flow와 manifest에 기록된 flow는 중복 생성하지 않음
- Flow 이름 기반 중복 체크 (이름이 같으면 기존 것을 사용)

#### 도구 설정

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `include_builtin_tools` | `bool` | `True` | 내장 도구 팩(web, json, files, document) 활성화. [Builtin Tools](#builtin-tools) 참조 |
| `allow_tool_generation` | `bool` | `False` | 필요한 경우 새 `FunctionTool` 코드 생성을 허용 |
| `allow_codegen_tool_use` | `bool` | `False` | 생성된 코드 내에서 `tool_use` (LLM이 도구를 선택하는 방식) 허용 여부 |

#### 셸 실행 제어

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed_shell_modes` | `list[str]` | `["readonly", "project_exec"]` | 셸 도구 허용 모드. `"readonly"`, `"workspace_write"`, `"project_exec"`, `"install_dependency"` |
| `shell_timeout_seconds` | `int` | `60` | 셸 명령 실행 타임아웃 |
| `shell_output_max_chars` | `int` | `4000` | 셸 명령 출력의 최대 문자 수. 초과 시 잘림 |

#### MCP 도구 생성 보조

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcp_server_commands` | `dict[str, list[str]]` | `{}` | 동적 생성 중 사용할 MCP server command map |
| `mcp_start_timeout_seconds` | `int` | `15` | MCP server 시작 대기 시간 |

#### 코드 생성 컨텍스트

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project_context_max_chars` | `int` | `4000` | 코드 생성 프롬프트에 포함되는 프로젝트 컨텍스트의 최대 문자 수 |
| `max_requirements` | `int` | `8` | planner gap requirement 최대 개수 |

#### 의존성 정책 (DependencyPolicy)

```python
from flowforge.types import DependencyPolicy

options = DynamicRunOptions(
    dependency_policy=DependencyPolicy(
        allow_install=True,              # pip install 도구 사용 허용
        allowed_packages=["httpx"],      # 화이트리스트 (비어있으면 전체 허용)
        denied_packages=["subprocess"],  # 블랙리스트 (항상 적용)
    ),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_install` | `bool` | `False` | `pip_install` 도구 사용 허용 여부 |
| `allowed_packages` | `list[str]` | `[]` | 설치 허용 패키지 화이트리스트. 빈 리스트면 전체 허용 |
| `denied_packages` | `list[str]` | `[]` | 설치 차단 패키지 블랙리스트. 화이트리스트보다 우선 |

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
단, 같은 이름의 flow가 DAG에 이미 존재하거나 manifest에 기록되어 있으면 재생성을 건너뛴다.

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

### 4. 자동 도구 선택

생성된 flow는 두 가지 방식으로 도구를 사용할 수 있다:

**Method A: `ctx.call_tool()` — Step에서 직접 호출**
```python
@step(order=2, prompt="PPT 렌더링")
async def render(ctx):
    result = await ctx.call_tool("pptx_create", path="report.pptx", slides=json_str)
```

**Method B: `ctx.call_llm("<tool_name>")` — LLM이 판단하여 호출**
```python
@step(order=1, prompt="데이터 분석 <json_select_fields>")
async def analyze(ctx):
    return await ctx.call_llm("데이터에서 핵심 필드를 추출해줘")
```

코드 생성 시 AI가 사용 가능한 도구 목록(파라미터 스키마 포함)을 보고 적절한 도구와 호출 방식을 자동으로 선택한다.

### 5. 파일 출력 자동 감지 (Output Artifact Detection)

유저 요청에 파일 생성 키워드가 포함되면, 적절한 도구 호출 단계가 자동으로 추가된다:

| 키워드 | 도구 | 확장자 |
|--------|------|--------|
| ppt, pptx, 프레젠테이션, 슬라이드, 발표 자료 | `pptx_create` | `.pptx` |
| csv, 스프레드시트 | `csv_write` | `.csv` |
| docx, 워드, word | `docx_create` | `.docx` |
| 차트, chart, 그래프, graph | `chart_create` | `.png` |
| 마크다운, markdown | `markdown_write` | `.md` |

예: "논문 요약을 PPT로 만들어줘" → 생성되는 flow에 자동으로 `pptx_create` 호출 step이 포함됨.

### 6. Replan & Execute

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
    dynamic_flow=True,
)
class EmptyAgent:
    pass  # flow 없음 — 모든 것을 런타임에 생성
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

## Manifest 영속화 & 캐싱

### 디렉토리 구조

`persist_generated=True`일 때 생성된 코드는 파일로 저장되고 `manifest.json`에 등록된다:

```
generated/
├── manifest.json          ← flow/tool 레코드 (DynamicManifest)
├── manifest.json.lock     ← fcntl.flock 동시성 보호
├── flows/
│   └── search_papers.py   ← 생성된 flow 코드
└── tools/
    └── custom_tool.py     ← 생성된 tool 코드
```

### manifest.json 예시

```json
{
  "version": 1,
  "flows": [
    {
      "name": "top_mountains_table",
      "file": "generated/flows/top_mountains_table.py",
      "class_name": "TopMountainsTableFlow",
      "downstream_flow_route": "",
      "bridge": "",
      "created_at": "2026-04-26T10:30:00+00:00"
    }
  ],
  "tools": []
}
```

### 캐싱 라이프사이클

`manifest.json`은 생성된 flow/tool의 인덱스다. FlowForge는 이를 통해
프로세스 재시작 후에도 이미 생성한 코드를 다시 로드하고, 같은 flow를
불필요하게 다시 만들지 않는다.

```
                                    ┌──────────────────────┐
                                    │   manifest.json      │
                                    │   + generated/*.py   │
                                    └──────┬───────────────┘
                                           │
  Process 1                                │           Process 2
  ─────────                                │           ─────────
  compile()                                │           compile()
    → DAG 비어있음                          │             → auto_load_generated=True
  run()                                    │             → manifest에서 flow 로드
    → planner: gap 감지                    │             → DAG에 포함
    → flow 생성                            │           run()
    → persist_generated=True ──────────────┘             → planner: covered=true
    → DAG에 inject                                       → 재생성 없이 바로 실행
    → 실행
```

**같은 프로세스 내 재실행:**

```python
# 첫 번째 실행 — flow 생성됨
result1 = await engine.run("산 높이 Top 5", planning_mode="autonomous")

# 두 번째 실행 — 같은 요청 (flow가 이미 DAG에 있으므로 재생성 안 함)
result2 = await engine.run("산 높이 Top 5", planning_mode="autonomous")
```

엔진은 flow 생성 전에 다음을 체크한다:
1. DAG에 같은 이름의 flow가 이미 있는지
2. manifest에 같은 이름의 flow가 기록되어 있는지

둘 중 하나라도 해당하면 생성을 건너뛴다.

이 체크는 dynamic generator가 같은 요청을 여러 번 받거나, 이전 실행에서
생성된 flow가 이미 `auto_load_generated=True`로 로드된 경우 중복 생성을
막기 위한 캐시 계층이다.

### 옵션별 동작

| Option | Behavior |
|--------|----------|
| `persist_generated=True` | 생성 코드를 파일로 저장하고 `manifest.json`에 기록 |
| `persist_generated=False` | 현재 프로세스의 DAG에만 주입. 재시작 후 사라짐 |
| `auto_load_generated=True` | `compile()` 시 manifest에 있는 flow/tool을 자동 import 후 DAG에 포함 |
| `auto_load_generated=False` | 이전 생성물은 파일로 남아도 자동 로드하지 않음 |

---

## Builtin Tools

`include_builtin_tools=True`(기본값)일 때, 생성된 flow가 사용할 수 있는 도구:

### 유틸리티 도구

| Tool | Description | Gate |
|------|-------------|------|
| `pip_install` | Python 패키지 설치 (`pip install`) | `DependencyPolicy(allow_install=True)` |
| `python_import_check` | Python 모듈 import 가능 여부 확인 | 항상 |
| `web_fetch_url` | URL에서 텍스트 가져오기 | 항상 |
| `json_select_fields` | JSON에서 특정 필드 추출 | 항상 |

### 파일 도구

| Tool | Description | Gate |
|------|-------------|------|
| `files_read_text` | 텍스트 파일 읽기 | project_root 내부만 |
| `files_write_text` | 텍스트 파일 쓰기 | project_root 내부만 |
| `files_list_dir` | 디렉토리 목록 | project_root 내부만 |

### 문서 생성 도구

| Tool | Description | Gate |
|------|-------------|------|
| `pdf_read_text` | PDF 텍스트 추출 (페이지 범위 지정 가능) | `pypdf` 필요 |
| `pptx_create` | PPTX 프레젠테이션 생성 (6종 레이아웃, 테마, 표, 발표 노트) | `python-pptx` 필요 |
| `csv_read` | CSV 파일 읽기 (dict 리스트 반환) | 항상 |
| `csv_write` | CSV 파일 쓰기 | 항상 |
| `docx_create` | Word 문서 생성 | `python-docx` 필요 |
| `markdown_write` | Markdown 파일 쓰기 | 항상 |
| `chart_create` | 차트 이미지(PNG) 생성 (bar, line, pie, scatter) | `matplotlib` 필요 |

외부 패키지가 필요한 도구는 미설치 시 안내 메시지와 함께 실패하며, `pip_install` 도구로 먼저 설치할 수 있다.

모든 파일 도구는 `_resolve_safe_path()`를 통해 `project_root` 바깥 경로 접근을 차단한다.

### Step에서 직접 도구 호출 (`ctx.call_tool`)

`StepContext.call_tool()`로 Step 함수 내에서 등록된 도구를 직접 호출할 수 있다:

```python
@step(order=1, prompt="PPT 렌더링")
async def render(ctx):
    slides = json.dumps([
        {"layout": "cover", "title": "보고서", "subtitle": "2026"},
        {"layout": "content", "title": "요약", "body": "핵심 내용..."},
        {"layout": "table", "title": "데이터", "table": {"headers": [...], "rows": [...]}},
    ])
    result = await ctx.call_tool("pptx_create", path="report.pptx", slides=slides)
```

`call_tool`은 전역 → flow → task → step 순으로 병합된 도구 목록에서 이름이 일치하는 `FunctionTool`을 찾아 실행한다. 도구가 없으면 `ValueError`를 발생시킨다.

### pptx_create 레이아웃

| Layout | 필수 필드 | 선택 필드 |
|--------|----------|----------|
| `cover` | `title` | `subtitle` |
| `content` | `title`, `body` | `notes` |
| `section` | `title` | `subtitle` |
| `comparison` | `title`, `left_title`, `left_body`, `right_title`, `right_body` | `notes` |
| `table` | `title`, `table.headers`, `table.rows` | `notes` |
| `blank` | | |

---

## LLM 응답 자동 정리

`ctx.call_llm()`의 응답에 마크다운 코드 펜스(`` ```json ... ``` ``)가 포함된 경우, 프레임워크가 자동으로 제거하고 JSON 파싱을 시도한다. 이를 통해 생성된 flow에서 `json.loads()`가 안정적으로 동작한다.

---

## Examples

- `examples/dynamic_paper_report_agent.py` — Static pipeline + dynamic upstream (arXiv 논문 검색)
- `examples/dynamic_bare_agent.py` — Bare agent: zero flows, zero tools, everything generated at runtime
- `examples/dynamic_empty_arxiv_math_agent.py` — 빈 agent + arXiv 수학 논문 검색
