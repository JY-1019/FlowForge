# Types Reference

All types are importable from `flowforge.types` or `flowforge`.

---

## LLMConfig

```python
from flowforge.types import LLMConfig

config = LLMConfig(
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=4096,
    api_key=None,   # falls back to ANTHROPIC_API_KEY env var
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model identifier |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0–1.0) |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `api_key` | `str \| None` | `None` | API key (uses env var if None) |

---

## BranchCondition

```python
from flowforge.types import BranchCondition

condition = BranchCondition(
    field="format",              # attribute/key to inspect on ctx.input
    enum=["json", "csv", "xml"], # valid values
)
```

| Field | Type | Description |
|-------|------|-------------|
| `field` | `str` | Attribute name (for Pydantic models) or dict key |
| `enum` | `list[str]` | Valid values that trigger branch routing |

---

## MCPServer

```python
from flowforge.types import MCPServer

mcp = MCPServer(
    url="https://api.example.com/mcp",
    name="example_api",
    description="Example MCP server",
)
```

---

## FunctionTool

```python
from flowforge.types import FunctionTool

def my_tool(query: str) -> str:
    return f"Result for: {query}"

tool = FunctionTool(
    name="my_search",
    description="Search for information",
    func=my_tool,
)
```

---

## HTTPTool

```python
from flowforge.types import HTTPTool

http_tool = HTTPTool(
    name="weather_api",
    url="https://api.weather.com/v1/current",
    method="GET",
    description="Get current weather data",
)
```

---

## ToolConfig (Union)

`ToolConfig = MCPServer | FunctionTool | HTTPTool`

Used in `@global_config(tools=[...])`:

```python
@global_config(
    prompt="...",
    tools=[
        MCPServer("https://search.api.com/mcp"),
        FunctionTool(name="compute", func=my_compute_fn, description="..."),
    ]
)
class MyAgent: ...
```

---

## DynamicRunOptions

동적 flow 생성의 전체 동작을 제어한다. `FlowForge.compile()`과 `engine.run()` 모두에 전달 가능.

```python
from flowforge import DynamicRunOptions
from flowforge.types import DependencyPolicy

options = DynamicRunOptions(
    project_root=".",
    generated_dir="generated",
    persist_generated=True,
    auto_load_generated=True,
    include_builtin_tools=True,
    dependency_policy=DependencyPolicy(allow_install=True),
)

engine = FlowForge.compile(MyAgent, dynamic_options=options)
result = await engine.run(input_data, dynamic_options=options)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | 동적 생성 on/off. `False`면 gap 감지 시에도 생성 안 함 |
| `project_root` | `str` | `""` | 프로젝트 루트 경로. 빈 문자열이면 `cwd()` 사용 |
| `generated_dir` | `str` | `"generated"` | 생성 코드 저장 디렉토리. `project_root` 내부여야 함 |
| `persist_generated` | `bool` | `True` | 생성 코드를 파일로 저장 + `manifest.json` 업데이트 |
| `auto_load_generated` | `bool` | `True` | 컴파일 시 `manifest.json`에서 이전 생성 flow 자동 로드 |
| `include_builtin_tools` | `bool` | `True` | 내장 도구 팩 활성화 (web, json, files, document tools) |
| `allow_codegen_tool_use` | `bool` | `False` | 생성 코드 내 `tool_use` (LLM 도구 선택) 허용 |
| `allowed_shell_modes` | `list[str]` | `["readonly"]` | 셸 도구 모드: `"readonly"`, `"readwrite"`, `"none"` |
| `shell_output_max_chars` | `int` | `4000` | 셸 출력 최대 문자 수 |
| `project_context_max_chars` | `int` | `4000` | 코드 생성 프롬프트의 프로젝트 컨텍스트 최대 문자 수 |

**캐싱 동작:**

- `persist_generated=True` + `auto_load_generated=True` (기본값): 생성된 flow가 프로세스 간 자동 재사용
- 같은 세션 내에서도 DAG/manifest 기반 중복 체크로 재생성 방지

자세한 설명은 [Dynamic Flow Generation Guide](../guides/dynamic-flow.md#dynamicrunoptions-상세-설명) 참조.

---

## DependencyPolicy

생성 코드의 패키지 설치를 제어한다. `DynamicRunOptions.dependency_policy`에 전달.

```python
from flowforge.types import DependencyPolicy

policy = DependencyPolicy(
    allow_install=True,                 # pip_install 도구 사용 허용
    allowed_packages=["httpx"],         # 화이트리스트 (비어있으면 전체 허용)
    denied_packages=["subprocess"],     # 블랙리스트 (항상 적용)
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_install` | `bool` | `False` | `pip_install` 도구 사용 허용 여부 |
| `allowed_packages` | `list[str]` | `[]` | 설치 허용 패키지. 빈 리스트면 전체 허용 |
| `denied_packages` | `list[str]` | `[]` | 설치 차단 패키지. 화이트리스트보다 우선 |
