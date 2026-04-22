"""대규모 엔터프라이즈 에이전트 — 복잡한 Flow/Task 구조 + Autonomous 모드 테스트.

=======================================================================
목적
=======================================================================

  30개 이상의 Flow (최대 5단계 깊이), 복잡한 Task 구조를 가진 에이전트에서
  FlowForge의 `planning_mode="autonomous"` 가 유저의 자연어 요청을 보고
  적절한 실행 경로(서브트리)를 스스로 찾아내는지 검증한다.

  핵심: 이 예제는 LLM을 예제 스크립트 레벨에서 직접 호출하지 않는다.
        대신 FlowForge 내장 LLMPlanner에 모든 결정을 맡긴다:

            1. FlowForge.compile(agent)          # DAG 컴파일
            2. await engine.generate_docs()      # 노드별 doc 생성(LLM)
            3. await engine.run_traced(
                   user_request,
                   planning_mode="autonomous",   # ← 내장 Planner가 경로 결정
               )

  실행 결과를 Mermaid 다이어그램으로 시각화하여 Planner가 어떤 노드를 선택했고
  어떤 노드를 건너뛰었는지 한눈에 확인할 수 있다.

=======================================================================
에이전트 구조 (5단계 깊이, 30+ Flow)
=======================================================================

  @global_config (EnterprisePlatformAgent)
  │
  ├─ @flow "data_platform"                          ← L1: 데이터 플랫폼
  │   ├─ @flow "ingestion"                          ← L2: 데이터 수집
  │   │   ├─ @flow "batch_ingestion"                ← L3: 배치 수집
  │   │   │   ├─ @flow "file_ingestion"             ← L4: 파일 수집
  │   │   │   │   └─ @flow "csv_processing"         ← L5: CSV 처리 (최대 깊이)
  │   │   │   │       └─ @task "parse_csv"
  │   │   │   │           ├─ step[1] detect_encoding
  │   │   │   │           ├─ step[2] validate_headers
  │   │   │   │           └─ step[3] parse_rows
  │   │   │   └─ @flow "db_ingestion"               ← L4: DB 수집
  │   │   │       └─ @task "extract_db"
  │   │   │           ├─ step[1] connect
  │   │   │           └─ step[2] extract
  │   │   └─ @flow "stream_ingestion"               ← L3: 스트림 수집
  │   │       └─ @task "consume_stream"
  │   │           ├─ step[1] subscribe
  │   │           └─ step[2] buffer_messages
  │   ├─ @flow "transformation"                     ← L2: 데이터 변환
  │   │   ├─ @flow "cleaning"                       ← L3: 데이터 정제
  │   │   │   └─ @task "clean_data"
  │   │   │       ├─ step[1] remove_nulls
  │   │   │       ├─ step[2] normalize_types
  │   │   │       └─ step[3] deduplicate
  │   │   ├─ @flow "enrichment"                     ← L3: 데이터 보강
  │   │   │   └─ @task "enrich_data"
  │   │   │       ├─ step[1] lookup_reference
  │   │   │       └─ step[2] compute_derived
  │   │   └─ @flow "aggregation"                    ← L3: 집계
  │   │       └─ @task "aggregate_data"
  │   │           ├─ step[1] group_by_key
  │   │           └─ step[2] compute_metrics
  │   └─ @flow "storage"                            ← L2: 저장
  │       ├─ @flow "warehouse"                      ← L3: 데이터 웨어하우스
  │       │   └─ @task "load_warehouse"
  │       │       ├─ step[1] create_schema
  │       │       └─ step[2] bulk_insert
  │       └─ @flow "lake"                           ← L3: 데이터 레이크
  │           └─ @task "store_lake"
  │               ├─ step[1] partition_data
  │               └─ step[2] write_parquet
  │
  ├─ @flow "ml_platform"                            ← L1: ML 플랫폼
  │   ├─ @flow "feature_engineering"                ← L2: 피처 엔지니어링
  │   │   ├─ @flow "feature_extraction"             ← L3: 피처 추출
  │   │   │   └─ @task "extract_features"
  │   │   │       ├─ step[1] analyze_schema
  │   │   │       ├─ step[2] compute_statistics
  │   │   │       └─ step[3] generate_features
  │   │   └─ @flow "feature_store"                  ← L3: 피처 스토어
  │   │       └─ @task "register_features"
  │   │           ├─ step[1] validate_features
  │   │           └─ step[2] publish_features
  │   ├─ @flow "training"                           ← L2: 모델 학습
  │   │   ├─ @flow "preprocessing"                  ← L3: 전처리
  │   │   │   └─ @task "prepare_dataset"
  │   │   │       ├─ step[1] split_data
  │   │   │       ├─ step[2] scale_features
  │   │   │       └─ step[3] encode_labels
  │   │   ├─ @flow "model_training"                 ← L3: 학습 실행
  │   │   │   ├─ @flow "hyperparameter_search"      ← L4: 하이퍼파라미터 탐색
  │   │   │   │   └─ @task "search_params"
  │   │   │   │       ├─ step[1] define_search_space
  │   │   │   │       └─ step[2] run_trials
  │   │   │   └─ @flow "distributed_training"       ← L4: 분산 학습
  │   │   │       └─ @task "train_model"
  │   │   │           ├─ step[1] init_cluster
  │   │   │           ├─ step[2] train_epochs
  │   │   │           └─ step[3] save_checkpoint
  │   │   └─ @flow "evaluation"                     ← L3: 모델 평가
  │   │       └─ @task "evaluate_model"
  │   │           ├─ step[1] compute_metrics
  │   │           ├─ step[2] generate_report
  │   │           └─ step[3] compare_baseline
  │   └─ @flow "deployment"                         ← L2: 모델 배포
  │       ├─ @flow "model_registry"                 ← L3: 모델 레지스트리
  │       │   └─ @task "register_model"
  │       │       ├─ step[1] version_model
  │       │       └─ step[2] tag_production
  │       ├─ @flow "serving"                        ← L3: 서빙
  │       │   └─ @task "deploy_endpoint"
  │       │       ├─ step[1] build_container
  │       │       ├─ step[2] deploy_canary
  │       │       └─ step[3] promote_production
  │       └─ @flow "monitoring"                     ← L3: 모니터링
  │           └─ @task "setup_monitoring"
  │               ├─ step[1] configure_alerts
  │               └─ step[2] setup_dashboards
  │
  ├─ @flow "devops"                                 ← L1: DevOps
  │   ├─ @flow "ci_pipeline"                        ← L2: CI 파이프라인
  │   │   ├─ @flow "build"                          ← L3: 빌드
  │   │   │   └─ @task "build_artifacts"
  │   │   │       ├─ step[1] resolve_deps
  │   │   │       ├─ step[2] compile_source
  │   │   │       └─ step[3] package_artifacts
  │   │   ├─ @flow "test"                           ← L3: 테스트
  │   │   │   └─ @task "run_tests"
  │   │   │       ├─ step[1] unit_tests
  │   │   │       ├─ step[2] integration_tests
  │   │   │       └─ step[3] coverage_report
  │   │   └─ @flow "security_scan"                  ← L3: 보안 스캔
  │   │       └─ @task "scan_vulnerabilities"
  │   │           ├─ step[1] sast_scan
  │   │           └─ step[2] dependency_audit
  │   └─ @flow "cd_pipeline"                        ← L2: CD 파이프라인
  │       ├─ @flow "staging_deploy"                 ← L3: 스테이징 배포
  │       │   └─ @task "deploy_staging"
  │       │       ├─ step[1] provision_env
  │       │       └─ step[2] deploy_services
  │       └─ @flow "production_deploy"              ← L3: 프로덕션 배포
  │           └─ @task "deploy_production"
  │               ├─ step[1] blue_green_switch
  │               ├─ step[2] smoke_test
  │               └─ step[3] notify_stakeholders
  │
  └─ @flow "observability"                          ← L1: 옵저버빌리티
      ├─ @flow "logging"                            ← L2: 로깅
      │   └─ @task "configure_logging"
      │       ├─ step[1] setup_log_pipeline
      │       └─ step[2] configure_retention
      ├─ @flow "metrics"                            ← L2: 메트릭
      │   └─ @task "configure_metrics"
      │       ├─ step[1] define_sli
      │       └─ step[2] setup_slo_alerts
      └─ @flow "tracing"                            ← L2: 분산 트레이싱
          └─ @task "configure_tracing"
              ├─ step[1] instrument_services
              └─ step[2] setup_trace_sampling

=======================================================================
실행 방법
=======================================================================

    python examples/enterprise_agent.py

=======================================================================
"""
import asyncio
import logging
import os
import time

from flowforge import global_config, flow, task, step, FlowForge
from flowforge.types import LLMConfig


# ═════════════════════════════════════════════════════════════════════════
# 유틸리티: step 함수 생성 팩토리
# ═════════════════════════════════════════════════════════════════════════

STEP_DELAY = 0.15  # 각 step 실행 시 딜레이 (초) — 실행 과정을 눈으로 볼 수 있도록


def _make_step(label: str):
    """주어진 label을 결과에 포함하는 step 핸들러를 생성한다."""
    async def handler(ctx):
        # 실행 진행 표시
        depth = ctx.task_ctx.task_name
        flow  = ctx.flow_ctx.flow_name
        print(f"    ▸ [{flow}/{depth}] {label} ...", end="", flush=True)
        await asyncio.sleep(STEP_DELAY)
        print(" done")

        inp = ctx.input
        # 이전 step 결과들을 누적
        if isinstance(inp, dict):
            trail = inp.get("trail", [])
        else:
            trail = getattr(inp, "trail", []) if inp else []
            if not isinstance(trail, list):
                trail = []
        trail = list(trail)  # 복사
        trail.append(label)

        # shared_data에 실행된 step 기록
        executed = ctx.shared_data.setdefault("executed_steps", [])
        executed.append(label)

        return {"trail": trail, "last_step": label}
    return handler


# ═════════════════════════════════════════════════════════════════════════
# L5 Flows (최대 깊이)
# ═════════════════════════════════════════════════════════════════════════

@flow(name="csv_processing", prompt=(
    "CSV 파일을 처리하는 파이프라인. BOM 감지, 인코딩 변환, 헤더 유효성 검증, "
    "데이터 타입 추론, 행 단위 파싱을 수행한다. 대용량 CSV의 경우 청크 단위로 "
    "처리하여 메모리 사용량을 최적화한다."
))
class CsvProcessingFlow:
    @task(name="parse_csv", prompt=(
        "CSV 파일을 파싱하여 구조화된 레코드로 변환한다. 인코딩 감지 → "
        "헤더 검증 → 행 파싱 순서로 진행하며, 각 단계에서 오류 발생 시 "
        "상세한 에러 위치(행/열)를 보고한다."
    ))
    class ParseCsvTask:
        @step(order=1, prompt=(
            "파일의 BOM(Byte Order Mark)을 확인하고 chardet 라이브러리로 "
            "인코딩을 감지한다. UTF-8, EUC-KR, Shift_JIS 등 주요 인코딩을 "
            "지원하며, 감지 신뢰도가 80% 미만이면 경고를 기록한다."
        ))
        async def detect_encoding(ctx):
            return await _make_step("csv.detect_encoding")(ctx)

        @step(order=2, prompt=(
            "CSV 헤더 행을 읽어 컬럼명의 유효성을 검증한다. 중복 컬럼명, "
            "빈 컬럼명, 예약어 사용 여부를 확인하고, 필수 컬럼이 누락된 "
            "경우 ValidationError를 발생시킨다."
        ))
        async def validate_headers(ctx):
            return await _make_step("csv.validate_headers")(ctx)

        @step(order=3, prompt=(
            "헤더 정보를 기반으로 각 행을 파싱한다. 데이터 타입을 자동 추론하고, "
            "날짜/숫자/불리언 등의 포맷 변환을 수행한다. 파싱 실패한 행은 "
            "에러 로그에 기록하되 처리를 계속 진행한다."
        ))
        async def parse_rows(ctx):
            return await _make_step("csv.parse_rows")(ctx)


# ═════════════════════════════════════════════════════════════════════════
# L4 Flows
# ═════════════════════════════════════════════════════════════════════════

@flow(name="file_ingestion", prompt=(
    "로컬 파일시스템 또는 클라우드 스토리지(S3, GCS, Azure Blob)에서 "
    "파일을 수집한다. CSV, JSON, Parquet, Avro 등 다양한 포맷을 지원하며 "
    "파일 크기와 포맷에 따라 최적의 처리 전략을 자동 선택한다."
))
class FileIngestionFlow:
    CsvProcessingFlow = CsvProcessingFlow


@flow(name="db_ingestion", prompt=(
    "관계형 데이터베이스(PostgreSQL, MySQL, Oracle)에서 데이터를 추출한다. "
    "CDC(Change Data Capture) 또는 전체 스냅샷 방식을 지원하며, "
    "커넥션 풀링과 쿼리 타임아웃을 관리한다."
))
class DbIngestionFlow:
    @task(name="extract_db", prompt=(
        "DB 연결 후 지정된 테이블/쿼리로 데이터를 추출한다. "
        "대용량 테이블은 커서 기반 페이지네이션으로 처리하고, "
        "추출 진행률을 실시간으로 보고한다."
    ))
    class ExtractDbTask:
        @step(order=1, prompt=(
            "데이터베이스 연결을 수립한다. 연결 파라미터(호스트, 포트, DB명, "
            "인증정보)를 검증하고, 연결 풀에서 커넥션을 할당받는다. "
            "SSL/TLS 설정이 있으면 인증서를 로드한다."
        ))
        async def connect(ctx):
            return await _make_step("db.connect")(ctx)

        @step(order=2, prompt=(
            "SQL 쿼리를 실행하여 결과를 추출한다. EXPLAIN ANALYZE로 실행 계획을 "
            "먼저 확인하고, 예상 소요시간이 임계치를 초과하면 경고를 발생시킨다. "
            "결과는 배치 단위(기본 10,000행)로 페치하여 메모리를 관리한다."
        ))
        async def extract(ctx):
            return await _make_step("db.extract")(ctx)


@flow(name="hyperparameter_search", prompt=(
    "모델의 하이퍼파라미터를 자동으로 탐색한다. Grid Search, Random Search, "
    "Bayesian Optimization 중 데이터 규모와 파라미터 공간에 따라 최적의 "
    "전략을 선택한다. 조기 종료(Early Stopping) 조건도 지원한다."
))
class HyperparameterSearchFlow:
    @task(name="search_params", prompt=(
        "탐색 공간을 정의하고 최적 파라미터 조합을 찾는다. "
        "각 trial의 성능 메트릭을 기록하고 중간 결과를 시각화한다."
    ))
    class SearchParamsTask:
        @step(order=1, prompt=(
            "탐색할 하이퍼파라미터의 범위와 분포를 정의한다. "
            "연속형(learning_rate: 1e-5 ~ 1e-1), 이산형(num_layers: 2~8), "
            "카테고리형(optimizer: adam|sgd|adamw) 파라미터를 구분한다."
        ))
        async def define_search_space(ctx):
            return await _make_step("hp.define_search_space")(ctx)

        @step(order=2, prompt=(
            "정의된 탐색 공간에서 trial을 실행한다. 각 trial은 독립된 "
            "학습 세션으로 실행되며, validation metric 기준으로 상위 N개 "
            "조합을 최종 후보로 선정한다."
        ))
        async def run_trials(ctx):
            return await _make_step("hp.run_trials")(ctx)


@flow(name="distributed_training", prompt=(
    "여러 GPU/노드에 걸쳐 모델을 분산 학습한다. "
    "Data Parallel 또는 Model Parallel 전략을 선택하며, "
    "gradient 동기화와 체크포인트 저장을 자동으로 관리한다."
))
class DistributedTrainingFlow:
    @task(name="train_model", prompt=(
        "분산 환경에서 모델을 학습한다. 클러스터 초기화, 에폭 반복 학습, "
        "체크포인트 저장의 세 단계로 진행되며, 노드 장애 시 "
        "마지막 체크포인트에서 자동 복구한다."
    ))
    class TrainModelTask:
        @step(order=1, prompt=(
            "분산 학습 클러스터를 초기화한다. 마스터 노드와 워커 노드 간 "
            "통신 채널(NCCL/Gloo)을 수립하고, GPU 메모리 할당 상태를 확인한다. "
            "CUDA OOM 위험 시 배치 크기 자동 조정을 제안한다."
        ))
        async def init_cluster(ctx):
            return await _make_step("train.init_cluster")(ctx)

        @step(order=2, prompt=(
            "설정된 에폭 수만큼 학습을 반복한다. 각 에폭마다 train loss, "
            "validation loss, learning rate를 기록하고, Early Stopping "
            "조건(patience=5, min_delta=0.001)을 평가한다."
        ))
        async def train_epochs(ctx):
            return await _make_step("train.train_epochs")(ctx)

        @step(order=3, prompt=(
            "학습된 모델 가중치와 옵티마이저 상태를 체크포인트 파일로 저장한다. "
            "저장 형식은 PyTorch state_dict이며, 모델 메타데이터(아키텍처, "
            "하이퍼파라미터, 학습 이력)도 함께 기록한다."
        ))
        async def save_checkpoint(ctx):
            return await _make_step("train.save_checkpoint")(ctx)


# ═════════════════════════════════════════════════════════════════════════
# L3 Flows
# ═════════════════════════════════════════════════════════════════════════

@flow(name="batch_ingestion", prompt=(
    "배치 방식으로 대량 데이터를 수집한다. 파일 기반 수집과 DB 기반 수집을 "
    "모두 지원하며, 수집 완료 후 데이터 무결성 체크섬을 검증한다."
))
class BatchIngestionFlow:
    FileIngestionFlow = FileIngestionFlow
    DbIngestionFlow = DbIngestionFlow


@flow(name="stream_ingestion", prompt=(
    "실시간 스트리밍 데이터를 수집한다. Kafka, Kinesis, Pub/Sub 등의 "
    "메시지 브로커에서 이벤트를 소비하고, 윈도우 기반 마이크로 배치로 "
    "버퍼링한다. 백프레셔(backpressure) 제어를 자동으로 수행한다."
))
class StreamIngestionFlow:
    @task(name="consume_stream", prompt=(
        "메시지 브로커에서 이벤트를 소비한다. 컨슈머 그룹 관리, "
        "오프셋 커밋 전략(at-least-once / exactly-once), "
        "역직렬화 및 메시지 검증을 수행한다."
    ))
    class ConsumeStreamTask:
        @step(order=1, prompt=(
            "메시지 브로커의 토픽에 구독한다. 파티션 할당 전략을 설정하고 "
            "초기 오프셋 위치(earliest/latest/특정 타임스탬프)를 결정한다."
        ))
        async def subscribe(ctx):
            return await _make_step("stream.subscribe")(ctx)

        @step(order=2, prompt=(
            "수신된 메시지를 시간/개수 기반 윈도우로 버퍼링한다. "
            "윈도우 크기(기본 5초 또는 1000건)에 도달하면 하나의 "
            "마이크로 배치로 묶어서 다음 단계로 전달한다."
        ))
        async def buffer_messages(ctx):
            return await _make_step("stream.buffer_messages")(ctx)


@flow(name="cleaning", prompt=(
    "원시 데이터의 품질 문제를 해결한다. NULL 값 처리, 타입 정규화, "
    "중복 제거를 순차적으로 수행하며, 각 단계에서 변경된 레코드 수와 "
    "품질 점수를 기록한다."
))
class CleaningFlow:
    @task(name="clean_data", prompt=(
        "데이터 품질을 보장하기 위한 정제 작업을 수행한다. "
        "입력 데이터의 컬럼별 NULL 비율, 타입 불일치율, 중복률을 "
        "먼저 프로파일링한 후 적절한 정제 전략을 적용한다."
    ))
    class CleanDataTask:
        @step(order=1, prompt=(
            "NULL 및 결측치를 처리한다. 컬럼별 전략을 적용: "
            "수치형은 중앙값/평균으로 대체, 범주형은 최빈값으로 대체, "
            "필수 컬럼이 NULL이면 해당 행을 제거한다."
        ))
        async def remove_nulls(ctx):
            return await _make_step("clean.remove_nulls")(ctx)

        @step(order=2, prompt=(
            "데이터 타입을 스키마에 맞게 정규화한다. 문자열로 저장된 "
            "숫자/날짜를 적절한 타입으로 변환하고, 변환 실패 건수를 기록한다."
        ))
        async def normalize_types(ctx):
            return await _make_step("clean.normalize_types")(ctx)

        @step(order=3, prompt=(
            "중복 레코드를 감지하고 제거한다. 정확 일치 + 유사도 기반 "
            "퍼지 매칭을 지원하며, 중복 그룹 중 가장 최신 레코드를 유지한다."
        ))
        async def deduplicate(ctx):
            return await _make_step("clean.deduplicate")(ctx)


@flow(name="enrichment", prompt=(
    "정제된 데이터에 외부 참조 정보를 결합하여 가치를 높인다. "
    "조인, 룩업, 파생 컬럼 계산 등을 수행한다."
))
class EnrichmentFlow:
    @task(name="enrich_data", prompt=(
        "데이터에 추가 정보를 결합한다. 참조 테이블 조인과 "
        "파생 컬럼 계산을 순차적으로 수행한다."
    ))
    class EnrichDataTask:
        @step(order=1, prompt=(
            "외부 참조 테이블(코드 마스터, 지역 정보 등)과 조인하여 "
            "원본 데이터에 부가 정보를 추가한다. 매칭되지 않는 행은 "
            "NULL로 표시하고 매칭률 통계를 보고한다."
        ))
        async def lookup_reference(ctx):
            return await _make_step("enrich.lookup_reference")(ctx)

        @step(order=2, prompt=(
            "기존 컬럼들의 조합으로 파생 컬럼을 계산한다. "
            "예: 매출 = 단가 × 수량, 연령대 = floor(나이/10)*10 등."
        ))
        async def compute_derived(ctx):
            return await _make_step("enrich.compute_derived")(ctx)


@flow(name="aggregation", prompt=(
    "정제·보강된 데이터를 그룹별로 집계한다. "
    "SUM, AVG, COUNT, MIN, MAX 등의 집계 함수와 "
    "시간 기반 윈도우 집계를 지원한다."
))
class AggregationFlow:
    @task(name="aggregate_data", prompt=(
        "데이터를 키 기준으로 그룹화하고 메트릭을 계산한다."
    ))
    class AggregateDataTask:
        @step(order=1, prompt=(
            "지정된 키 컬럼(예: 날짜, 카테고리, 지역)으로 "
            "데이터를 그룹화한다. 복합 키도 지원한다."
        ))
        async def group_by_key(ctx):
            return await _make_step("agg.group_by_key")(ctx)

        @step(order=2, prompt=(
            "각 그룹에 대해 집계 메트릭을 계산한다. "
            "합계, 평균, 건수, 최소, 최대, 백분위수 등을 산출한다."
        ))
        async def compute_metrics(ctx):
            return await _make_step("agg.compute_metrics")(ctx)


@flow(name="warehouse", prompt=(
    "처리된 데이터를 데이터 웨어하우스에 적재한다. "
    "스키마 관리, 파티셔닝, 인덱싱을 자동으로 수행하며 "
    "UPSERT(INSERT or UPDATE) 전략을 지원한다."
))
class WarehouseFlow:
    @task(name="load_warehouse", prompt=(
        "데이터 웨어하우스에 데이터를 적재한다. 대상 스키마를 "
        "자동 생성하고 벌크 인서트로 고속 적재한다."
    ))
    class LoadWarehouseTask:
        @step(order=1, prompt=(
            "데이터 구조에 맞는 테이블 스키마를 생성하거나 기존 스키마와 "
            "호환성을 검증한다. 스키마 변경이 필요하면 ALTER 문을 자동 생성한다."
        ))
        async def create_schema(ctx):
            return await _make_step("wh.create_schema")(ctx)

        @step(order=2, prompt=(
            "COPY 또는 멀티 로우 INSERT 문으로 대량 데이터를 적재한다. "
            "배치 크기를 자동 조절하여 DB 부하를 관리한다."
        ))
        async def bulk_insert(ctx):
            return await _make_step("wh.bulk_insert")(ctx)


@flow(name="lake", prompt=(
    "처리된 데이터를 데이터 레이크에 저장한다. "
    "파티셔닝 전략(날짜/카테고리)에 따라 디렉토리 구조를 "
    "생성하고 Parquet 포맷으로 저장한다."
))
class LakeFlow:
    @task(name="store_lake", prompt=(
        "데이터 레이크에 최적화된 형태로 데이터를 저장한다."
    ))
    class StoreLakeTask:
        @step(order=1, prompt=(
            "파티션 키(예: year/month/day)에 따라 데이터를 분할하고 "
            "각 파티션의 디렉토리 경로를 생성한다."
        ))
        async def partition_data(ctx):
            return await _make_step("lake.partition_data")(ctx)

        @step(order=2, prompt=(
            "분할된 데이터를 Snappy 압축 Parquet 파일로 변환하여 저장한다. "
            "Row Group 크기와 페이지 크기를 최적화하여 쿼리 성능을 확보한다."
        ))
        async def write_parquet(ctx):
            return await _make_step("lake.write_parquet")(ctx)


@flow(name="feature_extraction", prompt=(
    "원시 데이터에서 ML 모델에 사용할 피처를 추출한다. "
    "통계적 분석을 기반으로 유의미한 피처를 자동 생성하고, "
    "피처 중요도를 사전 평가한다."
))
class FeatureExtractionFlow:
    @task(name="extract_features", prompt=(
        "데이터 스키마를 분석하고, 통계를 계산하고, "
        "최종 피처 벡터를 생성한다."
    ))
    class ExtractFeaturesTask:
        @step(order=1, prompt=(
            "입력 데이터의 스키마(컬럼명, 타입, 카디널리티)를 분석하여 "
            "각 컬럼의 피처 후보 여부를 판단한다."
        ))
        async def analyze_schema(ctx):
            return await _make_step("feat.analyze_schema")(ctx)

        @step(order=2, prompt=(
            "각 컬럼의 기술 통계량(mean, std, skewness, kurtosis)을 계산하고 "
            "분포 특성을 파악한다. 이상치 비율도 함께 산출한다."
        ))
        async def compute_statistics(ctx):
            return await _make_step("feat.compute_statistics")(ctx)

        @step(order=3, prompt=(
            "분석 결과를 기반으로 최종 피처를 생성한다. 원-핫 인코딩, "
            "빈 도수 인코딩, 타겟 인코딩 등 컬럼 특성에 맞는 변환을 적용한다."
        ))
        async def generate_features(ctx):
            return await _make_step("feat.generate_features")(ctx)


@flow(name="feature_store", prompt=(
    "생성된 피처를 피처 스토어에 등록·관리한다. "
    "버전 관리, 메타데이터 추적, 온라인/오프라인 서빙을 지원한다."
))
class FeatureStoreFlow:
    @task(name="register_features", prompt=(
        "피처를 검증하고 피처 스토어에 등록한다."
    ))
    class RegisterFeaturesTask:
        @step(order=1, prompt=(
            "피처의 품질을 검증한다. NULL 비율, 값 범위, 분포 안정성을 "
            "확인하고, 기존 버전 대비 드리프트가 있는지 비교한다."
        ))
        async def validate_features(ctx):
            return await _make_step("fs.validate_features")(ctx)

        @step(order=2, prompt=(
            "검증 통과한 피처를 스토어에 등록한다. 버전 태그를 부여하고 "
            "메타데이터(생성 시각, 소스, 통계 정보)를 기록한다."
        ))
        async def publish_features(ctx):
            return await _make_step("fs.publish_features")(ctx)


@flow(name="preprocessing", prompt=(
    "학습 데이터를 전처리한다. 데이터 분할, 피처 스케일링, "
    "라벨 인코딩을 수행하여 모델 학습에 적합한 형태로 변환한다."
))
class PreprocessingFlow:
    @task(name="prepare_dataset", prompt=(
        "학습용 데이터셋을 준비한다. train/val/test 분할, "
        "피처 정규화, 라벨 인코딩을 순차적으로 수행한다."
    ))
    class PrepareDatasetTask:
        @step(order=1, prompt=(
            "데이터를 train/validation/test로 분할한다. "
            "층화 추출(Stratified Split)을 적용하여 클래스 분포를 유지한다. "
            "기본 비율: train 70%, val 15%, test 15%."
        ))
        async def split_data(ctx):
            return await _make_step("prep.split_data")(ctx)

        @step(order=2, prompt=(
            "수치형 피처를 스케일링한다. StandardScaler(평균 0, 표준편차 1) "
            "또는 MinMaxScaler(0~1 범위)를 적용한다. "
            "스케일러 파라미터는 train 셋에서만 fit한다."
        ))
        async def scale_features(ctx):
            return await _make_step("prep.scale_features")(ctx)

        @step(order=3, prompt=(
            "타겟 라벨을 수치형으로 인코딩한다. 이진 분류는 0/1, "
            "다중 클래스는 정수 인코딩, 다중 라벨은 멀티-핫 인코딩을 적용한다."
        ))
        async def encode_labels(ctx):
            return await _make_step("prep.encode_labels")(ctx)


@flow(name="model_training", prompt=(
    "전처리된 데이터로 ML 모델을 학습한다. 하이퍼파라미터 탐색과 "
    "분산 학습을 포함하며, 최적 모델을 선정하여 체크포인트로 저장한다."
))
class ModelTrainingFlow:
    HyperparameterSearchFlow = HyperparameterSearchFlow
    DistributedTrainingFlow = DistributedTrainingFlow


@flow(name="evaluation", prompt=(
    "학습된 모델의 성능을 종합적으로 평가한다. 정량적 메트릭 계산, "
    "성능 리포트 생성, 베이스라인 모델과의 비교 분석을 수행한다."
))
class EvaluationFlow:
    @task(name="evaluate_model", prompt=(
        "모델을 test 셋에서 평가하고 상세 리포트를 생성한다."
    ))
    class EvaluateModelTask:
        @step(order=1, prompt=(
            "Accuracy, Precision, Recall, F1-Score, AUC-ROC 등 "
            "분류 메트릭을 계산한다. 회귀 모델의 경우 MSE, MAE, R² 를 산출한다."
        ))
        async def compute_metrics(ctx):
            return await _make_step("eval.compute_metrics")(ctx)

        @step(order=2, prompt=(
            "혼동 행렬, ROC 커브, PR 커브, 피처 중요도 차트를 포함한 "
            "시각적 성능 리포트를 생성한다. HTML과 PDF 형식으로 출력한다."
        ))
        async def generate_report(ctx):
            return await _make_step("eval.generate_report")(ctx)

        @step(order=3, prompt=(
            "현재 모델의 성능을 베이스라인 모델(이전 프로덕션 버전)과 비교한다. "
            "통계적 유의성 검정(McNemar test)을 수행하고, "
            "배포 추천 여부를 판정한다."
        ))
        async def compare_baseline(ctx):
            return await _make_step("eval.compare_baseline")(ctx)


@flow(name="model_registry", prompt=(
    "학습·평가 완료된 모델을 레지스트리에 등록한다. "
    "버전 관리, 스테이지 전환(staging → production), 모델 메타데이터 관리를 수행한다."
))
class ModelRegistryFlow:
    @task(name="register_model", prompt=(
        "모델 아티팩트와 메타데이터를 레지스트리에 등록한다."
    ))
    class RegisterModelTask:
        @step(order=1, prompt=(
            "시맨틱 버저닝(major.minor.patch)으로 모델 버전을 부여한다. "
            "성능 개선이면 minor, 아키텍처 변경이면 major를 올린다."
        ))
        async def version_model(ctx):
            return await _make_step("reg.version_model")(ctx)

        @step(order=2, prompt=(
            "검증 완료된 모델에 'production' 태그를 부여하고, "
            "이전 production 모델은 'archived'로 전환한다."
        ))
        async def tag_production(ctx):
            return await _make_step("reg.tag_production")(ctx)


@flow(name="serving", prompt=(
    "모델을 API 엔드포인트로 서빙한다. 컨테이너 빌드, "
    "카나리 배포, 프로덕션 승격의 단계를 거친다."
))
class ServingFlow:
    @task(name="deploy_endpoint", prompt=(
        "모델 서빙 엔드포인트를 배포한다. 점진적 트래픽 전환으로 "
        "무중단 배포를 보장한다."
    ))
    class DeployEndpointTask:
        @step(order=1, prompt=(
            "모델 아티팩트를 포함한 서빙 컨테이너(Docker) 이미지를 빌드한다. "
            "의존성 레이어 캐싱으로 빌드 시간을 최적화한다."
        ))
        async def build_container(ctx):
            return await _make_step("serve.build_container")(ctx)

        @step(order=2, prompt=(
            "전체 트래픽의 5%만 새 버전으로 라우팅하는 카나리 배포를 수행한다. "
            "에러율과 레이턴시를 모니터링하여 이상 시 자동 롤백한다."
        ))
        async def deploy_canary(ctx):
            return await _make_step("serve.deploy_canary")(ctx)

        @step(order=3, prompt=(
            "카나리 검증 통과 후 트래픽 100%를 새 버전으로 전환한다. "
            "이전 버전은 롤백 대비용으로 30분간 유지한다."
        ))
        async def promote_production(ctx):
            return await _make_step("serve.promote_production")(ctx)


@flow(name="model_monitoring", prompt=(
    "배포된 모델의 성능과 데이터 드리프트를 실시간 모니터링한다. "
    "알림 규칙과 대시보드를 자동 구성한다."
))
class ModelMonitoringFlow:
    @task(name="setup_monitoring", prompt=(
        "모델 모니터링 인프라를 구성한다. 성능 저하 감지 알림과 "
        "운영 대시보드를 셋업한다."
    ))
    class SetupMonitoringTask:
        @step(order=1, prompt=(
            "모델 성능 메트릭(정확도, 레이턴시, 에러율)에 대한 "
            "알림 임계치를 설정한다. 슬랙/PagerDuty 연동도 구성한다."
        ))
        async def configure_alerts(ctx):
            return await _make_step("mon.configure_alerts")(ctx)

        @step(order=2, prompt=(
            "Grafana 대시보드를 자동 생성한다. 실시간 QPS, "
            "예측 분포, 피처 드리프트 차트를 포함한다."
        ))
        async def setup_dashboards(ctx):
            return await _make_step("mon.setup_dashboards")(ctx)


@flow(name="build", prompt=(
    "소스 코드를 빌드하여 배포 가능한 아티팩트를 생성한다. "
    "의존성 해석, 컴파일, 패키징의 단계를 수행한다."
))
class BuildFlow:
    @task(name="build_artifacts", prompt=(
        "빌드 파이프라인을 실행하여 아티팩트를 생성한다."
    ))
    class BuildArtifactsTask:
        @step(order=1, prompt=(
            "프로젝트의 의존성 트리를 해석하고 필요한 패키지를 "
            "다운로드한다. 버전 충돌이 있으면 해결 전략을 적용한다."
        ))
        async def resolve_deps(ctx):
            return await _make_step("build.resolve_deps")(ctx)

        @step(order=2, prompt=(
            "소스 코드를 컴파일한다. 타입 체크, 린트, "
            "바이트코드/바이너리 생성을 수행한다."
        ))
        async def compile_source(ctx):
            return await _make_step("build.compile_source")(ctx)

        @step(order=3, prompt=(
            "컴파일된 결과물을 Docker 이미지, wheel, JAR 등의 "
            "배포 가능한 아티팩트로 패키징한다."
        ))
        async def package_artifacts(ctx):
            return await _make_step("build.package_artifacts")(ctx)


@flow(name="test", prompt=(
    "빌드된 코드에 대해 자동화된 테스트를 실행한다. "
    "유닛 테스트, 통합 테스트, 커버리지 측정을 포함한다."
))
class TestFlow:
    @task(name="run_tests", prompt=(
        "테스트 스위트를 실행하고 결과를 리포트한다."
    ))
    class RunTestsTask:
        @step(order=1, prompt=(
            "pytest로 유닛 테스트를 실행한다. 병렬 실행(-n auto)으로 "
            "속도를 최적화하고, 실패한 테스트를 우선 리포트한다."
        ))
        async def unit_tests(ctx):
            return await _make_step("test.unit_tests")(ctx)

        @step(order=2, prompt=(
            "실제 DB/API를 사용하는 통합 테스트를 실행한다. "
            "테스트용 Docker 컨테이너를 자동으로 관리(testcontainers)한다."
        ))
        async def integration_tests(ctx):
            return await _make_step("test.integration_tests")(ctx)

        @step(order=3, prompt=(
            "코드 커버리지를 측정하고 리포트를 생성한다. "
            "라인/브랜치 커버리지를 산출하며, 임계치(80%) 미만이면 경고한다."
        ))
        async def coverage_report(ctx):
            return await _make_step("test.coverage_report")(ctx)


@flow(name="security_scan", prompt=(
    "코드와 의존성의 보안 취약점을 스캔한다. SAST(정적 분석)와 "
    "의존성 감사(SCA)를 수행하여 보안 리포트를 생성한다."
))
class SecurityScanFlow:
    @task(name="scan_vulnerabilities", prompt=(
        "보안 취약점을 스캔하고 리포트를 생성한다."
    ))
    class ScanVulnerabilitiesTask:
        @step(order=1, prompt=(
            "Semgrep/Bandit 등으로 소스 코드의 정적 보안 분석(SAST)을 수행한다. "
            "SQL Injection, XSS, 하드코딩된 시크릿 등을 감지한다."
        ))
        async def sast_scan(ctx):
            return await _make_step("sec.sast_scan")(ctx)

        @step(order=2, prompt=(
            "의존성 패키지의 알려진 취약점(CVE)을 점검한다. "
            "심각도별(Critical/High/Medium/Low) 분류하고 패치 버전을 추천한다."
        ))
        async def dependency_audit(ctx):
            return await _make_step("sec.dependency_audit")(ctx)


@flow(name="staging_deploy", prompt=(
    "스테이징 환경에 서비스를 배포한다. 프로덕션과 동일한 환경을 "
    "프로비저닝하고 서비스를 배포하여 통합 검증을 수행한다."
))
class StagingDeployFlow:
    @task(name="deploy_staging", prompt=(
        "스테이징 환경을 프로비저닝하고 서비스를 배포한다."
    ))
    class DeployStagingTask:
        @step(order=1, prompt=(
            "Terraform/Pulumi로 스테이징 인프라(VPC, 서브넷, 시큐리티 그룹, "
            "ECS/EKS 클러스터)를 프로비저닝한다."
        ))
        async def provision_env(ctx):
            return await _make_step("staging.provision_env")(ctx)

        @step(order=2, prompt=(
            "빌드된 아티팩트를 스테이징 클러스터에 배포한다. "
            "헬스체크 통과를 확인하고 E2E 테스트를 트리거한다."
        ))
        async def deploy_services(ctx):
            return await _make_step("staging.deploy_services")(ctx)


@flow(name="production_deploy", prompt=(
    "프로덕션 환경에 서비스를 배포한다. Blue-Green 배포 전략으로 "
    "무중단 배포를 수행하고, 스모크 테스트 후 이해관계자에게 통지한다."
))
class ProductionDeployFlow:
    @task(name="deploy_production", prompt=(
        "프로덕션 무중단 배포를 수행한다. Blue-Green 전환 → "
        "스모크 테스트 → 이해관계자 통지 순서로 진행한다."
    ))
    class DeployProductionTask:
        @step(order=1, prompt=(
            "Blue-Green 배포 전략으로 새 버전을 Green 환경에 배포하고, "
            "로드밸런서의 트래픽을 Green으로 전환한다. "
            "실패 시 즉시 Blue로 롤백한다."
        ))
        async def blue_green_switch(ctx):
            return await _make_step("prod.blue_green_switch")(ctx)

        @step(order=2, prompt=(
            "핵심 API 엔드포인트에 대한 스모크 테스트를 실행한다. "
            "응답 코드, 레이턴시, 필수 필드 존재 여부를 검증한다."
        ))
        async def smoke_test(ctx):
            return await _make_step("prod.smoke_test")(ctx)

        @step(order=3, prompt=(
            "배포 완료를 이해관계자(PO, QA, SRE)에게 Slack/이메일로 통지한다. "
            "배포 버전, 변경 사항 요약, 롤백 절차를 포함한다."
        ))
        async def notify_stakeholders(ctx):
            return await _make_step("prod.notify_stakeholders")(ctx)


@flow(name="logging", prompt=(
    "중앙 집중식 로깅 파이프라인을 구성한다. 애플리케이션 로그를 "
    "수집·파싱·저장하고 보관 정책을 관리한다."
))
class LoggingFlow:
    @task(name="configure_logging", prompt=(
        "로깅 인프라를 구성한다. 파이프라인 셋업과 보관 정책을 설정한다."
    ))
    class ConfigureLoggingTask:
        @step(order=1, prompt=(
            "Fluentd/Vector로 로그 수집 파이프라인을 구성한다. "
            "구조화 로깅(JSON) 포맷을 강제하고, 민감 정보를 마스킹한다."
        ))
        async def setup_log_pipeline(ctx):
            return await _make_step("log.setup_log_pipeline")(ctx)

        @step(order=2, prompt=(
            "로그 보관 정책을 설정한다. Hot(7일, SSD), Warm(30일, HDD), "
            "Cold(90일, S3 Glacier) 3-tier 전략을 적용한다."
        ))
        async def configure_retention(ctx):
            return await _make_step("log.configure_retention")(ctx)


@flow(name="metrics_collection", prompt=(
    "시스템 및 비즈니스 메트릭을 수집·저장하고, "
    "SLI/SLO 기반 알림을 구성한다."
))
class MetricsFlow:
    @task(name="configure_metrics", prompt=(
        "메트릭 수집과 알림을 구성한다."
    ))
    class ConfigureMetricsTask:
        @step(order=1, prompt=(
            "서비스별 SLI(Service Level Indicator)를 정의한다. "
            "가용성(uptime), 레이턴시(p99), 에러율을 핵심 지표로 설정한다."
        ))
        async def define_sli(ctx):
            return await _make_step("metrics.define_sli")(ctx)

        @step(order=2, prompt=(
            "SLO(Service Level Objective) 기반 알림을 구성한다. "
            "번 다운 차트(error budget burn rate)로 SLO 위반 예측 알림을 설정한다."
        ))
        async def setup_slo_alerts(ctx):
            return await _make_step("metrics.setup_slo_alerts")(ctx)


@flow(name="tracing", prompt=(
    "마이크로서비스 간 분산 트레이싱을 구성한다. "
    "OpenTelemetry 기반으로 요청 추적과 샘플링을 관리한다."
))
class TracingFlow:
    @task(name="configure_tracing", prompt=(
        "분산 트레이싱 인프라를 구성한다."
    ))
    class ConfigureTracingTask:
        @step(order=1, prompt=(
            "각 서비스에 OpenTelemetry SDK를 적용하여 자동 계측한다. "
            "HTTP, gRPC, DB 호출에 대한 스팬을 자동 생성한다."
        ))
        async def instrument_services(ctx):
            return await _make_step("trace.instrument_services")(ctx)

        @step(order=2, prompt=(
            "트레이스 샘플링 전략을 설정한다. 에러 트레이스는 100% 수집, "
            "정상 트레이스는 1% 샘플링으로 비용과 가시성 균형을 맞춘다."
        ))
        async def setup_trace_sampling(ctx):
            return await _make_step("trace.setup_trace_sampling")(ctx)


# ═════════════════════════════════════════════════════════════════════════
# L2 Flows
# ═════════════════════════════════════════════════════════════════════════

@flow(name="ingestion", prompt=(
    "다양한 소스에서 원시 데이터를 수집하는 최상위 수집 파이프라인. "
    "배치 수집과 스트림 수집을 모두 관리하며, 수집 모드를 "
    "데이터 특성에 따라 자동으로 선택한다."
))
class IngestionFlow:
    BatchIngestionFlow = BatchIngestionFlow
    StreamIngestionFlow = StreamIngestionFlow


@flow(name="transformation", prompt=(
    "수집된 원시 데이터를 분석/ML에 적합한 형태로 변환한다. "
    "정제(Cleaning) → 보강(Enrichment) → 집계(Aggregation) 파이프라인을 "
    "순차적으로 실행하며, 각 단계의 데이터 품질 점수를 추적한다."
))
class TransformationFlow:
    CleaningFlow = CleaningFlow
    EnrichmentFlow = EnrichmentFlow
    AggregationFlow = AggregationFlow


@flow(name="storage", prompt=(
    "변환 완료된 데이터를 목적에 맞는 저장소에 적재한다. "
    "구조화된 분석 쿼리용은 웨어하우스로, "
    "비정형/대용량 데이터는 레이크로 라우팅한다."
))
class StorageFlow:
    WarehouseFlow = WarehouseFlow
    LakeFlow = LakeFlow


@flow(name="feature_engineering", prompt=(
    "ML 모델 학습에 필요한 피처를 설계·추출·관리한다. "
    "원시 데이터에서 피처를 추출하고 피처 스토어에 등록하여 "
    "재사용 가능한 피처 파이프라인을 구축한다."
))
class FeatureEngineeringFlow:
    FeatureExtractionFlow = FeatureExtractionFlow
    FeatureStoreFlow = FeatureStoreFlow


@flow(name="training", prompt=(
    "데이터 전처리부터 모델 학습·평가까지의 전체 학습 파이프라인. "
    "전처리 → 하이퍼파라미터 탐색 → 분산 학습 → 평가 순서로 진행한다."
))
class TrainingFlow:
    PreprocessingFlow = PreprocessingFlow
    ModelTrainingFlow = ModelTrainingFlow
    EvaluationFlow = EvaluationFlow


@flow(name="deployment", prompt=(
    "학습·평가 완료된 모델을 프로덕션 환경에 배포한다. "
    "모델 레지스트리 등록 → API 서빙 → 모니터링 구성 순서로 진행한다."
))
class DeploymentFlow:
    ModelRegistryFlow = ModelRegistryFlow
    ServingFlow = ServingFlow
    ModelMonitoringFlow = ModelMonitoringFlow


@flow(name="ci_pipeline", prompt=(
    "지속적 통합(CI) 파이프라인. 코드 변경 시 자동으로 빌드, 테스트, "
    "보안 스캔을 수행하여 코드 품질을 보장한다."
))
class CiPipelineFlow:
    BuildFlow = BuildFlow
    TestFlow = TestFlow
    SecurityScanFlow = SecurityScanFlow


@flow(name="cd_pipeline", prompt=(
    "지속적 배포(CD) 파이프라인. CI 통과 후 스테이징 → 프로덕션 순서로 "
    "자동 배포하며, 각 단계에서 검증을 수행한다."
))
class CdPipelineFlow:
    StagingDeployFlow = StagingDeployFlow
    ProductionDeployFlow = ProductionDeployFlow


# ═════════════════════════════════════════════════════════════════════════
# L1 Flows (최상위)
# ═════════════════════════════════════════════════════════════════════════

@flow(name="data_platform", prompt=(
    "엔터프라이즈 데이터 플랫폼. 데이터 수집(Ingestion), 변환(Transformation), "
    "저장(Storage)의 전체 라이프사이클을 관리한다. 배치와 실시간 처리를 모두 "
    "지원하며, 데이터 리니지와 품질 추적을 자동화한다."
))
class DataPlatformFlow:
    IngestionFlow = IngestionFlow
    TransformationFlow = TransformationFlow
    StorageFlow = StorageFlow


@flow(name="ml_platform", prompt=(
    "엔드투엔드 ML 플랫폼. 피처 엔지니어링부터 모델 학습, 배포, 모니터링까지 "
    "ML 라이프사이클 전체를 관리한다. MLOps 모범 사례를 기반으로 "
    "재현 가능하고 자동화된 ML 파이프라인을 제공한다."
))
class MlPlatformFlow:
    FeatureEngineeringFlow = FeatureEngineeringFlow
    TrainingFlow = TrainingFlow
    DeploymentFlow = DeploymentFlow


@flow(name="devops", prompt=(
    "DevOps 자동화 플랫폼. CI(지속적 통합)와 CD(지속적 배포) 파이프라인을 "
    "통합 관리하며, 코드 변경부터 프로덕션 배포까지의 전체 워크플로우를 "
    "자동화한다. GitOps 원칙을 따른다."
))
class DevopsFlow:
    CiPipelineFlow = CiPipelineFlow
    CdPipelineFlow = CdPipelineFlow


@flow(name="observability", prompt=(
    "시스템 옵저버빌리티(관측 가능성) 플랫폼. 로깅, 메트릭, 분산 트레이싱의 "
    "세 가지 축(Three Pillars)을 통합 구성하여 시스템 상태를 "
    "실시간으로 파악하고 장애를 신속하게 탐지·진단한다."
))
class ObservabilityFlow:
    LoggingFlow = LoggingFlow
    MetricsFlow = MetricsFlow
    TracingFlow = TracingFlow


# ═════════════════════════════════════════════════════════════════════════
# Global Config (에이전트 루트)
# ═════════════════════════════════════════════════════════════════════════

@global_config(
    prompt=(
        "엔터프라이즈 플랫폼 관리 에이전트. 데이터 플랫폼, ML 플랫폼, "
        "DevOps, 옵저버빌리티의 4개 도메인을 통합 관리한다. "
        "각 도메인의 파이프라인을 구성·실행·모니터링하며, "
        "도메인 간 의존성과 데이터 흐름을 자동으로 조율한다."
    ),
    # NOTE: LLMConfig 는 @global_config 수준에서 FlowForge가 사용한다.
    #   - DocGenerator.generate_all() 가 각 노드 doc 생성 시 참조.
    #   - Autonomous/Hybrid LLMPlanner 가 경로 선택 시 참조.
    # API 키는 환경변수(ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY)에서
    # 자동으로 로드된다.
    llm_config=LLMConfig(verify_ssl=False),
)
class EnterprisePlatformAgent:
    DataPlatformFlow = DataPlatformFlow
    MlPlatformFlow = MlPlatformFlow
    DevopsFlow = DevopsFlow
    ObservabilityFlow = ObservabilityFlow


# ═════════════════════════════════════════════════════════════════════════
# 출력 헬퍼
# ═════════════════════════════════════════════════════════════════════════

SEP = "=" * 72
THIN = "-" * 72
MERMAID_DIR = os.path.expanduser("~/files")


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def sub(title: str) -> None:
    print(f"\n{THIN}\n  {title}\n{THIN}")


def save_mermaid(content: str, filename: str) -> str:
    """Mermaid 내용을 ~/files/ 에 저장하고 경로를 반환한다."""
    path = os.path.join(MERMAID_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ═════════════════════════════════════════════════════════════════════════
# Planner 로그 캡쳐
# ═════════════════════════════════════════════════════════════════════════
#
#  FlowForge의 ExecutionEngine은 autonomous / hybrid 모드로 돌 때 아래 라인을
#  `flowforge.execution.engine` 로거(INFO)로 남긴다:
#
#      "planner selected N nodes for mode=autonomous: {node_id_1, node_id_2, ...}"
#
#  예제 스크립트 레벨에서는 이 로그를 핸들러로 캡쳐해 두었다가 실행 후
#  "어떤 노드가 Planner에 의해 선택되었는지"를 사용자에게 보여준다.
#  ( LLM을 예제 스크립트가 직접 호출하지 않는다는 점이 핵심이다. )


class PlannerCaptureHandler(logging.Handler):
    """ExecutionEngine / LLMPlanner 로그 메시지를 수집한다."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def reset(self) -> None:
        self.records = []

    def planner_selection_line(self) -> str | None:
        """엔진이 Planner 결과를 로그로 남긴 가장 마지막 라인을 찾는다."""
        for rec in reversed(self.records):
            msg = rec.getMessage()
            if rec.name == "flowforge.execution.engine" and "planner selected" in msg:
                return msg
        return None

    def planner_invalid_routes(self) -> list[str]:
        """Planner가 반환한 무효 route 경고들을 수집한다."""
        warnings: list[str] = []
        for rec in reversed(self.records):
            msg = rec.getMessage()
            if rec.name == "flowforge.planner.llm_planner" and "invalid route" in msg:
                warnings.append(msg)
        return warnings

    def planner_failed(self) -> str | None:
        """LLM 호출이 실패해 deterministic 로 폴백한 경우 경고 메시지 반환."""
        for rec in reversed(self.records):
            if rec.name == "flowforge.execution.engine" and "planning failed" in rec.getMessage():
                return rec.getMessage()
        return None


def install_planner_capture() -> PlannerCaptureHandler:
    handler = PlannerCaptureHandler()
    # engine 로그와 planner 모듈 로그를 동시에 잡는다.
    for logger_name in (
        "flowforge.execution.engine",
        "flowforge.planner.llm_planner",
    ):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    return handler


# ═════════════════════════════════════════════════════════════════════════
# Autonomous 실행
# ═════════════════════════════════════════════════════════════════════════

async def run_autonomous(
    engine,
    user_request: str,
    all_nodes,
    mermaid_filename: str,
    planner_log: PlannerCaptureHandler,
):
    """FlowForge의 built-in autonomous 모드로 유저 요청을 실행한다.

    예제 스크립트는 LLM을 직접 호출하지 않는다 ― FlowForge.compile()로 만든
    engine 에 planning_mode="autonomous" 만 넘기면 내장 LLMPlanner가
    doc 메타데이터를 참조해 실행할 노드를 선택한다.
    """
    from flowforge.schema.dag import NodeType

    sub("실행 — planning_mode=\"autonomous\"")
    planner_log.reset()
    t0 = time.perf_counter()
    try:
        result, trace = await engine.run_traced(
            user_request,
            planning_mode="autonomous",
        )
    except Exception as e:
        print(f"  [FAIL] 실행 중 예외 발생: {type(e).__name__}: {e}")
        return None, None

    elapsed = (time.perf_counter() - t0) * 1000

    # Planner 가 어떤 Flow를 선택했는지 (로그에서 복구)
    sub("Planner 결정 (Flow-level)")
    planner_line = planner_log.planner_selection_line()
    fail_line = planner_log.planner_failed()
    invalid_routes = planner_log.planner_invalid_routes()
    if fail_line:
        print(f"  ⚠ LLM Planner 실패 → deterministic 폴백: {fail_line}")
    if planner_line:
        print(f"  {planner_line}")
    else:
        print("  (로그에서 planner 결정 라인을 찾지 못함 — deterministic 으로 실행되었을 수 있음)")
    if invalid_routes:
        print(f"\n  무효 route 경고 ({len(invalid_routes)}건):")
        for w in invalid_routes:
            print(f"    · {w}")

    # 실제 실행된 노드/step 요약
    sub("실행 결과")
    executed_steps = result.get("trail", []) if isinstance(result, dict) else []
    print(f"  실행된 step ({len(executed_steps)}개): "
          f"{' -> '.join(executed_steps) if executed_steps else '(없음)'}")
    print(f"  실행된 노드 ({len(trace.executed_node_ids)}개), 소요시간: {elapsed:.0f}ms")

    all_flow_ids = {n.id for n in all_nodes if n.type == NodeType.FLOW}
    executed_flow_ids = {nid for nid in trace.executed_node_ids if nid in all_flow_ids}
    skipped_flows = all_flow_ids - executed_flow_ids
    print(f"\n  Flow 커버리지: {len(executed_flow_ids)}/{len(all_flow_ids)} 실행")
    if skipped_flows:
        print(f"  건너뛴 Flow (상위 5개):")
        for fid in sorted(skipped_flows)[:5]:
            print(f"    · {fid}")
        if len(skipped_flows) > 5:
            print(f"    ... 외 {len(skipped_flows) - 5}개")

    # Mermaid 비교 다이어그램 저장
    mermaid_content = engine.compare_mermaid(trace)
    path = save_mermaid(mermaid_content, mermaid_filename)
    print(f"\n  Mermaid 저장 → {path}")
    return executed_steps, trace


# ═════════════════════════════════════════════════════════════════════════
# 메인 실행
# ═════════════════════════════════════════════════════════════════════════

async def main():
    os.makedirs(MERMAID_DIR, exist_ok=True)
    from flowforge.schema.dag import NodeType

    # Planner 로그 캡쳐 핸들러 설치 (autonomous 모드가 무엇을 결정했는지
    # 외부에서 관찰하기 위한 유일한 수단).
    planner_log = install_planner_capture()

    # ── Phase 1: 컴파일 ──────────────────────────────────────────────────
    header("Phase 1 — Compile")
    engine = FlowForge.compile(EnterprisePlatformAgent)
    all_nodes = engine.dag.get_all_nodes()
    print(f"\n  총 {len(all_nodes)}개 노드 컴파일 완료")
    print()

    type_counts: dict[str, int] = {}
    max_depth = 0
    for n in all_nodes:
        type_counts[n.type.value] = type_counts.get(n.type.value, 0) + 1
        depth = n.id.count(".")
        max_depth = max(max_depth, depth)

    for t, c in sorted(type_counts.items()):
        print(f"    {t:6}: {c}개")
    print(f"    최대 깊이: {max_depth} 단계")

    flow_nodes = [n for n in all_nodes if n.type == NodeType.FLOW]
    print(f"\n  Flow 목록 ({len(flow_nodes)}개):")
    for n in flow_nodes:
        indent = "    " + "  " * (n.id.count(".") - 1)
        print(f"{indent}{n.id}")

    dag_md = engine.mermaid()
    path = save_mermaid("```mermaid\n" + dag_md + "\n```\n", "full_dag.md")
    print(f"\n  전체 DAG Mermaid 저장 → {path}")

    # ── Phase 2: Doc 생성 (autonomous planner 가 참조할 메타데이터) ──────
    #
    #   autonomous / hybrid 모드는 각 Flow의 doc(summary / capabilities /
    #   preconditions) 을 근거로 경로를 결정한다.
    #
    #   planning_only=True → GLOBAL + FLOW 노드만 doc 생성 (Task/Step 제외).
    #   Planner는 Flow 단위로만 경로를 선택하므로 Task/Step의 doc은 불필요하다.
    #   또한 doc 생성이 병렬로 수행되어 전체 소요시간이 크게 단축된다.

    header("Phase 2 — Generate Docs (for AI Planner)")
    all_flow_count = len([n for n in all_nodes if n.type == NodeType.FLOW])
    print(f"\n  Flow 노드만 doc 생성 (planning_only=True): {all_flow_count + 1}개")
    print(f"  (Task/Step {len(all_nodes) - all_flow_count - 1}개는 건너뜀 — Planner에 불필요)")
    print(f"  병렬 LLM 호출 (concurrency=5)로 속도 최적화 중...")
    t0 = time.perf_counter()
    try:
        docs = await engine.generate_docs(planning_only=True)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  doc 생성 완료: {len(docs)}개 노드, 소요시간: {elapsed:.0f}ms")
        # 샘플 3개만 미리보기
        print("\n  Doc 샘플 (최대 3개):")
        for node_id, doc in list(docs.items())[:3]:
            summary = getattr(doc, "summary", "")[:100]
            print(f"    · {node_id}")
            print(f"        summary: {summary}")
    except Exception as e:
        print(f"  [WARN] doc 생성 중 일부/전체 실패 ({type(e).__name__}: {e})")
        print(f"         → fallback doc(프롬프트 첫 200자)으로 autonomous 모드가 동작합니다.")

    # ── Phase 3: Autonomous 실행 테스트 ─────────────────────────────────
    #
    #   같은 engine, 같은 DAG. 유저 요청만 바꿔서 planning_mode="autonomous"
    #   로 실행 → 내장 LLMPlanner 가 경로(서브트리)를 자율적으로 고른다.

    test_cases = [
        {
            "label": "특정 작업 (깊은 단일 서브트리 기대)",
            "request": "CSV 파일을 파싱해서 데이터를 추출해줘",
        },
        {
            "label": "여러 도메인에 걸친 요청 (폭넓은 서브트리 기대)",
            "request": (
                "CI 파이프라인으로 빌드하고 테스트한 다음, "
                "프로덕션에 배포하고 모니터링도 설정해줘"
            ),
        },
        {
            "label": "ML 특화 요청 (중간 깊이 기대)",
            "request": "모델을 학습시키고 평가까지 해줘. 하이퍼파라미터 튜닝도 포함해서.",
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        header(f"Autonomous Test {i} — {tc['label']}")
        print(f"\n  유저 요청: \"{tc['request']}\"")
        await run_autonomous(
            engine,
            tc["request"],
            all_nodes,
            f"autonomous_test_{i}.md",
            planner_log,
        )

    # ── Phase 4: 요약 ───────────────────────────────────────────────────
    header("Summary")
    print(f"\n  총 노드: {len(all_nodes)}개")
    print(f"  총 Flow: {len(flow_nodes)}개")
    print(f"  최대 깊이: {max_depth} 단계")
    print(f"  Autonomous 테스트: {len(test_cases)}건")
    print()
    print(f"  실행 흐름:")
    print(f"    1. FlowForge.compile(EnterprisePlatformAgent)")
    print(f"    2. await engine.generate_docs()                  # 노드별 doc(LLM)")
    print(f"    3. await engine.run_traced(req, planning_mode=\"autonomous\")")
    print(f"       └─ 내장 LLMPlanner 가 doc + 유저 요청으로 실행할 노드 결정")
    print()
    print(f"  Mermaid 파일 저장 위치: {MERMAID_DIR}/")
    print(f"    - full_dag.md            (전체 DAG)")
    print(f"    - autonomous_test_1~{len(test_cases)}.md  (autonomous 실행 결과 비교)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
