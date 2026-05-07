"""Korean autonomous-routing quickstart for FlowForge.

Run:

    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/korean_autonomous_routing_quickstart.py "환불 규정이 궁금해요. 결제는 어제 했고 아직 서비스를 거의 쓰지 않았습니다."

This example is intentionally small:

* no dynamic generation options;
* no external tools;
* multiple static flows with long Korean prompts;
* autonomous routing through generated flow docs;
* Korean pass_criteria prompts on the final answer steps.

The first run generates planning docs for GLOBAL/FLOW nodes, then asks the
planner to choose the best flow for the user's Korean request.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from flowforge import FlowForge, LLMConfig, flow, global_config, step, task


def _llm_config_from_env() -> LLMConfig:
    provider = os.getenv("FLOWFORGE_PROVIDER", "").strip().lower()
    model = os.getenv("FLOWFORGE_MODEL", "").strip()
    kwargs = {"temperature": 0.2, "max_tokens": 2048}

    if provider == "openai":
        return LLMConfig.for_openai(model=model or "gpt-4o", **kwargs)
    if provider == "google":
        return LLMConfig.for_gemini(model=model or "gemini-2.0-flash", **kwargs)
    return LLMConfig.for_claude(model=model or "claude-sonnet-4-6", **kwargs)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("message", value))
    return str(value)


@global_config(
    prompt=(
        "너는 한국어 고객 지원 요청을 분류하고 처리하는 FlowForge 데모 에이전트다. "
        "사용자의 문장을 보고 가장 알맞은 업무 흐름 하나를 고르는 것이 중요하다. "
        "답변은 항상 한국어로 작성하고, 친절하지만 과장하지 않는다. "
        "정책이나 기술 안내를 말할 때는 확정적으로 말할 수 있는 내용과 "
        "추가 확인이 필요한 내용을 구분한다. 사용자가 바로 다음 행동을 알 수 "
        "있도록 마지막에는 짧은 체크리스트를 포함한다."
    ),
    llm_config=_llm_config_from_env(),
)
class KoreanSupportAgent:
    @flow(
        name="refund_policy",
        prompt=(
            "환불, 결제 취소, 구독 해지, 청구 오류, 결제일 착오처럼 돈과 결제에 "
            "관련된 문의를 처리한다. 사용자가 환불 가능 여부, 필요한 정보, 처리 "
            "순서, 예외 조건을 묻는 경우 이 흐름이 가장 적합하다. 답변은 고객이 "
            "불안하지 않도록 현재 상황을 요약한 뒤, 환불 검토 기준과 다음 행동을 "
            "차분하게 안내해야 한다."
        ),
    )
    class RefundPolicyFlow:
        @task(
            name="refund_answer",
            prompt=(
                "사용자의 결제/환불 문의를 읽고, 환불 가능성을 단정하지 않으면서도 "
                "검토에 필요한 기준을 구체적으로 정리한다. 결제 시점, 사용량, 약관상 "
                "예외, 영수증 또는 계정 정보 확인 필요성을 빠뜨리지 않는다."
            ),
        )
        class RefundAnswerTask:
            @step(order=1, prompt="환불 문의에서 핵심 사실과 빠진 정보를 정리한다.")
            async def extract(ctx):
                message = _text(ctx.input)
                return {
                    "category": "refund_policy",
                    "message": message,
                    "facts": [
                        "사용자가 결제 또는 환불 관련 도움을 요청했다.",
                        "정확한 처리 가능 여부는 계정/결제 기록 확인이 필요하다.",
                    ],
                }

            @step(
                order=2,
                prompt=(
                    "환불 정책 담당자처럼 답변한다. 친절하지만 법적/정책적 확정을 "
                    "피하고, 확인이 필요한 항목을 명확히 분리한다."
                ),
                pass_criteria=(
                    "최종 답변은 반드시 한국어여야 한다. 사용자의 상황 요약, 환불 검토 "
                    "기준, 사용자가 준비해야 할 정보, 다음 단계 체크리스트를 모두 포함해야 "
                    "한다. '무조건 환불됩니다'처럼 근거 없는 확정 표현은 쓰면 안 된다."
                ),
                pass_criteria_max_retries=1,
            )
            async def answer(ctx):
                return await ctx.call_llm(
                    "다음 환불 문의에 답변해줘.\n\n문의와 정리된 사실:\n{message}\n{facts}"
                )

    @flow(
        name="technical_troubleshooting",
        prompt=(
            "로그인 실패, 오류 메시지, 페이지가 열리지 않음, API 응답 실패, 설치 문제, "
            "브라우저 호환성처럼 기술적 문제를 해결하는 흐름이다. 사용자가 에러 상황을 "
            "설명하거나 재현 방법, 점검 순서, 임시 우회 방법을 원할 때 선택한다. "
            "답변은 비전문가도 따라 할 수 있도록 증상 확인, 원인 후보, 안전한 조치, "
            "추가 로그 요청 순서로 구성해야 한다."
        ),
    )
    class TechnicalTroubleshootingFlow:
        @task(
            name="troubleshooting_answer",
            prompt=(
                "기술 지원 담당자처럼 사용자의 문제를 구조화한다. 사용자가 바로 해볼 수 "
                "있는 낮은 위험의 점검부터 안내하고, 데이터를 삭제하거나 설정을 크게 바꾸는 "
                "조치는 마지막에 주의 문구와 함께 제안한다."
            ),
        )
        class TroubleshootingAnswerTask:
            @step(order=1, prompt="기술 문의에서 증상, 환경, 빠진 정보를 분리한다.")
            async def extract(ctx):
                message = _text(ctx.input)
                return {
                    "category": "technical_troubleshooting",
                    "message": message,
                    "known": ["사용자가 기술적 장애나 오류 해결을 원한다."],
                    "missing": ["오류 메시지 전문", "사용 환경", "재현 단계"],
                }

            @step(
                order=2,
                prompt=(
                    "기술 문제 해결 가이드 작성자처럼 답변한다. 초보 사용자도 따라 할 수 "
                    "있게 단계별로 쓰되, 위험한 조치는 주의 문구를 붙인다."
                ),
                pass_criteria=(
                    "최종 답변은 한국어여야 하며, 1) 증상 요약, 2) 우선 점검할 항목, "
                    "3) 안전한 해결 단계, 4) 추가로 보내주면 좋은 정보, 5) 다음 단계 "
                    "체크리스트를 포함해야 한다."
                ),
                pass_criteria_max_retries=1,
            )
            async def answer(ctx):
                return await ctx.call_llm(
                    "다음 기술 지원 문의에 대한 해결 가이드를 작성해줘.\n\n{message}\n{known}\n{missing}"
                )

    @flow(
        name="account_onboarding",
        prompt=(
            "신규 가입, 초기 설정, 팀 초대, 권한 부여, 첫 프로젝트 생성, 기본 사용법처럼 "
            "처음 제품을 쓰는 사용자를 안내하는 흐름이다. 사용자가 어디서부터 시작해야 "
            "하는지 모르거나 설정 순서, 권장 초기 구성, 실수 방지 팁을 묻는 경우 선택한다."
        ),
    )
    class AccountOnboardingFlow:
        @task(
            name="onboarding_answer",
            prompt=(
                "온보딩 매니저처럼 사용자의 목표를 먼저 정리하고, 처음 10분 안에 끝낼 수 "
                "있는 설정 순서를 안내한다. 너무 많은 기능을 한꺼번에 소개하지 말고, "
                "계정/팀/권한/첫 작업 순서로 자연스럽게 설명한다."
            ),
        )
        class OnboardingAnswerTask:
            @step(order=1, prompt="온보딩 문의의 목표와 필요한 준비물을 정리한다.")
            async def extract(ctx):
                message = _text(ctx.input)
                return {
                    "category": "account_onboarding",
                    "message": message,
                    "goal": "사용자가 제품을 처음 설정하고 시작하도록 돕는다.",
                }

            @step(
                order=2,
                prompt="처음 사용하는 고객에게 보내는 온보딩 안내문을 작성한다.",
                pass_criteria=(
                    "최종 답변은 한국어여야 한다. 시작 전 준비물, 10분 안에 할 수 있는 "
                    "설정 순서, 권한/팀 초대 관련 주의점, 첫 성공 경험을 만드는 작은 "
                    "과제를 포함해야 한다."
                ),
                pass_criteria_max_retries=1,
            )
            async def answer(ctx):
                return await ctx.call_llm(
                    "다음 신규 사용자 문의에 대한 온보딩 답변을 작성해줘.\n\n{message}\n{goal}"
                )

    @flow(
        name="business_report",
        prompt=(
            "사용자가 회의 보고, 주간 요약, 고객 VOC 정리, 의사결정 메모, 임원 보고용 "
            "요약처럼 구조화된 비즈니스 문서를 원할 때 선택하는 흐름이다. 단순 고객 지원 "
            "답변보다 문서 형식, 핵심 요약, 리스크, 액션 아이템 정리가 중요할 때 적합하다."
        ),
    )
    class BusinessReportFlow:
        @task(
            name="report_answer",
            prompt=(
                "비즈니스 분석가처럼 흩어진 요청을 보고서 형태로 재구성한다. 결론을 먼저 "
                "쓰고, 근거와 리스크, 후속 액션을 구분한다. 모르는 정보는 추정하지 말고 "
                "추가 확인 필요 항목으로 남긴다."
            ),
        )
        class ReportAnswerTask:
            @step(order=1, prompt="보고서 요청에서 목적, 독자, 필요한 산출물을 정리한다.")
            async def extract(ctx):
                message = _text(ctx.input)
                return {
                    "category": "business_report",
                    "message": message,
                    "format": "요약, 핵심 근거, 리스크, 액션 아이템",
                }

            @step(
                order=2,
                prompt="임원이나 팀 리더가 빠르게 읽을 수 있는 한국어 업무 보고서를 작성한다.",
                pass_criteria=(
                    "최종 답변은 한국어여야 한다. 제목, 3줄 요약, 핵심 근거, 리스크/주의점, "
                    "액션 아이템을 포함해야 한다. 확인되지 않은 수치나 사실을 꾸며내면 안 된다."
                ),
                pass_criteria_max_retries=1,
            )
            async def answer(ctx):
                return await ctx.call_llm(
                    "다음 요청을 업무 보고서 형태로 정리해줘.\n\n{message}\n{format}"
                )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "message",
        nargs="?",
        default=(
            "환불 규정이 궁금해요. 결제는 어제 했고 아직 서비스를 거의 쓰지 않았습니다."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["autonomous", "deterministic"],
        default="autonomous",
        help="autonomous는 planner가 flow를 고르고, deterministic은 모든 flow를 순서대로 실행합니다.",
    )
    args = parser.parse_args()

    engine = FlowForge.compile(KoreanSupportAgent)
    if args.mode == "autonomous":
        await engine.generate_docs(planning_only=True)

    result, trace = await engine.run_traced(args.message, planning_mode=args.mode)

    print("\n=== Result ===")
    print(result)
    print("\n=== Executed Nodes ===")
    for node in trace.nodes:
        if node.succeeded:
            print(f"{node.execution_order:02d}. {node.node_type:5s} {node.node_id}")

    print("\n=== Mermaid ===")
    print(engine.run_mermaid(trace))


if __name__ == "__main__":
    asyncio.run(main())
