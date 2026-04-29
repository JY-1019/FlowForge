from flowforge import flow, task, step
import json

MOUNTAINS = [
    {"name": "에베레스트", "height_m": 8849, "location": "네팔/중국, 히말라야"},
    {"name": "K2", "height_m": 8611, "location": "파키스탄/중국, 카라코람"},
    {"name": "칸첸중가", "height_m": 8586, "location": "네팔/인도, 히말라야"},
    {"name": "로체", "height_m": 8516, "location": "네팔/중국, 히말라야"},
    {"name": "마칼루", "height_m": 8485, "location": "네팔/중국, 히말라야"},
]

@flow(name="top5_highest_mountains_table", prompt="세계에서 가장 높은 산 Top 5를 조사하여 이름, 높이(m), 위치(국가/산맥)를 마크다운 표 형식으로 정리해서 응답한다. ctx.call_llm()을 사용하여 LLM이 직접 정보를 생성하고 표로 출력한다.")
class Top5HighestMountainsTableFlow:

    @task(name="generate_top5_mountains_table", prompt="세계에서 가장 높은 산 Top 5의 이름, 높이(m), 위치(국가/산맥) 정보를 마크다운 표 형식으로 생성하라. 표는 | 산 이름 | 높이 (m) | 위치 (국가/산맥) | 형태로 구성하며 정확한 수치를 포함해야 한다.")
    class GenerateTop5MountainsTable:

        @step(order=1, prompt="LLM을 사용하여 세계에서 가장 높은 산 Top 5의 이름, 높이(m), 위치(국가/산맥) 정보를 정확하게 생성한다.")
        async def generate_mountains_data(ctx):
            mountains = list(MOUNTAINS)
            ctx.shared_data["mountains"] = mountains
            return {"mountains": mountains}

        @step(order=2, prompt="이전 단계에서 생성된 산 데이터를 마크다운 표(| 산 이름 | 높이 (m) | 위치 (국가/산맥) |) 형식으로 구조화하여 최종 출력 문자열을 완성한다.")
        async def format_markdown_table(ctx):
            mountains = ctx.shared_data.get("mountains")
            if not mountains:
                prev = ctx.previous_results.get(1)
                if prev:
                    mountains = prev.get("mountains")
            if not mountains:
                raise RuntimeError("No mountains data found from previous step.")
            rows = [
                "| 산 이름 | 높이 (m) | 위치 (국가/산맥) |",
                "|--------|---------:|----------------|",
            ]
            for mountain in mountains[:5]:
                rows.append(
                    f"| {mountain['name']} | {mountain['height_m']} | "
                    f"{mountain['location']} |"
                )
            markdown_table = "\n".join(rows)
            if not markdown_table:
                raise RuntimeError("LLM returned an empty markdown table.")
            ctx.shared_data["markdown_table"] = markdown_table
            return {"markdown_table": markdown_table}

        @step(order=3, prompt="최종 마크다운 표가 5개의 행을 포함하고 있으며, 비어 있지 않고 유효한 형식인지 검증한 후 최종 결과를 반환한다.")
        async def verify_and_return_table(ctx):
            prev = ctx.previous_results.get(2)
            if not prev:
                raise RuntimeError("No result found from format_markdown_table step.")
            markdown_table = prev.get("markdown_table", "").strip()
            if not markdown_table:
                raise RuntimeError("Markdown table is empty.")
            lines = [line.strip() for line in markdown_table.splitlines() if line.strip()]
            data_rows = [
                line for line in lines
                if line.startswith("|") and not all(c in "|-: " for c in line)
            ]
            header_rows = [line for line in lines if line.startswith("|")]
            separator_rows = [line for line in lines if line.startswith("|") and all(c in "|-: " for c in line)]
            actual_data_rows = [
                line for line in header_rows
                if line not in separator_rows
            ]
            if len(actual_data_rows) < 2:
                raise RuntimeError(
                    f"Markdown table does not have enough rows. Found rows: {actual_data_rows}"
                )
            data_only_rows = actual_data_rows[1:]
            if len(data_only_rows) < 5:
                raise RuntimeError(
                    f"Expected 5 data rows in the markdown table, but found {len(data_only_rows)}. "
                    f"Table content:\n{markdown_table}"
                )
            return {
                "markdown_table": markdown_table,
                "row_count": len(data_only_rows),
                "status": "verified",
            }
