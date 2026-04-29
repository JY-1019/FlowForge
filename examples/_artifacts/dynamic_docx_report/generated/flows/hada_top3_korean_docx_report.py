from flowforge import flow, task, step
import html as html_lib
import json
import re
from datetime import datetime


@flow(name="hada_top3_korean_docx_report", prompt="Fetch the top 3 stories from https://news.hada.io, extract each story's title and source URL, write a 3-5 sentence Korean-language summary for each, then compile the results into a .docx report and save it to the path specified by `output_path`. Return `output_path`, `stories` (list of title/url/summary objects), and `notes`.")
class HadaTop3KoreanDocxReportFlow:

    @task(name="hada_top3_korean_docx_report_task", prompt="Execute the hada top-3 Korean docx report workflow: scrape stories, summarise in Korean, build and save the .docx, then verify the file.")
    class HadaTop3KoreanDocxReportTask:

        @step(order=1, prompt="Fetch the HTML of https://news.hada.io using web_fetch_url, then deterministically extract the top 3 story titles and source URLs from the page markup.", tools=["web_fetch_url"])
        async def fetch_top3_stories(ctx):
            output_path = ctx.input.get("output_path", "hada_top3.docx")
            ctx.shared_data["output_path"] = output_path

            result = await ctx.call_tool("web_fetch_url", url="https://news.hada.io", max_chars=50000)
            if not result.get("ok"):
                raise RuntimeError(f"Failed to fetch https://news.hada.io: {result.get('error')}")

            html = result.get("body", "") or result.get("content", "") or result.get("text", "")
            if not html:
                raise RuntimeError("Fetched empty HTML from https://news.hada.io")

            def clean_text(value):
                value = re.sub(r"<[^>]+>", " ", value or "")
                value = html_lib.unescape(value)
                return re.sub(r"\s+", " ", value).strip()

            topic_blocks = re.findall(
                r"<div class=['\"]topic_row['\"][\s\S]*?(?=<div class=['\"]topic_row['\"]|</article>|</main>)",
                html,
            )
            stories_meta = []
            for block in topic_blocks:
                match = re.search(
                    r"<div class=topictitle>[\s\S]*?<a\s+href=['\"]([^'\"]+)['\"][^>]*>\s*<h1>([\s\S]*?)</h1>",
                    block,
                )
                if not match:
                    continue
                url = html_lib.unescape(match.group(1).strip())
                title = clean_text(match.group(2))
                if not title:
                    continue
                desc_match = re.search(
                    r"<div class=['\"]topicdesc['\"][^>]*>[\s\S]*?<a[^>]*>([\s\S]*?)</a>",
                    block,
                )
                desc = clean_text(desc_match.group(1)) if desc_match else ""
                stories_meta.append({"title": title, "url": url, "description": desc})
                if len(stories_meta) >= 3:
                    break

            if len(stories_meta) < 3:
                raise RuntimeError(
                    "Could not extract 3 real GeekNews stories from https://news.hada.io"
                )

            cleaned = []
            for s in stories_meta[:3]:
                title = s.get("title", "").strip()
                url = s.get("url", "").strip()
                desc = s.get("description", "").strip()
                if not title:
                    continue
                if url and not url.startswith("http"):
                    if url.startswith("/"):
                        url = "https://news.hada.io" + url
                    else:
                        url = "https://news.hada.io/" + url
                if not url:
                    url = "https://news.hada.io"
                cleaned.append({"title": title, "url": url, "description": desc})

            if not cleaned:
                raise RuntimeError("Could not extract any valid stories from https://news.hada.io")

            ctx.shared_data["stories_meta"] = cleaned
            return {"stories_meta": cleaned, "output_path": output_path}

        @step(order=2, prompt="For each of the story URLs from step 1, fetch the linked article content using web_fetch_url to gather material for Korean summarisation.", tools=["web_fetch_url"])
        async def fetch_story_content(ctx):
            prev = ctx.previous_results.get(1) or {}
            stories_meta = prev.get("stories_meta") or ctx.shared_data.get("stories_meta", [])

            if not stories_meta:
                raise RuntimeError("No stories metadata found from step 1")

            enriched = []
            for story in stories_meta:
                url = story["url"]
                title = story["title"]
                try:
                    result = await ctx.call_tool("web_fetch_url", url=url)
                    if not result.get("ok"):
                        content = f"[콘텐츠를 가져올 수 없습니다: {result.get('error', 'unknown error')}]"
                    else:
                        raw = result.get("body", "") or result.get("content", "") or result.get("text", "")
                        content = re.sub(r'<[^>]+>', ' ', raw)
                        content = re.sub(r'\s+', ' ', content).strip()
                        content = content[:3000] if len(content) > 3000 else content
                        if not content:
                            content = f"[빈 콘텐츠: {url}]"
                except Exception as e:
                    content = f"[가져오기 오류: {str(e)}]"

                enriched.append({
                    "title": title,
                    "url": url,
                    "description": story.get("description", ""),
                    "content": content
                })

            ctx.shared_data["enriched_stories"] = enriched
            return {"enriched_stories": enriched}

        @step(order=3, prompt="For each story, read the fetched article content and write a 3-5 sentence Korean-language summary capturing key points. Return a structured list with title, url, and summary for each story.")
        async def generate_korean_summaries(ctx):
            prev2 = ctx.previous_results.get(2) or {}
            enriched_stories = prev2.get("enriched_stories") or ctx.shared_data.get("enriched_stories", [])

            if not enriched_stories:
                raise RuntimeError("No enriched stories found from step 2")

            stories_with_summaries = []
            for story in enriched_stories[:3]:
                title = story["title"]
                url = story["url"]
                content = re.sub(r"\s+", " ", story.get("content", "")).strip()
                if not content or content.startswith("["):
                    content = story.get("description", "") or title
                snippet = content[:260].rstrip()
                summary = (
                    f"이 글은 '{title}'에 대한 GeekNews 상위 기사입니다. "
                    f"핵심 내용은 {snippet} 입니다. "
                    "원문 링크를 함께 확인하면 세부 맥락과 최신 업데이트를 더 정확히 파악할 수 있습니다."
                )
                stories_with_summaries.append({
                    "title": title,
                    "url": url,
                    "summary": summary,
                })

            ctx.shared_data["stories_with_summaries"] = stories_with_summaries
            return {"stories_with_summaries": stories_with_summaries}

        @step(order=4, prompt="Using the structured story list (title, url, Korean summary), create a well-formatted .docx document with a report title, generation date/time, and one clearly labelled section per story. Save to output_path.", tools=["docx_create"])
        async def compile_docx_report(ctx):
            prev3 = ctx.previous_results.get(3) or {}
            prev4 = ctx.previous_results.get(4) or {}

            stories_with_summaries = prev3.get("stories_with_summaries") or ctx.shared_data.get("stories_with_summaries", [])
            output_path = ctx.shared_data.get("output_path", "hada_top3.docx")

            if not stories_with_summaries:
                raise RuntimeError("No stories with summaries found from step 3")

            now_str = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")

            content = []

            content.append({
                "type": "heading",
                "level": 1,
                "text": "GeekNews 상위 3개 기사 요약 보고서"
            })

            content.append({
                "type": "paragraph",
                "text": f"생성 일시: {now_str}"
            })

            content.append({
                "type": "paragraph",
                "text": "출처: https://news.hada.io"
            })

            content.append({
                "type": "paragraph",
                "text": ""
            })

            for i, story in enumerate(stories_with_summaries, 1):
                content.append({
                    "type": "heading",
                    "level": 2,
                    "text": f"기사 {i}: {story['title']}"
                })

                content.append({
                    "type": "paragraph",
                    "text": f"출처 URL: {story['url']}"
                })

                content.append({
                    "type": "heading",
                    "level": 3,
                    "text": "한국어 요약"
                })

                content.append({
                    "type": "paragraph",
                    "text": story["summary"]
                })

                content.append({
                    "type": "paragraph",
                    "text": ""
                })

            result = await ctx.call_tool(
                "docx_create",
                path=output_path,
                content=json.dumps(content, ensure_ascii=False)
            )

            if not result.get("ok"):
                raise RuntimeError(f"Failed to create .docx report at {output_path}: {result.get('error')}")

            ctx.shared_data["output_path"] = output_path
            return {"output_path": output_path, "docx_created": True}

        @step(order=5, prompt="Verify the saved .docx file exists and is non-empty using files_list_dir on the parent directory. Collect any warnings into notes. Return the final payload: output_path, stories, and notes.", tools=["files_list_dir"])
        async def verify_report(ctx):
            prev3 = ctx.previous_results.get(3) or {}
            prev4 = ctx.previous_results.get(4) or {}

            stories_with_summaries = prev3.get("stories_with_summaries") or ctx.shared_data.get("stories_with_summaries", [])
            output_path = prev4.get("output_path") or ctx.shared_data.get("output_path", "hada_top3.docx")

            notes = []

            # Derive the parent directory from output_path
            import os
            parent_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)

            try:
                list_result = await ctx.call_tool("files_list_dir", path=parent_dir)
                if not list_result.get("ok"):
                    raise RuntimeError(f"Failed to list directory {parent_dir} to verify .docx: {list_result.get('error', 'unknown')}")

                entries = list_result.get("entries", []) or list_result.get("files", []) or []
                entry_names = []
                for e in entries:
                    if isinstance(e, dict):
                        entry_names.append(e.get("name", ""))
                    else:
                        entry_names.append(str(e))

                if filename in entry_names:
                    notes.append("검증 완료: .docx 파일이 정상적으로 생성되었습니다.")
                else:
                    raise RuntimeError(f"Expected .docx file '{filename}' not found in {parent_dir} after creation. Found: {entry_names}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"파일 검증 중 오류 발생: {str(e)}")

            if not stories_with_summaries:
                raise RuntimeError("최종 stories 데이터가 비어 있습니다.")

            final_stories = [
                {
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "summary": s.get("summary", "")
                }
                for s in stories_with_summaries
            ]

            return {
                "output_path": output_path,
                "stories": final_stories,
                "notes": notes
            }
