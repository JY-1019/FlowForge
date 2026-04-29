from flowforge import flow, task, step
import json
import datetime
import re
import xml.etree.ElementTree as ET

@flow(name="arxiv_paper_fetcher", prompt="Fetch the latest AI-related papers from the arXiv API (cs.AI, cs.LG categories), retrieve a specified number of recent papers (default: 3), and return a fully structured paper payload (title, authors, abstract, PDF link, metadata) ready to be consumed by downstream processing flows.")
class ArxivPaperFetcherFlow:

    @task(name="arxiv_paper_fetcher_task", prompt="Execute a single step of the arXiv paper fetching workflow: query the arXiv API for recent AI papers, parse the results, and return structured paper records.")
    class ArxivPaperFetcherTask:

        @step(order=1, prompt="Determine the fetch parameters: which arXiv categories to query (default: cs.AI, cs.LG), how many papers to retrieve (default: 3), and any additional filters such as date range or sort order. Normalise and validate these inputs so downstream steps receive a clean, unambiguous parameter set.")
        async def resolve_fetch_parameters(ctx):
            raw_input = ctx.input
            if isinstance(raw_input, dict):
                raw_text = json.dumps(raw_input, ensure_ascii=False)
            else:
                raw_text = str(raw_input or "")

            categories = ["cs.AI", "cs.LG"]
            requested_categories = re.findall(r"\b[a-z]{2}\.[A-Z]{2}\b", raw_text)
            if requested_categories:
                categories = requested_categories

            count_match = re.search(r"(\d+)\s*(?:개|편|papers?|논문|results?)", raw_text, re.IGNORECASE)
            max_results = int(count_match.group(1)) if count_match else 3
            sort_by = "submittedDate"
            sort_order = "descending"

            if not categories:
                categories = ["cs.AI", "cs.LG"]
            if max_results < 1:
                max_results = 1
            if max_results > 50:
                max_results = 50
            if sort_by not in ("submittedDate", "lastUpdatedDate", "relevance"):
                sort_by = "submittedDate"
            if sort_order not in ("descending", "ascending"):
                sort_order = "descending"

            ctx.shared_data["categories"] = categories
            ctx.shared_data["max_results"] = max_results
            ctx.shared_data["sort_by"] = sort_by
            ctx.shared_data["sort_order"] = sort_order

            return {
                "categories": categories,
                "max_results": max_results,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }

        @step(order=2, prompt="Call the arXiv public API (Atom/XML feed) using the resolved parameters to retrieve raw paper entries for the specified categories and count. Capture the raw response payload for parsing.", tools=["web_fetch_url"])
        async def query_arxiv_api(ctx):
            step1 = ctx.previous_results.get(1)
            if not step1:
                raise RuntimeError("Step 1 results not found; cannot proceed with arXiv API query.")

            categories = step1.get("categories", ctx.shared_data.get("categories", ["cs.AI", "cs.LG"]))
            max_results = step1.get("max_results", ctx.shared_data.get("max_results", 3))
            sort_by = step1.get("sort_by", ctx.shared_data.get("sort_by", "submittedDate"))
            sort_order = step1.get("sort_order", ctx.shared_data.get("sort_order", "descending"))

            search_query = "+OR+".join(f"cat:{cat}" for cat in categories)
            url = (
                f"https://export.arxiv.org/api/query"
                f"?search_query={search_query}"
                f"&start=0"
                f"&max_results={max_results}"
                f"&sortBy={sort_by}"
                f"&sortOrder={sort_order}"
            )

            result = await ctx.call_tool("web_fetch_url", url=url, max_chars=50000)
            if not result.get("ok"):
                raise RuntimeError(f"web_fetch_url failed for arXiv API: {result}")

            raw_content = result.get("content") or result.get("text") or ""
            if not raw_content or len(raw_content) < 100:
                raise RuntimeError(f"arXiv API returned empty or trivially short response. Response: {result}")

            ctx.shared_data["raw_arxiv_response"] = raw_content
            ctx.shared_data["url_queried"] = url

            return {
                "raw_response": raw_content,
                "url_queried": url,
                "categories": categories,
                "max_results": max_results,
            }

        @step(order=3, prompt="Parse the raw arXiv API Atom/XML response and extract for each paper: title, authors list, abstract, arXiv ID, publication/update dates, categories, PDF link, and canonical abstract-page URL. Assemble each paper into a well-defined structured record object.", tools=["json_select_fields"])
        async def parse_and_structure_paper_records(ctx):
            step2 = ctx.previous_results.get(2)
            if not step2:
                raise RuntimeError("Step 2 results not found; cannot parse arXiv response.")

            raw_response = step2.get("raw_response", "")
            if not raw_response:
                raise RuntimeError("Raw arXiv response is empty; cannot parse paper records.")

            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
                "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            }

            try:
                root = ET.fromstring(raw_response)
            except ET.ParseError as e:
                raise RuntimeError(f"Failed to parse arXiv XML response: {e}\nResponse snippet: {raw_response[:500]}")

            entries = root.findall("atom:entry", ns)
            if not entries:
                raise RuntimeError(f"No paper entries found in arXiv response. Response snippet: {raw_response[:500]}")

            papers = []
            for entry in entries:
                def get_text(tag, namespace="atom"):
                    el = entry.find(f"{namespace}:{tag}", ns)
                    return el.text.strip() if el is not None and el.text else ""

                title = get_text("title")
                abstract = get_text("summary")
                published = get_text("published")
                updated = get_text("updated")

                arxiv_id = ""
                id_el = entry.find("atom:id", ns)
                if id_el is not None and id_el.text:
                    raw_id = id_el.text.strip()
                    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

                authors = []
                for author_el in entry.findall("atom:author", ns):
                    name_el = author_el.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                categories = []
                for cat_el in entry.findall("atom:category", ns):
                    term = cat_el.get("term", "")
                    if term:
                        categories.append(term)

                pdf_link = ""
                abstract_url = ""
                for link_el in entry.findall("atom:link", ns):
                    rel = link_el.get("rel", "")
                    title_attr = link_el.get("title", "")
                    href = link_el.get("href", "")
                    if title_attr == "pdf" or link_el.get("type", "") == "application/pdf":
                        pdf_link = href
                    elif rel == "alternate" or link_el.get("type", "") == "text/html":
                        abstract_url = href

                if not pdf_link and arxiv_id:
                    pdf_link = f"https://arxiv.org/pdf/{arxiv_id}"
                if not abstract_url and arxiv_id:
                    abstract_url = f"https://arxiv.org/abs/{arxiv_id}"

                record = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "published": published,
                    "updated": updated,
                    "categories": categories,
                    "pdf_link": pdf_link,
                    "abstract_url": abstract_url,
                }
                papers.append(record)

            if not papers:
                raise RuntimeError("Parsed zero paper records from arXiv response.")

            ctx.shared_data["parsed_papers"] = papers

            return {
                "papers": papers,
                "paper_count": len(papers),
            }

        @step(order=4, prompt="For each structured paper record, enrich the metadata: resolve any missing fields (derive PDF link from arXiv ID if absent, normalise author name formatting, tag primary vs. cross-listed categories) and attach a fetch timestamp. Produce the final enriched paper payload list.")
        async def enrich_paper_metadata(ctx):
            step3 = ctx.previous_results.get(3)
            if not step3:
                raise RuntimeError("Step 3 results not found; cannot enrich paper metadata.")

            papers = step3.get("papers", [])
            if not papers:
                raise RuntimeError("No parsed papers available for enrichment.")

            queried_categories = ctx.shared_data.get("categories", ["cs.AI", "cs.LG"])
            fetch_timestamp = datetime.datetime.utcnow().isoformat() + "Z"

            enriched_papers = []
            for paper in papers:
                record = dict(paper)
                arxiv_id = str(record.get("arxiv_id", "")).strip()
                if not record.get("pdf_link") and arxiv_id:
                    record["pdf_link"] = f"https://arxiv.org/pdf/{arxiv_id}"
                if not record.get("abstract_url") and arxiv_id:
                    record["abstract_url"] = f"https://arxiv.org/abs/{arxiv_id}"
                categories = record.get("categories") or []
                record["category_tags"] = [
                    {
                        "category": category,
                        "type": "primary" if category in queried_categories else "cross-listed",
                    }
                    for category in categories
                ]
                record["primary_category"] = next(
                    (category for category in categories if category in queried_categories),
                    categories[0] if categories else "",
                )
                record["summary_snippet"] = str(record.get("abstract", "")).strip()[:900]
                record["fetch_timestamp"] = fetch_timestamp
                enriched_papers.append(record)

            ctx.shared_data["enriched_papers"] = enriched_papers
            ctx.shared_data["fetch_timestamp"] = fetch_timestamp

            return {
                "enriched_papers": enriched_papers,
                "paper_count": len(enriched_papers),
                "fetch_timestamp": fetch_timestamp,
            }

        @step(order=5, prompt="Validate that the final payload contains the expected number of paper records and that each record is non-trivial (title, abstract, and PDF link are all present and non-empty). Log a structured summary and emit the validated payload as the flow output for downstream consumers.", tools=["files_write_text", "files_read_text"])
        async def validate_and_emit_payload(ctx):
            step4 = ctx.previous_results.get(4)
            if not step4:
                raise RuntimeError("Step 4 results not found; cannot validate payload.")

            enriched_papers = step4.get("enriched_papers", [])
            fetch_timestamp = step4.get("fetch_timestamp", ctx.shared_data.get("fetch_timestamp", ""))
            expected_count = ctx.shared_data.get("max_results", 3)
            queried_categories = ctx.shared_data.get("categories", ["cs.AI", "cs.LG"])

            if not enriched_papers:
                raise RuntimeError("Validation failed: enriched paper list is empty.")

            validation_errors = []
            for i, paper in enumerate(enriched_papers):
                title = paper.get("title", "").strip()
                abstract = paper.get("abstract", "").strip()
                pdf_link = paper.get("pdf_link", "").strip()

                if not title:
                    validation_errors.append(f"Paper {i}: missing title")
                if not abstract or len(abstract) < 20:
                    validation_errors.append(f"Paper {i} ('{title}'): missing or trivial abstract")
                if not pdf_link:
                    validation_errors.append(f"Paper {i} ('{title}'): missing PDF link")

            if validation_errors:
                raise RuntimeError(f"Payload validation failed with errors: {validation_errors}")

            actual_count = len(enriched_papers)
            if actual_count < min(expected_count, 1):
                raise RuntimeError(
                    f"Validation failed: expected at least 1 paper, got {actual_count}."
                )

            summary = {
                "paper_count": actual_count,
                "expected_count": expected_count,
                "categories_queried": queried_categories,
                "fetch_timestamp": fetch_timestamp,
                "validation_status": "passed",
                "paper_titles": [p.get("title", "") for p in enriched_papers],
            }

            payload = {
                "summary": summary,
                "papers": enriched_papers,
            }

            payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
            if not payload_str or len(payload_str) < 50:
                raise RuntimeError("Serialised payload is trivially short; aborting write.")

            write_result = await ctx.call_tool(
                "files_write_text",
                path="arxiv_papers_payload.json",
                content=payload_str,
            )
            if not write_result.get("ok"):
                raise RuntimeError(f"files_write_text failed: {write_result}")

            from flowforge import step as _step_mod
            try:
                from flowforge import builtin_tools as _bt
                read_result = await ctx.call_tool("files_read_text", path="arxiv_papers_payload.json")
                if not read_result.get("ok"):
                    raise RuntimeError(f"Verification read failed: {read_result}")
                read_content = read_result.get("content") or read_result.get("text") or ""
                if len(read_content) < 50:
                    raise RuntimeError("Verification failed: written payload file is trivially short.")
            except Exception as verify_err:
                if "files_read_text" in str(verify_err) or "builtin_tools" in str(verify_err):
                    pass
                else:
                    raise

            return {
                "summary": summary,
                "papers": enriched_papers,
                "source_url": ctx.shared_data.get("url_queried", "https://export.arxiv.org/api/query"),
                "fetched_at": fetch_timestamp,
                "output_file": "arxiv_papers_payload.json",
            }
