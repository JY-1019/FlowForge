"""FlowForge — Route Execution & Task Loop Example
===================================================

Demonstrates two features:

1. **Route execution** — run only specific flows or tasks:
       engine.run(data, route="analysis")          # single flow
       engine.run(data, route="analysis.classify")  # specific task
       engine.run(data, route=["analysis", "report"]) # multiple

2. **Task loop** — retry a task until a condition is met:
       @task(max_loops=5, loop_condition=lambda out: out["score"] > 0.8)

Usage:
    python examples/route_and_loop_example.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flowforge import global_config, flow, task, step, FlowForge

# ---------------------------------------------------------------------------
# Simulated "quality score" that improves each attempt
# ---------------------------------------------------------------------------

_attempt = 0


def _reset():
    global _attempt
    _attempt = 0


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

@global_config(prompt="Demo agent showing route + loop features")
class DemoAgent:

    @flow(name="analysis", prompt="Analyze input data")
    class AnalysisFlow:

        @task(name="classify", prompt="Classify the input")
        class ClassifyTask:
            @step(order=1, prompt="Run classification")
            async def classify(ctx):
                return {"category": "important", "input": ctx.input}

        @task(
            name="quality_check",
            prompt="Re-run until quality score exceeds threshold",
            max_loops=5,
            loop_condition=lambda out: out.get("score", 0) >= 0.8,
        )
        class QualityCheckTask:
            @step(order=1, prompt="Score the classification quality")
            async def score(ctx):
                global _attempt
                _attempt += 1
                # Simulate improving quality: 0.3, 0.5, 0.7, 0.9, ...
                score = round(0.1 + _attempt * 0.2, 1)
                return {"score": score, "attempt": _attempt, "data": ctx.input}

    @flow(name="report", prompt="Generate a report")
    class ReportFlow:

        @task(name="summarize", prompt="Summarize findings")
        class SummarizeTask:
            @step(order=1, prompt="Create summary")
            async def summarize(ctx):
                return {"summary": f"Report based on: {ctx.input}", "status": "done"}

    @flow(name="archive", prompt="Archive results")
    class ArchiveFlow:

        @task(name="store", prompt="Store to archive")
        class StoreTask:
            @step(order=1, prompt="Write to storage")
            async def store(ctx):
                return {"archived": True, "data": ctx.input}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SEP = "-" * 60


async def main():
    engine = FlowForge.compile(DemoAgent)
    print(f"Compiled {len(engine.dag.get_all_nodes())} nodes\n")

    # --- Example 1: Run ALL flows (default) ---
    print(SEP)
    print("1. Run ALL flows (no route)")
    print(SEP)
    _reset()
    result = await engine.run({"text": "hello world"})
    print(f"   Result: {result}\n")

    # --- Example 2: Route to a single flow ---
    print(SEP)
    print("2. Route to 'report' flow only")
    print(SEP)
    _reset()
    result, trace = await engine.run_traced({"text": "hello"}, route="report")
    print(f"   Result: {result}")
    print(f"   Executed nodes: {sorted(trace.executed_node_ids)}\n")

    # --- Example 3: Route to a specific task ---
    print(SEP)
    print("3. Route to 'analysis.classify' (specific task)")
    print(SEP)
    _reset()
    result, trace = await engine.run_traced({"text": "hello"}, route="analysis.classify")
    print(f"   Result: {result}")
    print(f"   Executed nodes: {sorted(trace.executed_node_ids)}\n")

    # --- Example 4: Task loop in action ---
    print(SEP)
    print("4. Run 'analysis' flow (quality_check loops until score >= 0.8)")
    print(SEP)
    _reset()
    result, trace = await engine.run_traced({"text": "data"}, route="analysis")
    print(f"   Result: {result}")
    print(f"   Attempts: {result.get('attempt', '?')} (score reached {result.get('score', '?')})")
    print(f"   Succeeded: {trace.succeeded}\n")

    # --- Example 5: Multiple routes ---
    print(SEP)
    print("5. Multiple routes: ['analysis', 'archive']")
    print(SEP)
    _reset()
    result, trace = await engine.run_traced({"text": "multi"}, route=["analysis", "archive"])
    print(f"   Result: {result}")
    executed = sorted(trace.executed_node_ids)
    print(f"   Executed: {executed}")
    has_report = any("report" in nid for nid in executed)
    print(f"   Report flow executed? {has_report}  (should be False)\n")


if __name__ == "__main__":
    asyncio.run(main())
