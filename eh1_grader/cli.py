"""Grade a setlist from the command line and print a scorecard.

    uv run eh1-grade --list                                   # setlists in the traces
    uv run eh1-grade --setlist sl-2026-11-02-1819-4vtm \\
        --set ../duckdb-analytics-agent/evals/analytics-eval.json
    uv run eh1-grade --setlist … --set … --judge-model claude-haiku-5-5 --regrade

The traces are the agent's (--traces-db, default: the agent checkout next to this one).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from duckdb_agent import TracesServerOutdated, connect_traces_db
from duckdb_agent.config import API_MAX_RETRIES

from .grade import GraderSettings, grade_setlist
from .traces import create_tables, list_setlists, read_grades

DEFAULT_TRACES = str(Path(__file__).resolve().parent.parent.parent / "duckdb-analytics-agent" / "agent_traces.duckdb")


def scorecard(grades: list[dict], question_set: dict) -> str:
    """Per category and preset column: average CORRECT · CAREFUL · EFFICIENT, rules broken."""
    category = {q["id"]: q.get("category") or "-" for q in question_set["questions"]}
    names = {c["id"]: c["name"] for c in question_set.get("categories", [])}
    cells = defaultdict(list)
    columns = sorted({g["column"] or "?" for g in grades})
    for g in grades:
        cells[(category.get(g["question"], "-"), g["column"] or "?")].append(g)

    def avg(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return f"{sum(values) / len(values):.1f}" if values else " - "

    lines = [f"{'category':<34}" + "".join(f"{'col ' + c:>22}" for c in columns)]
    for cat in sorted({k[0] for k in cells}, key=lambda c: (c == "-", c)):
        label = f"{cat} · {names.get(cat, '')}"[:33]
        row = f"{label:<34}"
        for c in columns:
            rs = cells.get((cat, c), [])
            broken = sum(1 for r in rs if r["rules_passed"] is False)
            text = f"{avg(rs, 'correct')} {avg(rs, 'careful')} {avg(rs, 'efficient')}" + (f" ✕{broken}" if broken else "")
            row += f"{text:>22}"
        lines.append(row)
    lines.append("")
    for c in columns:
        rs = [g for g in grades if (g["column"] or "?") == c]
        totals = [r["correct"] + r["careful"] + r["efficient"] for r in rs if r["correct"] is not None and r["careful"] is not None]
        full = sum(1 for t in totals if t == 6)
        cost = sum(r["run_cost_usd"] or 0 for r in rs)
        per = f"${cost / full:.4f} per full-marks answer" if full else "no full-marks answers"
        lines.append(f"col {c}: {sum(totals) / len(totals) if totals else 0:.1f}/6 avg · {full}/{len(rs)} full marks · {per}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Grade a setlist's runs and print the scorecard.")
    parser.add_argument("--traces-db", default=DEFAULT_TRACES)
    parser.add_argument("--list", action="store_true", help="List the setlists in the traces and exit.")
    parser.add_argument("--setlist", help="The setlist ID (sl-…).")
    parser.add_argument("--set", help="The question set file the setlist ran.")
    parser.add_argument("--judge-model", default=GraderSettings.judge_model)
    parser.add_argument("--judge-effort", default=GraderSettings.judge_effort)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--regrade", action="store_true", help="Grade again even if graded with these settings.")
    args = parser.parse_args(argv)
    load_dotenv()

    try:
        traces = connect_traces_db(args.traces_db)
    except TracesServerOutdated as e:
        print(f"Traces unavailable: {e}", file=sys.stderr)
        return 1
    try:
        if args.list:
            create_tables(traces)
            for s in list_setlists(traces):
                print(f"{s['id']}  {s['question_set']} r{s['revision']}  {s['runs']} runs  "
                      f"{s['name'] or ''}  {s['started_at']}")
            return 0
        if not args.setlist or not args.set:
            parser.error("--setlist and --set are needed (or --list)")
        question_set = json.loads(Path(args.set).read_text())
        settings = GraderSettings(judge_model=args.judge_model, judge_effort=args.judge_effort)
        client = anthropic.Anthropic(max_retries=API_MAX_RETRIES)

        def progress(event):
            if event["event"] == "grade_start":
                print(f"Grading {event['to_grade']} of {event['runs']} runs "
                      f"({event['skipped']} already graded, {event['unmatched']} not in the set)…", file=sys.stderr)
            elif event["event"] == "graded":
                scores = f"{event['correct']} {event['careful']} {event['efficient']}"
                flag = "" if event["rules_passed"] is not False else "  rule ✕"
                err = f"  ({event['judge_error']})" if event.get("judge_error") else ""
                print(f"  {event['question']:<5} {event['column']}#{event['repeat']}  {scores}{flag}{err}", file=sys.stderr)
            elif event["event"] == "grade_error":
                print(f"  ! {event['error']}", file=sys.stderr)

        summary = grade_setlist(traces, question_set, args.setlist, settings, client, args.regrade, args.concurrency, progress)
        print(f"Graded {summary.graded}, skipped {summary.skipped}, judge cost ${summary.judge_cost_usd:.4f}", file=sys.stderr)
        print(scorecard(read_grades(traces, args.setlist, summary.grader_key), question_set))
        return 1 if summary.errors else 0
    finally:
        traces.close()


if __name__ == "__main__":
    sys.exit(main())
