"""Grading a setlist: every run, against its question, under one grader setting. And
grading one run (RUN mode): against a question, or a quick check you write for it.

For each run: the trace rules and EFFICIENT come from its traces; CORRECT and CAREFUL
from the judge (only when the question has a check). Each grade is written to
eval_grades as soon as it's done, so a stopped grading keeps what it finished, and
runs already graded with the same settings are skipped unless `regrade` is set.
"""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

import anthropic
from duckdb_agent import TracesDB

from .efficiency import DEFAULT_ITERATION_BUDGET, DEFAULT_WASTE_FOR_ONE, efficiency
from .judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL, JUDGE_VERSION, MAX_INSTRUCTIONS, judge
from .reference import References
from .rules import check_rules
from .traces import Run, create_tables, graded_run_ids, run_by_id, run_grades, setlist_runs, source_spec, write_grade

SUPPORTED_FORMAT_VERSION = 2


@dataclass
class GraderSettings:
    """What a grade depends on, besides the run. Changing any of it is a new grade."""
    name: str = "Standard rubric"
    revision: int = 1
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_effort: str = DEFAULT_JUDGE_EFFORT
    tolerance: str = "±0.5%"
    iteration_budget: int = DEFAULT_ITERATION_BUDGET
    # Your own notes to the judge, sent with every question (the Rubric flyout).
    instructions: str = ""
    # EFFICIENT: how much waste still scores 1 (the Efficient flyout).
    waste_for_one: int = DEFAULT_WASTE_FOR_ONE

    @classmethod
    def from_dict(cls, data: dict | None) -> "GraderSettings":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        settings = cls(**known)
        settings.instructions = str(settings.instructions or "").strip()[:MAX_INSTRUCTIONS]
        settings.iteration_budget = min(50, max(1, int(settings.iteration_budget)))
        settings.waste_for_one = min(10, max(0, int(settings.waste_for_one)))
        return settings


# Settings added after grades were already stored: left out of the key while at their
# defaults, so those grades keep their keys.
LATER_SETTINGS = {"instructions": "", "waste_for_one": DEFAULT_WASTE_FOR_ONE}


def grader_key(settings: GraderSettings, question_set: dict) -> str:
    """Grades are kept per grader setting, question set revision, and judge version."""
    grader = {k: v for k, v in asdict(settings).items() if not (k in LATER_SETTINGS and v == LATER_SETTINGS[k])}
    payload = {"grader": grader, "set": question_set["id"], "revision": question_set.get("revision", 1),
               "judge": JUDGE_VERSION}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def check_question_set(question_set: dict) -> None:
    """The essentials; the app's reader (packages/formats) is the full validator."""
    if question_set.get("format") != "dh-question-set":
        raise ValueError("not a Duck Harness question set")
    if not isinstance(question_set.get("formatVersion"), int) or question_set["formatVersion"] > SUPPORTED_FORMAT_VERSION:
        raise ValueError("this question set was saved by a newer version")
    if not question_set.get("id") or not isinstance(question_set.get("questions"), list):
        raise ValueError('a question set needs an "id" and "questions"')


def normalize_check(check: dict | None) -> dict:
    """Version 1 and 2 checks in one shape: sql as a list, expect as correct."""
    check = dict(check or {})
    if isinstance(check.get("sql"), str):
        check["sql"] = [check["sql"]]
    if "expect" in check and "correct" not in check:
        check["correct"] = check.pop("expect")
    return check


@dataclass
class Summary:
    setlist_run: str
    grader_key: str
    runs: int = 0
    graded: int = 0
    skipped: int = 0
    unmatched: int = 0
    judge_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False


def grade_run(run: Run, question: dict, question_set: dict, settings: GraderSettings, key: str,
              references: References, client: anthropic.Anthropic | None) -> dict:
    """One run's grade row (for eval_grades)."""
    check = normalize_check(question.get("check"))
    rules, rules_passed = check_rules(check.get("rules", []), run)
    eff, eff_reason = efficiency(run, settings.iteration_budget, settings.waste_for_one)
    row = {
        "run_id": run.run_id, "grader_key": key, "graded_at": datetime.now(),
        "setlist_run": run.labels.get("setlist_run"), "question_set": question_set["id"],
        "question_set_revision": question_set.get("revision", 1), "question": question["id"],
        "grader": asdict(settings), "efficient": eff, "efficient_reason": eff_reason,
        "rules": rules, "rules_passed": rules_passed,
    }
    judged = {k: check.get(k) for k in ("sql", "reference", "correct", "careful", "fail", "bonus", "tpchQuery")}
    if not any(judged.values()):
        return {**row, "correct_reason": "No check for this question: only Efficient and rules are graded."}
    source = question.get("source") or question_set.get("source")
    refs = [references.run(source, sql) for sql in check.get("sql", [])] if source else []
    row["reference"] = [{k: r[k] for k in ("sql", "rows", "columns", "preview", "error")} for r in refs]
    if client is None:
        return {**row, "judge_error": "no judge (no API client)"}
    verdict = judge(client, question, check, refs, run.answer, settings.judge_model, settings.judge_effort, settings.tolerance,
                    settings.instructions)
    return {
        **row,
        "correct": verdict.correct, "careful": verdict.careful,
        "correct_reason": verdict.correct_reason, "careful_reason": verdict.careful_reason,
        "judge_unsure": verdict.unsure, "judge_values": verdict.values,
        "judge_model": settings.judge_model, "judge_cost_usd": verdict.cost_usd, "judge_error": verdict.error,
    }


def grade_setlist(
    traces: TracesDB,
    question_set: dict,
    setlist_run: str,
    settings: GraderSettings | None = None,
    client: anthropic.Anthropic | None = None,
    regrade: bool = False,
    concurrency: int = 4,
    on_event: Callable[[dict], None] | None = None,
    cancel: threading.Event | None = None,
    run_ids: list[str] | None = None,
) -> Summary:
    """Grade every run of a setlist. `on_event` gets {"event": "grade_start" | "graded" |
    "grade_error", ...} as it goes; setting `cancel` stops after the runs in progress.
    With `run_ids`, only those runs are graded (again, if they were): one cell's repeats."""
    check_question_set(question_set)
    settings = settings or GraderSettings()
    key = grader_key(settings, question_set)
    emit = on_event or (lambda event: None)
    cancel = cancel or threading.Event()
    create_tables(traces)

    runs = setlist_runs(traces, setlist_run)
    questions = {q["id"]: q for q in question_set["questions"]}
    done = set() if regrade else graded_run_ids(traces, setlist_run, key)
    summary = Summary(setlist_run=setlist_run, grader_key=key, runs=len(runs))
    todo = []
    only = set(run_ids) if run_ids is not None else None
    for run in runs:
        if only is not None and run.run_id not in only:
            continue
        if run.question not in questions:
            summary.unmatched += 1
        elif only is None and run.run_id in done:
            summary.skipped += 1
        else:
            todo.append(run)
    emit({"event": "grade_start", "setlist_run": setlist_run, "grader_key": key, "runs": len(runs),
          "to_grade": len(todo), "skipped": summary.skipped, "unmatched": summary.unmatched})

    references = References()
    write_lock = threading.Lock()

    def one(run: Run) -> None:
        if cancel.is_set():
            return
        try:
            row = grade_run(run, questions[run.question], question_set, settings, key, references, client)
            with write_lock:
                write_grade(traces, row)
                summary.graded += 1
                summary.judge_cost_usd += row.get("judge_cost_usd") or 0.0
            emit({"event": "graded", "run_id": run.run_id, "question": run.question,
                  "column": run.labels.get("column"), "repeat": run.labels.get("repeat"),
                  "correct": row.get("correct"), "careful": row.get("careful"), "efficient": row["efficient"],
                  "rules_passed": row["rules_passed"], "judge_error": row.get("judge_error"),
                  "judge_cost_usd": row.get("judge_cost_usd")})
        except Exception as e:  # one bad run shouldn't stop the rest
            message = f"{run.question} ({run.run_id}): {type(e).__name__}: {e}"
            with write_lock:
                summary.errors.append(message)
            emit({"event": "grade_error", "run_id": run.run_id, "question": run.question, "error": message})

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(one, todo))
    finally:
        references.close()
    summary.cancelled = cancel.is_set()
    return summary


QUICK_SET = "quick"


def quick_key(settings: GraderSettings, check: dict, source: dict | None) -> str:
    """A quick check's grades are kept per grader setting and per check: change the check
    and it's a new grade, beside the old one."""
    content = hashlib.sha1(json.dumps({"check": check, "source": source}, sort_keys=True).encode()).hexdigest()[:10]
    return grader_key(settings, {"id": f"{QUICK_SET}:{content}", "revision": 1})


def clean_quick_check(check: dict | None) -> dict:
    """{correct, careful?, sql: [...]} from what the app sent; CORRECT IS is required."""
    check = check if isinstance(check, dict) else {}
    correct = str(check.get("correct") or "").strip()
    if not correct:
        raise ValueError('a quick check needs "correct": what a correct answer says')
    sql = check.get("sql") or []
    sql = [sql] if isinstance(sql, str) else sql
    sql = [str(q).strip() for q in sql if str(q).strip()]
    careful = str(check.get("careful") or "").strip()
    return {"correct": correct, **({"careful": careful} if careful else {}), **({"sql": sql} if sql else {})}


def grade_one(
    traces: TracesDB,
    run_id: str,
    settings: GraderSettings | None = None,
    client: anthropic.Anthropic | None = None,
    question_set: dict | None = None,
    question_id: str | None = None,
    quick_check: dict | None = None,
) -> dict:
    """Grade one run, against a question of `question_set` or against `quick_check`
    ({correct, careful, sql}; reference SQL runs on the run's own data). Writes the grade
    (keeping any override) and returns it as run_grades shows it."""
    settings = settings or GraderSettings()
    create_tables(traces)
    run = run_by_id(traces, run_id)
    if run is None:
        raise ValueError(f"no run {run_id!r} in the traces")
    if question_set is not None:
        check_question_set(question_set)
        question = next((q for q in question_set["questions"] if q.get("id") == question_id), None)
        if question is None:
            raise ValueError(f"no question {question_id!r} in {question_set['id']}")
        key = grader_key(settings, question_set)
        extra = {}
    else:
        check = clean_quick_check(quick_check)
        source = source_spec(run)
        key = quick_key(settings, check, source)
        question = {"id": None, "prompt": run.prompt, "check": check}
        question_set = {"id": QUICK_SET, "revision": None, "source": source, "questions": [question]}
        extra = {"quick_check": {**check, "source": source}}
    references = References()
    try:
        row = grade_run(run, question, question_set, settings, key, references, client)
    finally:
        references.close()
    if question_set["id"] == QUICK_SET and not question_set["source"] and quick_check and question["check"].get("sql"):
        row["reference"] = [{"sql": q, "rows": None, "columns": None, "preview": None,
                             "error": "This run didn't record its data (runs before the agent kept data_source)."}
                            for q in question["check"]["sql"]]
    write_grade(traces, {**row, **extra})
    return next(g for g in run_grades(traces, run_id) if g["grader_key"] == key)
