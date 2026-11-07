"""Drive EH-1's grader from the app over JSON Lines on stdin/stdout, like the agent's
duck-harness-serve.

    uv run eh1-serve --traces-db ../duckdb-analytics-agent/agent_traces.duckdb

Requests (each has an "id" and a "type"; every message names its request's "id"):
    {"type": "hello"}
        -> {"type": "hello", "protocol": 1, "version": "0.1.0", "traces_schema": "…",
            "grades_schema": "…", "defaults": {...GraderSettings}, "judge_models": [...],
            "scale": [{score, "2", "1", "0"}, ...]}  (the rubric, for the app to show)
        "version" is eh1-grader's; "traces_schema" is the agent traces schema it reads (from
        the duck-harness package it was built with), which must match the agent's;
        "grades_schema" names EH-1's own tables (eval_grades, eval_setlists).
    {"type": "setlists", "grader": {...}, "question_sets": [{"id", "revision"}, ...]}
        -> {"type": "setlists", "setlists": [{id, name, note, question_set, revision, runs,
            completed, cancelled, columns, repeats, concurrency, cost_usd, started_at, ended_at,
            presets: [{column, name, modified, config}], grading}, ...]}
        grading is {graded, total, other_settings} under these grader settings, with the
        question set revision the app would grade with (from "question_sets", the sets it
        has, in its order); it's left out without "grader".
    {"type": "rename", "setlist_run": "sl-…", "name": "…" | null, "note": "…" | null}
        -> {"type": "ok"}  (your name and note for a setlist, kept in eval_setlists; blank
        or null shows DH-1's again)
    {"type": "grade", "setlist_run": "sl-…", "question_set": {...the set's JSON},
     "grader": {...GraderSettings fields}, "regrade": false, "concurrency": 4}
        -> {"type": "accepted"}
        -> {"type": "event", "event": "grade_start" | "graded" | "grade_error", ...} (many)
        -> {"type": "result", "result": {setlist_run, grader_key, runs, graded, skipped,
            unmatched, judge_cost_usd, errors, cancelled}}
    {"type": "grades", "setlist_run": "sl-…", "question_set": {...}, "grader": {...}}
        -> {"type": "grades", "grader_key": "…", "grades": [{run_id, question, column, repeat,
            status, correct, careful, efficient, …reasons, rules, …}, ...]}
    {"type": "override", "run_id": "…", "question_set": {...}, "grader": {...},
     "override": {"correct": 0-2, "careful": 0-2, "why": "…"} | null}  -> {"type": "ok"}
        (null clears it; regrading keeps it)
    {"type": "stop", "grade": "<id of a grade or follow request>"}  -> {"type": "ok"}
"grade" also takes "run_ids": [...] to grade just those runs again (one scorecard cell).
"override" takes "grader_key" in place of "question_set" + "grader" (for quick checks).

Following DH-1 (the rack's AUTO-GRADE; see follow.py):
    {"type": "follow", "grader": {...}, "question_sets": [{...a set's JSON}, ...], "interval": 5}
        -> {"type": "accepted"}
        -> {"type": "event", "event": "follow_start" | "graded" | "grade_error" | "follow_skipped",
            "setlist_run": "sl-…", ...}  (as setlists' runs finish and are graded)
        -> {"type": "result", "result": {graded, judge_cost_usd, errors, cancelled}}  (after "stop")
    Only one follow runs at a time: a new one (new grader settings, say) replaces it.

RUN mode, one run at a time:
    {"type": "runs", "kind": "all" | "single" | "ab" | "setlist", "search": "…", "limit": 100}
        -> {"type": "runs", "runs": [{run_id, started_at, prompt, status, preset, labels,
            cost_usd, grade: {question_set, question, total, rules_passed} | null}, ...]}
    {"type": "run", "run_id": "…"}
        -> {"type": "run", "run": {run_id, prompt, answer, status, labels, preset, model,
            source, ...}, "grades": [every grade of it, newest first]}
    {"type": "grade_run", "run_id": "…", "grader": {...},
     "question_set": {...}, "question": "Q3"}            (against a question)
     or "quick_check": {"correct": "…", "careful": "…", "sql": ["…"]}   (against your check)
        -> {"type": "accepted"}, then {"type": "result", "grade": {...}}
Errors come back as {"type": "error", "error": "..."}. End of input stops and exits.
"""

import argparse
import dataclasses
import json
import sys
import threading
from datetime import datetime

import anthropic
from dotenv import load_dotenv
from importlib.metadata import PackageNotFoundError, version

from duckdb_agent import SCHEMA_VERSION, connect_traces_db
from duckdb_agent.config import API_MAX_RETRIES

from .follow import DEFAULT_INTERVAL_SECONDS, Follower, FollowTotals, grading_revision
from .grade import GraderSettings, check_question_set, grade_one, grade_setlist, grader_key
from .judge import JUDGE_PRICING, SCALE
from .traces import (
    GRADES_SCHEMA, MAX_NAME, MAX_NOTE, clean_text, create_tables, grading_status, list_runs, list_setlists, read_grades, rename_setlist,
    run_detail, run_grades, set_override, setlist_exists,
)

PROTOCOL_VERSION = 1


def grader_version() -> str:
    """The installed eh1-grader version, as hello reports it."""
    try:
        return version("eh1-grader")
    except PackageNotFoundError:
        return "unknown"


class Server:
    def __init__(self, stdin, stdout, traces_db: str, traces_server: bool = True, client=None):
        self.stdin, self.stdout = stdin, stdout
        self.traces_db, self.traces_server = traces_db, traces_server
        self._client = client
        self._write_lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()

    def traces(self):
        return connect_traces_db(self.traces_db, server=self.traces_server)

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=API_MAX_RETRIES)
        return self._client

    def send(self, message: dict) -> None:
        line = json.dumps(message, default=str)
        with self._write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def error(self, request_id, text: str) -> None:
        self.send({"id": request_id, "type": "error", "error": text})

    def handle(self, request: dict) -> bool:
        rid, kind = request.get("id"), request.get("type")
        if not isinstance(rid, str) or not rid:
            self.error(rid, 'every request needs a string "id"')
            return True
        try:
            if kind == "hello":
                self.send({"id": rid, "type": "hello", "protocol": PROTOCOL_VERSION,
                           "version": grader_version(), "traces_schema": SCHEMA_VERSION,
                           "grades_schema": GRADES_SCHEMA,
                           "defaults": dataclasses.asdict(GraderSettings()),
                           "judge_models": ["claude-haiku-5-5", *JUDGE_PRICING], "scale": SCALE})
            elif kind == "setlists":
                self.setlists(rid, request)
            elif kind == "rename":
                self.rename(rid, request)
            elif kind == "runs":
                traces = self.traces()
                try:
                    create_tables(traces)
                    runs = list_runs(traces, str(request.get("kind") or "all"), str(request.get("search") or ""),
                                     min(500, int(request.get("limit") or 100)))
                finally:
                    traces.close()
                self.send({"id": rid, "type": "runs", "runs": runs})
            elif kind == "run":
                traces = self.traces()
                try:
                    create_tables(traces)
                    run = run_detail(traces, str(request.get("run_id") or ""))
                    if run is None:
                        raise ValueError(f"no run {request.get('run_id')!r} in the traces")
                    grades = run_grades(traces, run["run_id"])
                finally:
                    traces.close()
                self.send({"id": rid, "type": "run", "run": run, "grades": grades})
            elif kind == "grade_run":
                self.start_grade_run(rid, request)
            elif kind == "grades":
                question_set = request.get("question_set") or {}
                check_question_set(question_set)
                key = grader_key(GraderSettings.from_dict(request.get("grader")), question_set)
                traces = self.traces()
                try:
                    create_tables(traces)
                    grades = read_grades(traces, request.get("setlist_run", ""), key)
                finally:
                    traces.close()
                self.send({"id": rid, "type": "grades", "grader_key": key, "grades": grades})
            elif kind == "grade":
                self.start_grade(rid, request)
            elif kind == "override":
                self.override(rid, request)
            elif kind == "follow":
                self.start_follow(rid, request)
            elif kind == "stop":
                with self._jobs_lock:
                    job = self._jobs.get(request.get("grade"))
                if job is None:
                    self.error(rid, f"no grading in progress with id {request.get('grade')!r}")
                else:
                    job["cancel"].set()
                    self.send({"id": rid, "type": "ok"})
            elif kind == "shutdown":
                self.stop_all()
                self.send({"id": rid, "type": "ok"})
                return False
            else:
                self.error(rid, f"unknown request type {kind!r}")
        except Exception as e:
            self.error(rid, f"{type(e).__name__}: {e}")
        return True

    def setlists(self, rid: str, request: dict) -> None:
        traces = self.traces()
        try:
            create_tables(traces)
            setlists = list_setlists(traces)
            if request.get("grader") is not None:
                settings = GraderSettings.from_dict(request.get("grader"))
                sets = [s for s in request.get("question_sets") or [] if isinstance(s, dict) and s.get("id")]
                keys = {s["id"]: grader_key(settings, {"id": s["question_set"], "revision": grading_revision(s, sets)})
                        for s in setlists}
                status = grading_status(traces, keys)
                for s in setlists:
                    s["grading"] = status[s["id"]]
        finally:
            traces.close()
        self.send({"id": rid, "type": "setlists", "setlists": setlists})

    def rename(self, rid: str, request: dict) -> None:
        setlist_run = request.get("setlist_run")
        if not isinstance(setlist_run, str) or not setlist_run:
            raise ValueError('"setlist_run" must be a setlist ID')
        traces = self.traces()
        try:
            create_tables(traces)
            if not setlist_exists(traces, setlist_run):
                raise ValueError(f"no setlist {setlist_run!r} in the traces")
            rename_setlist(traces, setlist_run, clean_text(request.get("name"), MAX_NAME),
                           clean_text(request.get("note"), MAX_NOTE))
        finally:
            traces.close()
        self.send({"id": rid, "type": "ok"})

    def override(self, rid: str, request: dict) -> None:
        """Your CORRECT and CAREFUL for one graded run, kept next to the judge's."""
        if isinstance(request.get("grader_key"), str) and request["grader_key"]:
            key = request["grader_key"]
        else:
            question_set = request.get("question_set") or {}
            check_question_set(question_set)
            key = grader_key(GraderSettings.from_dict(request.get("grader")), question_set)
        value = request.get("override")
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError('"override" must be {"correct", "careful", "why"} or null')
            for name in ("correct", "careful"):
                if value.get(name) not in (0, 1, 2):
                    raise ValueError(f'"{name}" must be 0, 1, or 2')
            value = {"correct": value["correct"], "careful": value["careful"],
                     "why": str(value.get("why") or "").strip()[:500], "at": datetime.now().isoformat(timespec="seconds")}
        traces = self.traces()
        try:
            create_tables(traces)
            if not set_override(traces, str(request.get("run_id", "")), key, value):
                raise ValueError("that run isn't graded with these settings")
        finally:
            traces.close()
        self.send({"id": rid, "type": "ok"})

    def start_grade_run(self, rid: str, request: dict) -> None:
        """Grade one run in the background (the judge takes a few seconds)."""
        run_id = request.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('"run_id" must be a run ID')
        question_set = request.get("question_set")
        if question_set is None and request.get("quick_check") is None:
            raise ValueError('grade_run needs "question_set" and "question", or "quick_check"')

        def work():
            traces = None
            try:
                traces = self.traces()
                grade = grade_one(traces, run_id, GraderSettings.from_dict(request.get("grader")), self.client,
                                  question_set=question_set, question_id=request.get("question"),
                                  quick_check=request.get("quick_check"))
                self.send({"id": rid, "type": "result", "grade": grade})
            except Exception as e:
                self.error(rid, f"{type(e).__name__}: {e}")
            finally:
                if traces is not None:
                    traces.close()
                with self._jobs_lock:
                    self._jobs.pop(rid, None)

        thread = threading.Thread(target=work, name=f"grade-run-{rid}", daemon=True)
        with self._jobs_lock:
            self._jobs[rid] = {"thread": thread, "cancel": threading.Event()}
        self.send({"id": rid, "type": "accepted"})
        thread.start()

    def start_grade(self, rid: str, request: dict) -> None:
        question_set = request.get("question_set") or {}
        check_question_set(question_set)
        setlist_run = request.get("setlist_run")
        if not isinstance(setlist_run, str) or not setlist_run:
            raise ValueError('"setlist_run" must be a setlist ID')
        cancel = threading.Event()

        def work():
            traces = None
            try:
                traces = self.traces()
                summary = grade_setlist(
                    traces, question_set, setlist_run,
                    settings=GraderSettings.from_dict(request.get("grader")),
                    client=self.client, regrade=bool(request.get("regrade")),
                    concurrency=int(request.get("concurrency") or 4),
                    run_ids=request.get("run_ids"),
                    on_event=lambda event: self.send({"id": rid, "type": "event", **event}),
                    cancel=cancel,
                )
                self.send({"id": rid, "type": "result", "result": dataclasses.asdict(summary)})
            except Exception as e:
                self.error(rid, f"{type(e).__name__}: {e}")
            finally:
                if traces is not None:
                    traces.close()
                with self._jobs_lock:
                    self._jobs.pop(rid, None)

        thread = threading.Thread(target=work, name=f"grade-{rid}", daemon=True)
        with self._jobs_lock:
            self._jobs[rid] = {"thread": thread, "cancel": cancel}
        self.send({"id": rid, "type": "accepted"})
        thread.start()

    def start_follow(self, rid: str, request: dict) -> None:
        sets = request.get("question_sets")
        if not isinstance(sets, list):
            raise ValueError('"question_sets" must be a list of question sets')
        for question_set in sets:
            check_question_set(question_set)
        settings = GraderSettings.from_dict(request.get("grader"))
        interval = float(request.get("interval") or DEFAULT_INTERVAL_SECONDS)
        cancel = threading.Event()
        with self._jobs_lock:
            # One follow at a time: a new one replaces the old. It waits for the old one to
            # finish the runs it's grading, so no run is graded (and paid for) twice.
            replaced = [job["thread"] for job in self._jobs.values() if job.get("follow")]
            for job in self._jobs.values():
                if job.get("follow"):
                    job["cancel"].set()

        def work():
            traces = None
            try:
                for thread in replaced:
                    thread.join()
                if cancel.is_set():  # replaced or stopped while waiting
                    self.send({"id": rid, "type": "result", "result": dataclasses.asdict(FollowTotals(cancelled=True))})
                    return
                traces = self.traces()
                follower = Follower(traces, sets, settings, self.client, cancel=cancel, reconnect=self.traces,
                                    on_event=lambda event: self.send({"id": rid, "type": "event", **event}))
                try:
                    totals = follower.run(interval)
                finally:
                    traces = follower.traces  # it may have reconnected: close the one it ended with
                self.send({"id": rid, "type": "result", "result": dataclasses.asdict(totals)})
            except Exception as e:
                self.error(rid, f"{type(e).__name__}: {e}")
            finally:
                if traces is not None:
                    traces.close()
                with self._jobs_lock:
                    self._jobs.pop(rid, None)

        thread = threading.Thread(target=work, name=f"follow-{rid}", daemon=True)
        with self._jobs_lock:
            self._jobs[rid] = {"thread": thread, "cancel": cancel, "follow": True}
        self.send({"id": rid, "type": "accepted"})
        thread.start()

    def wait_all(self, timeout: float = 60) -> None:
        with self._jobs_lock:
            threads = [job["thread"] for job in self._jobs.values()]
        for thread in threads:
            thread.join(timeout)

    def stop_all(self) -> None:
        with self._jobs_lock:
            for job in self._jobs.values():
                job["cancel"].set()
        self.wait_all()

    def serve_forever(self) -> None:
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self.error(None, f"invalid JSON: {e}")
                continue
            if not isinstance(request, dict):
                self.error(None, "each request must be a JSON object")
                continue
            if not self.handle(request):
                return
        self.stop_all()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve EH-1's grader over JSON Lines on stdin/stdout.")
    parser.add_argument("--traces-db", default="agent_traces.duckdb", help="The agent's traces database.")
    parser.add_argument("--no-traces-server", action="store_true", help="Open the traces file directly.")
    args = parser.parse_args(argv)
    load_dotenv()  # ANTHROPIC_API_KEY from .env, unless the app passed one
    Server(sys.stdin, sys.stdout, args.traces_db, traces_server=not args.no_traces_server).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
