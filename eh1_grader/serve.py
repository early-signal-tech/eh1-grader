"""Drive EH-1's grader from the app over JSON Lines on stdin/stdout, like the agent's
duck-harness-serve.

    uv run eh1-serve --traces-db ../duckdb-analytics-agent/agent_traces.duckdb

Requests (each has an "id" and a "type"; every message names its request's "id"):
    {"type": "hello"}
        -> {"type": "hello", "protocol": 1, "defaults": {...GraderSettings}, "judge_models": [...]}
    {"type": "setlists"}
        -> {"type": "setlists", "setlists": [{id, name, note, question_set, revision, runs,
            completed, columns, repeats, concurrency, cost_usd, started_at, ended_at}, ...]}
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
    {"type": "stop", "grade": "<id of a grade request>"}  -> {"type": "ok"}
"grade" also takes "run_ids": [...] to grade just those runs again (one scorecard cell).
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
from duckdb_agent import connect_traces_db
from duckdb_agent.config import API_MAX_RETRIES

from .grade import GraderSettings, check_question_set, grade_setlist, grader_key
from .judge import JUDGE_PRICING
from .traces import create_grades_table, list_setlists, read_grades, set_override

PROTOCOL_VERSION = 1


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
                           "defaults": dataclasses.asdict(GraderSettings()),
                           "judge_models": ["claude-haiku-5-5", *JUDGE_PRICING]})
            elif kind == "setlists":
                traces = self.traces()
                try:
                    self.send({"id": rid, "type": "setlists", "setlists": list_setlists(traces)})
                finally:
                    traces.close()
            elif kind == "grades":
                question_set = request.get("question_set") or {}
                check_question_set(question_set)
                key = grader_key(GraderSettings.from_dict(request.get("grader")), question_set)
                traces = self.traces()
                try:
                    create_grades_table(traces)
                    grades = read_grades(traces, request.get("setlist_run", ""), key)
                finally:
                    traces.close()
                self.send({"id": rid, "type": "grades", "grader_key": key, "grades": grades})
            elif kind == "grade":
                self.start_grade(rid, request)
            elif kind == "override":
                self.override(rid, request)
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

    def override(self, rid: str, request: dict) -> None:
        """Your CORRECT and CAREFUL for one graded run, kept next to the judge's."""
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
            create_grades_table(traces)
            if not set_override(traces, str(request.get("run_id", "")), key, value):
                raise ValueError("that run isn't graded with these settings")
        finally:
            traces.close()
        self.send({"id": rid, "type": "ok"})

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
