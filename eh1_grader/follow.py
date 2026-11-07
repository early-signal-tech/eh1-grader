"""Following DH-1: grade setlist runs as they finish (the rack's AUTO-GRADE, and
`eh1-grade --follow`).

Every few seconds the traces are checked for setlists with a run that ended since
following began. Each such setlist's finished runs that aren't graded yet (with these
grader settings) are graded, so the scorecard fills in while the setlist plays. Switching
it on mid-setlist also picks up the runs that finished just before; setlists that were
already done when following began are left alone. Only setlists are followed: single runs
and A/B compares are graded when you ask (RUN mode).

A setlist is graded with its own question set, chosen from the sets given the way the app
chooses (grading_revision). A setlist whose set isn't given is reported once and skipped.
A run that fails to grade isn't tried again while following.

Following can sit idle for a long time between setlists. If a check fails (the traces
server lets an idle connection's lease lapse: "Connection heartbeat lease expired"), the
follower opens a new connection and carries on at the next check, rather than stopping.
"""

import threading
from dataclasses import dataclass, field
from typing import Callable

import anthropic
from duckdb_agent import TracesDB

from .grade import GraderSettings, grade_setlist, grader_key
from .traces import create_tables, graded_run_ids, lit

DEFAULT_INTERVAL_SECONDS = 5.0


def grading_revision(setlist: dict, sets: list[dict]) -> int:
    """The question set revision a setlist is graded with: the one it ran, when that's
    among `sets`; else the first of `sets` with that id; else the one it ran (as the app's
    findSet)."""
    same = [s.get("revision", 1) for s in sets if s["id"] == setlist["question_set"]]
    ran = setlist.get("revision")
    if ran is None:
        return same[0] if same else 1
    return ran if ran in same or not same else same[0]


def set_for(question_set: str, revision: int | None, sets: list[dict]) -> dict | None:
    """The set a setlist that ran `question_set` (at `revision`) is graded with, if given."""
    wanted = grading_revision({"question_set": question_set, "revision": revision}, sets)
    return next((s for s in sets if s["id"] == question_set and s.get("revision", 1) == wanted), None)


@dataclass
class FollowTotals:
    graded: int = 0
    judge_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False


def last_ended(traces: TracesDB):
    """When the newest run ended: following picks up setlists with a run after this."""
    return traces.execute("SELECT max(ended_at) FROM agent_runs").fetchone()[0]


def active_setlists(traces: TracesDB, since) -> list[tuple[str, str, int | None]]:
    """Setlists with a run that ended after `since` (all of them when it's None), oldest
    first: (setlist_run, question set id, revision)."""
    after = "" if since is None else f"AND ended_at > {lit(since.isoformat())}::TIMESTAMP"
    return [tuple(row) for row in traces.execute(f"""
        SELECT labels.setlist_run::VARCHAR, any_value(labels.setlist::VARCHAR),
               max(labels.setlist_revision::INTEGER)
        FROM agent_runs
        WHERE labels.setlist_run IS NOT NULL AND ended_at IS NOT NULL
        GROUP BY 1
        HAVING bool_or(true {after})
        ORDER BY min(started_at)
    """).fetchall()]


def finished_runs(traces: TracesDB, setlist_run: str) -> set[str]:
    return {r[0] for r in traces.execute(
        f"SELECT run_id FROM agent_runs WHERE labels.setlist_run::VARCHAR = {lit(setlist_run)} AND ended_at IS NOT NULL"
    ).fetchall()}


class Follower:
    """Grades newly finished setlist runs, one check at a time (`check`), or in a loop
    until cancelled (`run`). Events go to `on_event`:

        {"event": "follow_start", "since": iso | None}
        {"event": "graded", "setlist_run", "run_id", "question", ...}   (as grade_setlist)
        {"event": "grade_error", "setlist_run", "run_id", "error"}
        {"event": "follow_skipped", "setlist_run", "question_set", "revision", "reason"}
        {"event": "follow_retry", "error"}       (a check failed; reconnecting)
        {"event": "follow_reconnected"}          (following again)

    With `reconnect` (a function that opens the traces again), a failed check reconnects
    and following goes on; without it, the failure ends `run`.
    """

    def __init__(self, traces: TracesDB, question_sets: list[dict], settings: GraderSettings | None = None,
                 client: anthropic.Anthropic | None = None, on_event: Callable[[dict], None] | None = None,
                 cancel: threading.Event | None = None, concurrency: int = 4,
                 reconnect: Callable[[], TracesDB] | None = None):
        self.traces, self.sets = traces, question_sets
        self.reconnect = reconnect
        self.settings = settings or GraderSettings()
        self.client, self.concurrency = client, concurrency
        self.emit = on_event or (lambda event: None)
        self.cancel = cancel or threading.Event()
        self.totals = FollowTotals()
        self._attempted: set[str] = set()
        self._skipped: set[str] = set()
        create_tables(traces)
        self.since = last_ended(traces)
        self.emit({"event": "follow_start", "since": self.since.isoformat() if self.since else None})

    def check(self) -> int:
        """Grade what's newly finished. Returns how many runs were graded."""
        before = self.totals.graded
        for setlist_run, set_id, revision in active_setlists(self.traces, self.since):
            if self.cancel.is_set():
                break
            question_set = set_for(set_id, revision, self.sets)
            if question_set is None:
                if setlist_run not in self._skipped:
                    self._skipped.add(setlist_run)
                    self.emit({"event": "follow_skipped", "setlist_run": setlist_run, "question_set": set_id,
                               "revision": revision, "reason": f"no question set {set_id!r} to grade it with"})
                continue
            key = grader_key(self.settings, question_set)
            todo = finished_runs(self.traces, setlist_run) - graded_run_ids(self.traces, setlist_run, key) - self._attempted
            if not todo:
                continue
            self._attempted |= todo
            summary = grade_setlist(
                self.traces, question_set, setlist_run, self.settings, self.client,
                concurrency=self.concurrency, run_ids=sorted(todo), cancel=self.cancel,
                on_event=lambda event, sl=setlist_run: self._forward(sl, event),
            )
            self.totals.graded += summary.graded
            self.totals.judge_cost_usd += summary.judge_cost_usd
            self.totals.errors += summary.errors
        return self.totals.graded - before

    def _forward(self, setlist_run: str, event: dict) -> None:
        if event["event"] in ("graded", "grade_error"):  # grade_start is per check; not news
            self.emit({**event, "setlist_run": setlist_run})

    def run(self, interval: float = DEFAULT_INTERVAL_SECONDS) -> FollowTotals:
        """Check, then wait `interval` seconds, until cancelled. A failed check reconnects
        (with `reconnect`) and following goes on; the next check tries again."""
        broken = False
        while not self.cancel.is_set():
            try:
                if broken:
                    self._reopen()
                    broken = False
                    self.emit({"event": "follow_reconnected"})
                self.check()
            except Exception as e:
                if self.reconnect is None:
                    raise
                if not broken:  # say it once, not every interval while it stays down
                    self.emit({"event": "follow_retry", "error": f"{type(e).__name__}: {e}"})
                broken = True
            self.cancel.wait(interval)
        self.totals.cancelled = True
        return self.totals

    def _reopen(self) -> None:
        try:
            self.traces.close()
        except Exception:
            pass  # it's the broken one
        self.traces = self.reconnect()
