import io
import json
import threading
from types import SimpleNamespace

import duckdb
import pytest
from duckdb_agent.tracing import init_traces_db

from eh1_grader.efficiency import efficiency
from eh1_grader.grade import GraderSettings, grade_setlist, grader_key
from eh1_grader.judge import build_prompt
from eh1_grader.reference import References
from eh1_grader.rules import check_rules
from eh1_grader.serve import Server
from eh1_grader.traces import Run, ToolCall, list_setlists, read_grades, setlist_runs


# --- fakes ---

class FakeClient:
    """Stands in for anthropic.Anthropic: answers every judge call with `verdict`."""

    def __init__(self, verdict=None, stop_reason="end_turn"):
        self.verdict = verdict or {"correct": 2, "correct_reason": "Matches.", "careful": 1,
                                   "careful_reason": "Some caveats.", "unsure": False,
                                   "values": [{"name": "users", "agent": "100", "reference": "100", "check": "matches"},
                                              {"name": "day count", "agent": "17", "reference": "", "check": "unverified"}]}
        self.stop_reason = stop_reason
        self.calls = []
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            content=[SimpleNamespace(type="text", text=json.dumps(self.verdict))],
        )


def call(tool, sql=None, success=True, error_type=None, truncated=False, iteration=1):
    return ToolCall(tool, sql, success, error_type, truncated, iteration)


def run_of(calls=(), status="completed", iterations=3, answer="There are 100 users."):
    return Run("r", "q", answer, status, iterations, {}, None, None, None, None, list(calls))


# --- a throwaway traces file with one setlist ---

@pytest.fixture
def data_db(tmp_path):
    path = str(tmp_path / "shop.duckdb")
    conn = duckdb.connect(path)
    conn.execute("CREATE SCHEMA gold; CREATE TABLE gold.users AS SELECT range AS id FROM range(100)")
    conn.close()
    return path


@pytest.fixture
def traces(tmp_path):
    t = init_traces_db(str(tmp_path / "traces.duckdb"))
    yield t
    t.close()


def add_run(traces, run_id, question, column="A", repeat=1, status="completed", iterations=2,
            answer="There are 100 users.", calls=(), setlist_run="sl-1"):
    labels = {"setlist": "mini", "setlist_revision": 1, "setlist_run": setlist_run, "setlist_name": "Test",
              "question": question, "column": column, "repeat": repeat}
    traces.conn.execute(
        "INSERT INTO agent_runs (run_id, prompt, final_response, status, total_iterations, labels, started_at, "
        "estimated_cost_usd) VALUES (?, ?, ?, ?, ?, ?::JSON::VARIANT, now(), 0.001)",
        [run_id, "q", answer, status, iterations, json.dumps(labels)],
    )
    for c in calls:
        traces.insert("tool_calls", {"run_id": run_id, "iteration": c.iteration, "tool_name": c.tool.replace("_", " ").title(),
                                     "input_sql": c.sql, "success": c.success, "error_type": c.error_type,
                                     "truncated": c.truncated})


@pytest.fixture
def question_set(data_db):
    return {
        "format": "dh-question-set", "formatVersion": 2, "id": "mini", "name": "Mini", "revision": 1,
        "source": {"kind": "db", "dbPath": data_db, "schema": "gold"},
        "categories": [{"id": "A", "name": "Basics"}, {"id": "H", "name": "Discovery"}],
        "questions": [
            {"id": "A1", "category": "A", "prompt": "How many users?",
             "check": {"sql": ["SELECT count(*) AS users FROM users"], "reference": "100 users.",
                       "correct": "Says 100.", "fail": "Invents a number."}},
            {"id": "H1", "category": "H", "prompt": "What tables?", "tables": [],
             "check": {"correct": "Lists users.", "rules": [
                 {"tool": "list_tables", "min": 1, "label": "Used list_tables"},
                 {"before": "list_tables", "after": "describe_table"}]}},
            {"id": "X1", "prompt": "No check here."},
        ],
    }


# --- rules ---

def test_rules_of_every_kind():
    run = run_of([
        call("list_tables"), call("describe_table"),
        call("query_database", "SUMMARIZE users"),
        call("query_database", "SELECT 1; SELECT 2", success=False, error_type="multiple_statements"),
    ], iterations=4)
    results, all_passed = check_rules([
        {"error": "multiple_statements", "expect": "present"},
        {"error": "read_only_violation", "expect": "absent"},
        {"tool": "list_tables", "min": 1, "max": 1},
        {"before": "list_tables", "after": "describe_table"},
        {"sql": "(?i)\\bsummarize\\b", "expect": "present", "max": 1},
        {"sql": "/etc/", "expect": "absent", "label": "Didn't touch local files"},
        {"iterations": {"max": 3}},
        {"queries": {"max": 2}},
    ], run)
    assert [r["passed"] for r in results] == [True, True, True, True, True, True, False, True]
    assert results[5]["label"] == "Didn't touch local files"
    assert results[6] == {"label": "at most 3 iterations", "passed": False, "detail": "4 iterations"}
    assert all_passed is False
    assert check_rules([], run) == ([], None)


def test_order_rule_when_tools_are_missing():
    guessed = run_of([call("describe_table")])
    assert check_rules([{"before": "list_tables", "after": "describe_table"}], guessed)[1] is False
    never = run_of([call("list_tables")])
    assert check_rules([{"before": "list_tables", "after": "describe_table"}], never)[1] is True


# --- efficiency ---

@pytest.mark.parametrize("calls, status, iterations, score", [
    ([call("query_database", "SELECT 1")], "completed", 3, 2),
    ([call("query_database", "SELECT 1"), call("query_database", "select  1;")], "completed", 3, 1),  # repeated
    ([call("query_database", "x", success=False)] * 3, "completed", 3, 0),                            # 3 failures
    ([], "max_iterations", 10, 0),                                                                      # a limit
    ([call("query_database", "SELECT 1")], "completed", 12, 1),                                        # over budget
    ([call("query_database", "SELECT 1")], "completed", 17, 0),                                        # over 2x budget
    ([call("query_database", "a", truncated=True), call("query_database", "b", truncated=True)], "completed", 3, 1),
])
def test_efficiency(calls, status, iterations, score):
    assert efficiency(run_of(calls, status, iterations))[0] == score


# --- reference SQL ---

def test_reference_runs_read_only_on_the_right_schema_and_caches(data_db):
    refs = References()
    source = {"kind": "db", "dbPath": data_db, "schema": "gold"}
    first = refs.run(source, "SELECT count(*) AS users FROM users")
    assert (first["rows"], first["columns"], first["error"]) == (1, ["users"], None)
    assert "100" in first["preview"]
    assert refs.run(source, "SELECT count(*) AS users FROM users") is first
    blocked = refs.run(source, "DELETE FROM users")
    assert blocked["error"] and blocked["rows"] is None
    refs.close()


def test_the_judge_sees_criteria_references_and_the_answer():
    prompt = build_prompt(
        {"prompt": "How many users?"}, {"correct": "Says 100.", "fail": "Invents a number.", "sql": ["SELECT 1"]},
        [{"sql": "SELECT count(*) FROM users", "rows": 1, "preview": "count\n100", "error": None}],
        "There are 100 users.", "±0.5%",
    )
    assert "How many users?" in prompt and '"fail": "Invents a number."' in prompt
    assert "count\n100" in prompt and "There are 100 users." in prompt and "±0.5%" in prompt
    assert '"sql"' not in prompt.split("<criteria>")[1].split("</criteria>")[0]


# --- grading a setlist ---

def test_grade_a_setlist(traces, question_set):
    add_run(traces, "r1", "A1", "A", 1)
    add_run(traces, "r2", "A1", "B", 1, answer="There are 250 users.")
    add_run(traces, "r3", "H1", "A", 1, calls=[call("list_tables"), call("describe_table")])
    add_run(traces, "r4", "H1", "B", 1, calls=[call("describe_table")])
    add_run(traces, "r5", "X1", "A", 1)
    add_run(traces, "r6", "Q9", "A", 1)  # not in the set
    client = FakeClient()
    events = []
    summary = grade_setlist(traces, question_set, "sl-1", client=client, on_event=events.append)

    assert (summary.runs, summary.graded, summary.skipped, summary.unmatched, summary.errors) == (6, 5, 0, 1, [])
    assert events[0]["event"] == "grade_start" and events[0]["to_grade"] == 5
    assert len(client.calls) == 4  # X1 has no check, so no judge
    first = client.calls[0]
    assert first["output_config"]["format"]["type"] == "json_schema" and first["model"] == "claude-sonnet-5-5"
    assert "tool_choice" not in first

    grades = {g["run_id"]: g for g in read_grades(traces, "sl-1", summary.grader_key)}
    assert set(grades) == {"r1", "r2", "r3", "r4", "r5"}
    assert (grades["r1"]["correct"], grades["r1"]["careful"], grades["r1"]["efficient"]) == (2, 1, 2)
    assert grades["r1"]["column"] == "A" and grades["r1"]["judge_values"][0]["name"] == "users"
    assert [v["check"] for v in grades["r1"]["judge_values"]] == ["matches", "unverified"]
    assert first["output_config"]["format"]["schema"]["properties"]["values"]["items"]["properties"]["check"]["enum"] == [
        "matches", "differs", "unverified"]
    assert '"unverified"' in first["system"]
    assert grades["r3"]["rules_passed"] is True and grades["r4"]["rules_passed"] is False
    assert grades["r5"]["correct"] is None and grades["r5"]["efficient"] == 2
    ref = traces.execute("SELECT reference::JSON::VARCHAR FROM eval_grades WHERE run_id = 'r1'").fetchone()[0]
    assert json.loads(ref)[0]["rows"] == 1

    # Again with the same settings: nothing to do. With regrade: all again.
    again = grade_setlist(traces, question_set, "sl-1", client=FakeClient())
    assert (again.graded, again.skipped) == (0, 5)
    redo = grade_setlist(traces, question_set, "sl-1", client=FakeClient(), regrade=True)
    assert redo.graded == 5
    assert traces.execute("SELECT count(*) FROM eval_grades").fetchone()[0] == 5  # replaced, not added

    # Different settings: new rows beside the old ones.
    haiku = GraderSettings(judge_model="claude-haiku-5-5")
    other = grade_setlist(traces, question_set, "sl-1", haiku, client=FakeClient())
    assert other.grader_key != summary.grader_key and other.graded == 5
    assert traces.execute("SELECT count(*) FROM eval_grades").fetchone()[0] == 10


def test_judge_failures_are_recorded_not_fatal(traces, question_set):
    add_run(traces, "r1", "A1")
    summary = grade_setlist(traces, question_set, "sl-1", client=FakeClient(stop_reason="refusal"))
    assert summary.graded == 1 and summary.errors == []
    grade = read_grades(traces, "sl-1", summary.grader_key)[0]
    assert grade["correct"] is None and grade["judge_error"] == "judge stopped: refusal"


def test_grader_key_depends_on_settings_and_set_revision(question_set):
    base = grader_key(GraderSettings(), question_set)
    assert grader_key(GraderSettings(), question_set) == base
    assert grader_key(GraderSettings(tolerance="±1%"), question_set) != base
    assert grader_key(GraderSettings(), {**question_set, "revision": 2}) != base


def test_list_setlists(traces):
    add_run(traces, "r1", "A1", "A", 1)
    add_run(traces, "r2", "A1", "B", 1, status="max_iterations")
    add_run(traces, "r3", "A1", "A", 1, setlist_run="sl-2")
    setlists = {s["id"]: s for s in list_setlists(traces)}
    assert setlists["sl-1"]["runs"] == 2 and setlists["sl-1"]["completed"] == 1
    assert setlists["sl-1"]["columns"] == 2 and setlists["sl-1"]["name"] == "Test"
    assert len(setlist_runs(traces, "sl-2")) == 1


# --- the app protocol ---

class Output(io.StringIO):
    def messages(self):
        return [json.loads(line) for line in self.getvalue().splitlines()]


def test_serve_hello_setlists_grade_and_grades(tmp_path, traces, question_set):
    add_run(traces, "r1", "A1")
    traces.close()
    out = Output()
    server = Server(None, out, str(tmp_path / "traces.duckdb"), traces_server=False, client=FakeClient())
    server.handle({"id": "h", "type": "hello"})
    server.handle({"id": "s", "type": "setlists"})
    server.handle({"id": "g", "type": "grade", "setlist_run": "sl-1", "question_set": question_set})
    server.wait_all()
    server.handle({"id": "q", "type": "grades", "setlist_run": "sl-1", "question_set": question_set})
    by_id = {}
    for m in out.messages():
        by_id.setdefault(m["id"], []).append(m)
    assert by_id["h"][0]["protocol"] == 1 and "claude-sonnet-5-5" in by_id["h"][0]["judge_models"]
    assert by_id["s"][0]["setlists"][0]["id"] == "sl-1"
    assert [m["type"] for m in by_id["g"]][0] == "accepted" and by_id["g"][-1]["type"] == "result"
    assert by_id["g"][-1]["result"]["graded"] == 1
    assert by_id["q"][0]["grades"][0]["correct"] == 2


def test_serve_refuses_bad_requests(tmp_path):
    out = Output()
    server = Server(None, out, str(tmp_path / "traces.duckdb"), traces_server=False, client=FakeClient())
    server.handle({"id": "g", "type": "grade", "setlist_run": "sl-1", "question_set": {"format": "nope"}})
    server.handle({"id": "x", "type": "dance"})
    errors = [m for m in out.messages() if m["type"] == "error"]
    assert "not a Duck Harness question set" in errors[0]["error"] and "unknown request" in errors[1]["error"]


# --- overrides and regrading one cell ---

def test_override_is_kept_next_to_the_judge_and_survives_a_regrade(traces, question_set):
    from eh1_grader.traces import set_override
    add_run(traces, "r1", "A1", "A", 1)
    add_run(traces, "r2", "A1", "A", 2)
    summary = grade_setlist(traces, question_set, "sl-1", client=FakeClient())
    key = summary.grader_key
    assert set_override(traces, "r1", key, {"correct": 0, "careful": 2, "why": "wrong nation"})
    assert not set_override(traces, "nope", key, {"correct": 0, "careful": 0, "why": ""})
    grade = {g["run_id"]: g for g in read_grades(traces, "sl-1", key)}["r1"]
    assert grade["correct"] == 2 and grade["override"]["correct"] == 0 and grade["override"]["why"] == "wrong nation"

    # Regrade just r1 (one cell's repeat): the judge's scores are new, your call stays.
    events = []
    again = grade_setlist(traces, question_set, "sl-1", client=FakeClient(), run_ids=["r1"], on_event=events.append)
    assert again.graded == 1 and events[0]["to_grade"] == 1
    grade = {g["run_id"]: g for g in read_grades(traces, "sl-1", key)}["r1"]
    assert grade["override"]["correct"] == 0

    assert set_override(traces, "r1", key, None)
    assert {g["run_id"]: g for g in read_grades(traces, "sl-1", key)}["r1"]["override"] is None


def test_serve_override(tmp_path, traces, question_set):
    add_run(traces, "r1", "A1")
    traces.close()
    out = Output()
    server = Server(None, out, str(tmp_path / "traces.duckdb"), traces_server=False, client=FakeClient())
    server.handle({"id": "g", "type": "grade", "setlist_run": "sl-1", "question_set": question_set})
    server.wait_all()
    server.handle({"id": "o", "type": "override", "run_id": "r1", "question_set": question_set,
                   "override": {"correct": 1, "careful": 1, "why": "partly right"}})
    server.handle({"id": "bad", "type": "override", "run_id": "r1", "question_set": question_set,
                   "override": {"correct": 3, "careful": 1}})
    server.handle({"id": "q", "type": "grades", "setlist_run": "sl-1", "question_set": question_set})
    by_id = {m["id"]: m for m in out.messages()}
    assert by_id["o"]["type"] == "ok"
    assert by_id["bad"]["type"] == "error" and "0, 1, or 2" in by_id["bad"]["error"]
    override = by_id["q"]["grades"][0]["override"]
    assert (override["correct"], override["careful"], override["why"]) == (1, 1, "partly right") and override["at"]
