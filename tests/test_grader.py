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
from eh1_grader.traces import Run, ToolCall, create_tables, list_setlists, read_grades, rename_setlist, setlist_runs


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
            answer="There are 100 users.", calls=(), setlist_run="sl-1", effort="low", labels=None, prompt="q",
            data_source=None):
    if labels is None:
        labels = {"setlist": "mini", "setlist_revision": 1, "setlist_run": setlist_run, "setlist_name": "Test",
                  "question": question, "column": column, "repeat": repeat}
    preset = {"id": f"p-{column}", "name": f"Preset {column}", "revision": 1, "modified": column == "B"}
    config = {"model": "claude-haiku-5-5", "effort": effort, "max_iterations": 10}
    traces.conn.execute(
        "INSERT INTO agent_runs (run_id, prompt, final_response, status, total_iterations, labels, preset, config, "
        "data_source, started_at, estimated_cost_usd) VALUES (?, ?, ?, ?, ?, ?::JSON::VARIANT, ?::JSON::VARIANT, "
        "?::JSON::VARIANT, ?::JSON::VARIANT, now(), 0.001)",
        [run_id, prompt, answer, status, iterations, json.dumps(labels) if labels else None, json.dumps(preset),
         json.dumps(config), json.dumps(data_source) if data_source else None],
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
    assert first["output_config"]["format"]["type"] == "json_schema" and first["model"] == "claude-haiku-5-5"
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
    sonnet = GraderSettings(judge_model="claude-sonnet-5-5")
    other = grade_setlist(traces, question_set, "sl-1", sonnet, client=FakeClient())
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
    create_tables(traces)
    add_run(traces, "r1", "A1", "A", 1)
    add_run(traces, "r2", "A1", "B", 1, status="max_iterations", effort="high")
    add_run(traces, "r4", "H1", "B", 1, status="cancelled", effort="high")
    add_run(traces, "r3", "A1", "A", 1, setlist_run="sl-2")
    setlists = {s["id"]: s for s in list_setlists(traces)}
    one = setlists["sl-1"]
    assert one["runs"] == 3 and one["completed"] == 1 and one["cancelled"] == 1
    assert one["columns"] == 2 and one["name"] == "Test"
    assert [(p["column"], p["name"], p["modified"], p["config"]["effort"]) for p in one["presets"]] == [
        ("A", "Preset A", False, "low"), ("B", "Preset B", True, "high")]
    assert len(setlist_runs(traces, "sl-2")) == 1


def test_rename_is_yours_and_blank_shows_dh1s_again(traces):
    create_tables(traces)
    add_run(traces, "r1", "A1")
    rename_setlist(traces, "sl-1", "After the prompt change", None)
    s = list_setlists(traces)[0]
    assert (s["name"], s["note"]) == ("After the prompt change", None)
    rename_setlist(traces, "sl-1", None, "Haiku only")
    s = list_setlists(traces)[0]
    assert (s["name"], s["note"]) == ("Test", "Haiku only")  # DH-1's label is back
    rename_setlist(traces, "sl-1", None, None)
    assert traces.execute("SELECT count(*) FROM eval_setlists").fetchone()[0] == 0


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


def test_serve_setlists_with_grading_status_and_rename(tmp_path, traces, question_set):
    add_run(traces, "r1", "A1")
    add_run(traces, "r2", "H1")
    add_run(traces, "r3", "A1", setlist_run="sl-2")
    traces.close()
    out = Output()
    server = Server(None, out, str(tmp_path / "traces.duckdb"), traces_server=False, client=FakeClient())
    server.handle({"id": "g", "type": "grade", "setlist_run": "sl-1", "question_set": question_set,
                   "run_ids": ["r1"]})
    server.wait_all()
    sets = [{"id": "mini", "revision": 1}]
    server.handle({"id": "s", "type": "setlists", "grader": {}, "question_sets": sets})
    server.handle({"id": "o", "type": "setlists", "grader": {"judge_model": "claude-opus-5-5"}, "question_sets": sets})
    server.handle({"id": "n", "type": "rename", "setlist_run": "sl-2", "name": "  Second   try ", "note": ""})
    server.handle({"id": "bad", "type": "rename", "setlist_run": "sl-9", "name": "x"})
    server.handle({"id": "plain", "type": "setlists"})
    by_id = {m["id"]: m for m in out.messages()}
    now = {s["id"]: s for s in by_id["s"]["setlists"]}
    # r1 graded: A1's judge says 2 + 1, the trace says 2 for efficiency.
    assert now["sl-1"]["grading"] == {"graded": 1, "total": 5.0, "other_settings": 0}
    assert now["sl-2"]["grading"]["graded"] == 0
    other = {s["id"]: s for s in by_id["o"]["setlists"]}
    assert other["sl-1"]["grading"] == {"graded": 0, "total": None, "other_settings": 1}
    assert by_id["n"]["type"] == "ok"
    assert by_id["bad"]["type"] == "error" and "no setlist" in by_id["bad"]["error"]
    plain = {s["id"]: s for s in by_id["plain"]["setlists"]}
    assert plain["sl-2"]["name"] == "Second try" and "grading" not in plain["sl-2"]


def test_grading_revision_follows_the_apps_sets():
    from eh1_grader.serve import grading_revision
    ran = {"question_set": "mini", "revision": 1}
    assert grading_revision(ran, [{"id": "mini", "revision": 2}, {"id": "mini", "revision": 1}]) == 1
    assert grading_revision(ran, [{"id": "mini", "revision": 2}]) == 2
    assert grading_revision(ran, []) == 1
    assert grading_revision({"question_set": "mini", "revision": None}, [{"id": "mini", "revision": 3}]) == 3


# --- RUN mode: one run at a time ---

def test_list_runs_by_kind_and_search(traces):
    from eh1_grader.traces import list_runs
    create_tables(traces)
    add_run(traces, "s1", "A1")
    add_run(traces, "one", None, labels={}, prompt="What was revenue in July?")
    add_run(traces, "a", None, labels={"ab": "A", "pair": "ab-1"}, prompt="Top customers")
    ids = lambda kind, search="": [r["run_id"] for r in list_runs(traces, kind, search)]  # noqa: E731
    assert set(ids("all")) == {"s1", "one", "a"}
    assert ids("single") == ["one"] and ids("ab") == ["a"] and ids("setlist") == ["s1"]
    assert ids("all", "revenue july") == ["one"]
    with pytest.raises(ValueError):
        list_runs(traces, "nope")


def test_a_runs_data_comes_from_what_it_asked_for(traces, data_db):
    from eh1_grader.traces import run_by_id, source_spec
    add_run(traces, "t", None, labels={}, data_source={"tpch": 0.01})
    add_run(traces, "d", None, labels={}, data_source={"db": data_db, "schema": "gold"})
    add_run(traces, "old", None, labels={})
    assert source_spec(run_by_id(traces, "t")) == {"kind": "tpch", "tpchScale": 0.01}
    assert source_spec(run_by_id(traces, "d")) == {"kind": "db", "dbPath": data_db, "schema": "gold"}
    assert source_spec(run_by_id(traces, "old")) is None


def test_grade_one_against_a_question_shares_the_setlist_grade(traces, question_set):
    from eh1_grader.grade import grade_one
    add_run(traces, "r1", "A1")
    grade = grade_one(traces, "r1", client=FakeClient(), question_set=question_set, question_id="A1")
    assert (grade["question_set"], grade["question"], grade["correct"]) == ("mini", "A1", 2)
    # The same key a setlist grading uses: the scorecard sees it.
    assert read_grades(traces, "sl-1", grader_key(GraderSettings(), question_set))[0]["run_id"] == "r1"
    with pytest.raises(ValueError, match="no question"):
        grade_one(traces, "r1", client=FakeClient(), question_set=question_set, question_id="Z9")


def test_a_quick_check_runs_its_sql_on_the_runs_own_data(traces, data_db):
    from eh1_grader.grade import grade_one
    add_run(traces, "q1", None, labels={}, prompt="How many users?", data_source={"db": data_db, "schema": "gold"})
    client = FakeClient()
    check = {"correct": "Says 100 users.", "sql": ["SELECT count(*) AS users FROM users"], "careful": ""}
    grade = grade_one(traces, "q1", client=client, quick_check=check)
    assert grade["question_set"] == "quick" and grade["question"] is None
    assert grade["quick_check"]["correct"] == "Says 100 users." and "careful" not in grade["quick_check"]
    assert grade["reference"][0]["rows"] == 1 and "100" in grade["reference"][0]["preview"]
    assert "How many users?" in json.dumps(client.calls[0]["messages"])
    # A different check is a different grade, beside the first.
    again = grade_one(traces, "q1", client=client, quick_check={**check, "correct": "Says 100."})
    assert again["grader_key"] != grade["grader_key"]
    with pytest.raises(ValueError, match="correct"):
        grade_one(traces, "q1", client=client, quick_check={"careful": "x"})


def test_a_quick_check_on_a_run_without_recorded_data(traces):
    from eh1_grader.grade import grade_one
    add_run(traces, "old", None, labels={}, prompt="How many?")
    grade = grade_one(traces, "old", client=FakeClient(), quick_check={"correct": "100", "sql": ["SELECT 1"]})
    assert "didn't record its data" in grade["reference"][0]["error"]
    assert grade["correct"] == 2  # the judge still grades against CORRECT IS


def test_create_tables_adds_new_columns_to_an_old_table(traces):
    traces.conn.execute("CREATE TABLE eval_grades (run_id VARCHAR, grader_key VARCHAR)")
    create_tables(traces)
    columns = {r[0] for r in traces.conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'eval_grades'").fetchall()}
    assert {"quick_check", "override", "judge_values"} <= columns


def test_serve_runs_run_grade_run_and_override_by_key(tmp_path, traces, data_db):
    add_run(traces, "q1", None, labels={}, prompt="How many users?", data_source={"db": data_db, "schema": "gold"})
    traces.close()
    out = Output()
    server = Server(None, out, str(tmp_path / "traces.duckdb"), traces_server=False, client=FakeClient())
    server.handle({"id": "l", "type": "runs", "kind": "single"})
    server.handle({"id": "g", "type": "grade_run", "run_id": "q1", "quick_check": {"correct": "100 users"}})
    server.wait_all()
    key = [m for m in out.messages() if m["id"] == "g"][-1]["grade"]["grader_key"]
    server.handle({"id": "o", "type": "override", "run_id": "q1", "grader_key": key,
                   "override": {"correct": 1, "careful": 1, "why": "close"}})
    server.handle({"id": "r", "type": "run", "run_id": "q1"})
    server.handle({"id": "bad", "type": "grade_run", "run_id": "q1"})
    by_id = {}
    for m in out.messages():
        by_id.setdefault(m["id"], []).append(m)
    assert by_id["l"][0]["runs"][0]["run_id"] == "q1"
    assert [m["type"] for m in by_id["g"]] == ["accepted", "result"]
    run = by_id["r"][0]
    assert run["run"]["source"] == {"kind": "db", "dbPath": data_db, "schema": "gold"} and run["run"]["source_recorded"]
    assert run["grades"][0]["override"]["why"] == "close"
    assert by_id["bad"][0]["type"] == "error"


# --- the rubric and Efficient settings ---

def test_later_settings_leave_old_keys_alone(question_set):
    base = grader_key(GraderSettings(), question_set)
    # The key a grade got before instructions and waste_for_one existed.
    import hashlib
    from dataclasses import asdict
    from eh1_grader.judge import JUDGE_VERSION
    old = {k: v for k, v in asdict(GraderSettings()).items() if k not in ("instructions", "waste_for_one")}
    payload = {"grader": old, "set": "mini", "revision": 1, "judge": JUDGE_VERSION}
    assert base == hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    assert grader_key(GraderSettings(instructions="Be lenient on rounding."), question_set) != base
    assert grader_key(GraderSettings(waste_for_one=1), question_set) != base


def test_settings_are_cleaned():
    s = GraderSettings.from_dict({"instructions": "  x " * 2000, "iteration_budget": 999, "waste_for_one": -3, "nope": 1})
    assert len(s.instructions) <= 2000 and s.iteration_budget == 50 and s.waste_for_one == 0


def test_instructions_go_to_the_judge(traces, question_set):
    add_run(traces, "r1", "A1")
    client = FakeClient()
    grade_setlist(traces, question_set, "sl-1", settings=GraderSettings(instructions="Ignore currency symbols."), client=client)
    assert "<grader_notes>\nIgnore currency symbols.\n</grader_notes>" in client.calls[0]["messages"][0]["content"]
    client = FakeClient()
    grade_setlist(traces, question_set, "sl-1", client=client, regrade=True)
    assert "grader_notes" not in client.calls[0]["messages"][0]["content"]


def test_waste_for_one():
    run = run_of([call("query_database", f"SELECT {n}", success=False) for n in range(3)])
    assert efficiency(run)[0] == 0
    assert efficiency(run, waste_for_one=3)[0] == 1


def test_hello_has_the_scale(tmp_path):
    out = Output()
    Server(None, out, str(tmp_path / "t.duckdb"), traces_server=False).handle({"id": "h", "type": "hello"})
    hello = out.messages()[0]
    assert [s["score"] for s in hello["scale"]] == ["CORRECT", "CAREFUL", "EFFICIENT"]
    assert hello["defaults"]["waste_for_one"] == 2 and hello["defaults"]["instructions"] == ""


def test_hello_reports_versions(tmp_path):
    from duckdb_agent import SCHEMA_VERSION
    from eh1_grader.traces import GRADES_SCHEMA
    out = Output()
    Server(None, out, str(tmp_path / "t.duckdb"), traces_server=False).handle({"id": "h", "type": "hello"})
    hello = out.messages()[0]
    assert hello["version"] not in ("", "unknown")
    assert hello["traces_schema"] == SCHEMA_VERSION  # the agent's, from the package it was built with
    assert hello["grades_schema"] == GRADES_SCHEMA and len(GRADES_SCHEMA) == 12


# --- following DH-1 ---

def end(traces, run_id, minutes_ago=0):
    traces.conn.execute(f"UPDATE agent_runs SET ended_at = now() - INTERVAL {int(minutes_ago)} MINUTE WHERE run_id = ?",
                        [run_id])


def test_follow_grades_setlists_that_play_after_it_starts(traces, question_set):
    from eh1_grader.follow import Follower
    add_run(traces, "old1", "A1", setlist_run="sl-old")
    end(traces, "old1", minutes_ago=60)
    add_run(traces, "mid1", "A1", setlist_run="sl-mid")
    end(traces, "mid1", minutes_ago=30)  # finished before following began, but its setlist plays on
    events = []
    follower = Follower(traces, [question_set], client=FakeClient(), on_event=events.append)
    assert events[0]["event"] == "follow_start"
    assert follower.check() == 0  # nothing new yet

    add_run(traces, "mid2", "H1", setlist_run="sl-mid")
    add_run(traces, "new1", "A1", setlist_run="sl-new")
    add_run(traces, "new2", "A1", column="B", setlist_run="sl-new")  # still running: no ended_at
    end(traces, "mid2")
    end(traces, "new1")
    assert follower.check() == 3
    graded = {(e["setlist_run"], e["run_id"]) for e in events if e["event"] == "graded"}
    assert graded == {("sl-mid", "mid1"), ("sl-mid", "mid2"), ("sl-new", "new1")}  # not sl-old, not new2 yet

    end(traces, "new2")
    assert follower.check() == 1
    assert follower.check() == 0  # nothing twice
    assert traces.execute("SELECT count(*) FROM eval_grades WHERE run_id LIKE 'old%'").fetchone()[0] == 0
    assert follower.totals.graded == 4


def test_follow_skips_a_setlist_without_its_question_set(traces, question_set):
    from eh1_grader.follow import Follower
    events = []
    follower = Follower(traces, [question_set], client=FakeClient(), on_event=events.append)
    labels = {"setlist": "other", "setlist_revision": 1, "setlist_run": "sl-x", "question": "Q1", "column": "A", "repeat": 1}
    add_run(traces, "x1", "Q1", labels=labels)
    end(traces, "x1")
    follower.check()
    follower.check()
    skipped = [e for e in events if e["event"] == "follow_skipped"]
    assert len(skipped) == 1 and skipped[0]["setlist_run"] == "sl-x" and "other" in skipped[0]["reason"]


def test_follow_uses_the_revision_the_setlist_ran(question_set):
    from eh1_grader.follow import set_for
    r2 = {**question_set, "revision": 2}
    assert set_for("mini", 1, [r2, question_set]) is question_set
    assert set_for("mini", 1, [r2]) is r2  # the one you have
    assert set_for("nope", 1, [question_set]) is None


def test_serve_follow_then_stop(tmp_path, question_set):
    import time
    path = str(tmp_path / "traces.duckdb")
    out = Output()
    server = Server(None, out, path, traces_server=False, client=FakeClient())
    server.handle({"id": "f", "type": "follow", "question_sets": [question_set], "interval": 0.05})
    time.sleep(0.3)  # follow_start, with nothing new
    writer = init_traces_db(path)  # traces_server=False: the server's connection and this one share the file
    add_run(writer, "r1", "A1", setlist_run="sl-live")
    end(writer, "r1")
    deadline = time.time() + 5
    while time.time() < deadline and not any(m.get("event") == "graded" for m in out.messages()):
        time.sleep(0.05)
    server.handle({"id": "s", "type": "stop", "grade": "f"})
    server.wait_all(5)
    writer.close()
    messages = [m for m in out.messages() if m["id"] == "f"]
    assert messages[0]["type"] == "accepted"
    assert [m["event"] for m in messages if m["type"] == "event"][:2] == ["follow_start", "graded"]
    result = messages[-1]
    assert result["type"] == "result" and result["result"]["graded"] == 1 and result["result"]["cancelled"]


def test_a_new_follow_replaces_the_old(tmp_path, question_set):
    out = Output()
    server = Server(None, out, str(tmp_path / "traces.duckdb"), traces_server=False, client=FakeClient())
    server.handle({"id": "f1", "type": "follow", "question_sets": [question_set], "interval": 0.05})
    server.handle({"id": "f2", "type": "follow", "question_sets": [question_set], "interval": 0.05})
    import time
    time.sleep(0.3)
    server.handle({"id": "s", "type": "stop", "grade": "f2"})
    server.wait_all(5)
    results = {m["id"] for m in out.messages() if m["type"] == "result"}
    assert results == {"f1", "f2"}  # f1 ended when f2 began; f2 when stopped


def test_follow_reconnects_after_a_failed_check(traces, question_set):
    """An idle connection to the traces server can expire; following reconnects and goes on.
    While reconnecting keeps failing, it says so once."""
    from eh1_grader.follow import Follower
    events = []
    cancel = threading.Event()
    attempts = {"reconnect": 0, "check": 0}

    def reconnect():
        attempts["reconnect"] += 1
        if attempts["reconnect"] == 1:
            raise RuntimeError("traces server not answering")  # still down the first time
        return traces

    follower = Follower(traces, [question_set], client=FakeClient(), on_event=events.append, cancel=cancel, reconnect=reconnect)
    original = follower.check

    def flaky_check():
        attempts["check"] += 1
        if attempts["check"] == 1:
            raise RuntimeError("Invalid Input Error: Connection heartbeat lease expired")
        cancel.set()
        return original()

    follower.check = flaky_check
    follower.traces.close = lambda: None  # don't close the fixture's connection on reopen
    totals = follower.run(interval=0.01)
    kinds = [e["event"] for e in events]
    assert kinds == ["follow_start", "follow_retry", "follow_reconnected"]
    assert "lease expired" in events[1]["error"]
    assert attempts == {"reconnect": 2, "check": 2} and totals.cancelled


def test_follow_without_reconnect_still_stops_on_a_failure(traces, question_set):
    from eh1_grader.follow import Follower
    follower = Follower(traces, [question_set], client=FakeClient())
    follower.check = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        follower.run(interval=0.01)
