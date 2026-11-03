"""Reading runs from the agent's traces, and the tables EH-1 writes there (eval_grades,
eval_setlists).

The traces database belongs to the agent (duckdb-analytics-agent). EH-1 connects the
same way the agent does, through its background Quack server, so it can read while
runs write. Quack doesn't forward query parameters, so values go into the SQL as
escaped literals (see `lit`).
"""

import json
from dataclasses import dataclass, field

from duckdb_agent import TracesDB
from duckdb_agent.tools import TOOL_LABELS


def lit(value) -> str:
    """A SQL literal for a string, number, or None."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


@dataclass
class ToolCall:
    tool: str
    sql: str | None
    success: bool
    error_type: str | None
    truncated: bool
    iteration: int


@dataclass
class Run:
    run_id: str
    prompt: str
    answer: str
    status: str
    iterations: int
    labels: dict
    source_db: str | None
    source_schema: str | None
    cost_usd: float | None
    duration_ms: int | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def question(self) -> str | None:
        return self.labels.get("question")


def _labels(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


TOOL_IDS = {label: tool for tool, label in TOOL_LABELS.items()}


def tool_id(name: str | None) -> str:
    """tool_calls.tool_name holds the display label ("List Tables"); rules use the tool's
    id (list_tables). The agent's own label table, then a plain fallback."""
    name = (name or "").strip()
    return TOOL_IDS.get(name) or name.lower().replace(" ", "_")


def setlist_runs(traces: TracesDB, setlist_run: str) -> list[Run]:
    """Every run of one setlist (by its ID, the setlist_run label), with its tool calls."""
    rows = traces.execute(f"""
        SELECT run_id, prompt, coalesce(final_response, ''), status, coalesce(total_iterations, 0),
               labels::JSON::VARCHAR, source_db, source_schema, estimated_cost_usd, duration_ms
        FROM agent_runs
        WHERE labels.setlist_run::VARCHAR = {lit(setlist_run)}
        ORDER BY started_at
    """).fetchall()
    runs = [
        Run(run_id=r[0], prompt=r[1] or "", answer=r[2], status=r[3] or "", iterations=r[4],
            labels=_labels(r[5]), source_db=r[6] or None, source_schema=r[7], cost_usd=r[8], duration_ms=r[9])
        for r in rows
    ]
    if not runs:
        return runs
    ids = ", ".join(lit(r.run_id) for r in runs)
    calls = traces.execute(f"""
        SELECT run_id, tool_name, input_sql, success, error_type, coalesce(truncated, false), iteration
        FROM tool_calls WHERE run_id IN ({ids}) ORDER BY id
    """).fetchall()
    by_id = {r.run_id: r for r in runs}
    for run_id, tool, sql, success, error_type, truncated, iteration in calls:
        by_id[run_id].tool_calls.append(
            ToolCall(tool_id(tool), sql, bool(success), error_type, bool(truncated), iteration or 0))
    return runs


def list_setlists(traces: TracesDB) -> list[dict]:
    """Every setlist in the traces, newest first: its ID, name, set, size, status, and the
    preset each column ran. A name or note set in EH-1 (eval_setlists) wins over the one
    DH-1 labelled the runs with. Needs the EH-1 tables (`create_tables`)."""
    rows = traces.execute("""
        WITH s AS (
            SELECT labels.setlist_run::VARCHAR AS id,
                   any_value(labels.setlist::VARCHAR) AS question_set,
                   max(labels.setlist_revision::INTEGER) AS revision,
                   any_value(labels.setlist_name::VARCHAR) AS label_name,
                   any_value(labels.setlist_note::VARCHAR) AS label_note,
                   max(labels.setlist_concurrency::INTEGER) AS concurrency,
                   count(*) AS runs,
                   count(*) FILTER (WHERE status = 'completed') AS completed,
                   count(*) FILTER (WHERE status = 'cancelled') AS cancelled,
                   count(DISTINCT labels."column"::VARCHAR) AS columns,
                   max(labels."repeat"::INTEGER) AS repeats,
                   sum(estimated_cost_usd) AS cost_usd,
                   min(started_at) AS started_at,
                   max(ended_at) AS ended_at
            FROM agent_runs
            WHERE labels.setlist_run IS NOT NULL
            GROUP BY ALL
        )
        SELECT s.id, s.question_set, s.revision, coalesce(n.name, s.label_name), coalesce(n.note, s.label_note),
               s.concurrency, s.runs, s.completed, s.cancelled, s.columns, s.repeats, s.cost_usd,
               s.started_at, s.ended_at
        FROM s LEFT JOIN eval_setlists n ON n.setlist_run = s.id
        ORDER BY s.started_at DESC
    """).fetchall()
    keys = ["id", "question_set", "revision", "name", "note", "concurrency", "runs", "completed", "cancelled",
            "columns", "repeats", "cost_usd", "started_at", "ended_at"]
    presets: dict[str, list[dict]] = {}
    for setlist, column, preset, config in traces.execute("""
        SELECT labels.setlist_run::VARCHAR, labels."column"::VARCHAR,
               any_value(preset::JSON::VARCHAR), any_value(config::JSON::VARCHAR)
        FROM agent_runs
        WHERE labels.setlist_run IS NOT NULL
        GROUP BY ALL
        ORDER BY 1, 2
    """).fetchall():
        preset = json.loads(preset) if preset else {}
        presets.setdefault(setlist, []).append({
            "column": column, "name": preset.get("name"), "modified": bool(preset.get("modified")),
            "config": json.loads(config) if config else {},
        })
    out = []
    for row in rows:
        item = dict(zip(keys, row))
        for k in ("started_at", "ended_at"):
            item[k] = item[k].isoformat() if item[k] is not None else None
        item["presets"] = presets.get(item["id"], [])
        out.append(item)
    return out


def grading_status(traces: TracesDB, keys: dict[str, str]) -> dict[str, dict]:
    """For each setlist in `keys` (setlist ID -> the grader key it would be graded with now):
    how many of its runs are graded with that key, their average total out of 6 (your
    overrides counted), and how many other grader settings have graded it."""
    rows = traces.execute("""
        SELECT setlist_run, grader_key, count(*),
               avg(coalesce(override.correct::INTEGER, correct) + coalesce(override.careful::INTEGER, careful) + efficient)
        FROM eval_grades
        GROUP BY ALL
    """).fetchall()
    out = {sid: {"graded": 0, "total": None, "other_settings": 0} for sid in keys}
    for setlist, key, graded, total in rows:
        if setlist not in out:
            continue
        if key == keys[setlist]:
            out[setlist].update(graded=graded, total=total)
        else:
            out[setlist]["other_settings"] += 1
    return out


# --- eval_setlists: EH-1's names for setlists ---

EVAL_SETLISTS = {
    "setlist_run": "VARCHAR",
    # Your words, exactly as typed (whitespace tidied). NULL: use the name or note DH-1
    # labelled the runs with, if any.
    "name": "VARCHAR",
    "note": "VARCHAR",
    "renamed_at": "TIMESTAMP",
}
MAX_NAME, MAX_NOTE = 80, 200


def clean_text(value, limit: int) -> str | None:
    """Whitespace tidied and cut to `limit`; blank is None (as DH-1's cleanName)."""
    text = " ".join(str(value or "").split())[:limit]
    return text or None


def setlist_exists(traces: TracesDB, setlist_run: str) -> bool:
    return bool(traces.execute(
        f"SELECT count(*) FROM agent_runs WHERE labels.setlist_run::VARCHAR = {lit(setlist_run)}").fetchone()[0])


def rename_setlist(traces: TracesDB, setlist_run: str, name: str | None, note: str | None) -> None:
    """Give a setlist your name and note. None for both drops EH-1's row, so DH-1's labels
    (or the suggested title) show again."""
    traces.execute(f"DELETE FROM eval_setlists WHERE setlist_run = {lit(setlist_run)}")
    if name is None and note is None:
        return
    traces.execute(f"INSERT INTO eval_setlists VALUES ({lit(setlist_run)}, {lit(name)}, {lit(note)}, now()::TIMESTAMP)")


# --- eval_grades: one row per run per grader setting ---

EVAL_GRADES = {
    "run_id": "VARCHAR",
    # A hash of the grader settings and the question set's id and revision: grades from
    # different settings sit side by side, and a regrade replaces only its own rows.
    "grader_key": "VARCHAR",
    "graded_at": "TIMESTAMP",
    "setlist_run": "VARCHAR",
    "question_set": "VARCHAR",
    "question_set_revision": "INTEGER",
    "question": "VARCHAR",
    # The grader settings: preset name and revision, judge model and effort, tolerance...
    "grader": "VARIANT",
    # 0-2 each. CORRECT and CAREFUL come from the judge (NULL without a check or when the
    # judge failed); EFFICIENT from the traces.
    "correct": "INTEGER",
    "careful": "INTEGER",
    "efficient": "INTEGER",
    "correct_reason": "VARCHAR",
    "careful_reason": "VARCHAR",
    "efficient_reason": "VARCHAR",
    "judge_unsure": "BOOLEAN",
    # [{name, agent, reference, check}]: the values the judge compared, check being
    # matches | differs | unverified (no reference gives the value). Grades from judge
    # version 1 have a matches flag instead.
    "judge_values": "VARIANT",
    # [{label, passed, detail}] and whether all passed (NULL when there are none).
    "rules": "VARIANT",
    "rules_passed": "BOOLEAN",
    # What the reference SQL returned: [{sql, rows, columns, preview, error}].
    "reference": "VARIANT",
    "judge_model": "VARCHAR",
    "judge_cost_usd": "DOUBLE",
    "judge_error": "VARCHAR",
    # A person's correction, kept next to the judge's scores: {correct, careful, why, at}.
    "override": "VARIANT",
}


def create_tables(traces: TracesDB) -> None:
    """Create EH-1's tables (eval_grades, eval_setlists) if they aren't there, on the traces
    server when connected to one."""
    tables = {"eval_grades": EVAL_GRADES, "eval_setlists": EVAL_SETLISTS}
    ddl = {name: f"CREATE TABLE IF NOT EXISTS {name} (" + ", ".join(f'"{c}" {t}' for c, t in columns.items()) + ")"
           for name, columns in tables.items()}
    if not traces.remote:
        for sql in ddl.values():
            traces.conn.execute(sql)
        return
    run = lambda sql: traces.conn.execute(  # noqa: E731
        "SELECT * FROM quack_query(?, ?, token = ?)", [traces.quack_uri, sql, traces._token]).fetchall()
    names = ", ".join(lit(n) for n in tables)
    existing = {r[0] for r in run(f"SELECT table_name FROM information_schema.tables WHERE table_name IN ({names})")}
    missing = [n for n in tables if n not in existing]
    if not missing:
        return
    for name in missing:
        run(ddl[name])
    # The attached catalog lists the server's tables as of attaching: attach again so
    # this connection sees the new tables.
    traces.conn.execute("USE memory")
    traces.conn.execute("DETACH traces")
    traces.conn.execute("ATTACH ? AS traces (TYPE quack, TOKEN ?)", [traces.quack_uri, traces._token])
    traces.conn.execute("USE traces")


def _variant_json(value) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def write_grade(traces: TracesDB, row: dict) -> None:
    """Replace this run's grade for this grader setting with `row`. A person's override
    on the old row is kept unless `row` sets its own: regrading never drops your call."""
    if row.get("override") is None:
        kept = current_override(traces, row["run_id"], row["grader_key"])
        if kept is not None:
            row = {**row, "override": kept}
    names = list(EVAL_GRADES)
    values = [_variant_json(row.get(n)) if EVAL_GRADES[n] == "VARIANT" else row.get(n) for n in names]
    placeholders = ", ".join("?::JSON::VARIANT" if EVAL_GRADES[n] == "VARIANT" else "?" for n in names)
    quoted = ", ".join(f'"{n}"' for n in names)
    where = f"run_id = {lit(row['run_id'])} AND grader_key = {lit(row['grader_key'])}"
    traces.execute(f"DELETE FROM eval_grades WHERE {where}")
    if not traces.remote:
        traces.conn.execute(f"INSERT INTO eval_grades ({quoted}) VALUES ({placeholders})", values)
        return
    # Quack doesn't forward parameters: bind into a local temp table, copy across.
    traces.conn.execute("CREATE TEMP TABLE IF NOT EXISTS stage_eval_grades AS FROM eval_grades LIMIT 0")
    try:
        traces.conn.execute(f"INSERT INTO temp.main.stage_eval_grades ({quoted}) VALUES ({placeholders})", values)
        traces.conn.execute(f"INSERT INTO eval_grades ({quoted}) SELECT {quoted} FROM temp.main.stage_eval_grades")
    finally:
        traces.conn.execute("DELETE FROM temp.main.stage_eval_grades")


def current_override(traces: TracesDB, run_id: str, grader_key: str) -> dict | None:
    rows = traces.execute(f"""
        SELECT override::JSON::VARCHAR FROM eval_grades
        WHERE run_id = {lit(run_id)} AND grader_key = {lit(grader_key)} AND override IS NOT NULL
    """).fetchall()
    return json.loads(rows[0][0]) if rows and rows[0][0] else None


def set_override(traces: TracesDB, run_id: str, grader_key: str, override: dict | None) -> bool:
    """Store (or, with None, clear) a person's CORRECT and CAREFUL for one graded run.
    Returns False if the run isn't graded with this setting."""
    where = f"run_id = {lit(run_id)} AND grader_key = {lit(grader_key)}"
    if not traces.execute(f"SELECT count(*) FROM eval_grades WHERE {where}").fetchone()[0]:
        return False
    value = "NULL" if override is None else f"{lit(json.dumps(override, default=str))}::JSON::VARIANT"
    traces.execute(f"UPDATE eval_grades SET override = {value} WHERE {where}")
    return True


def graded_run_ids(traces: TracesDB, setlist_run: str, grader_key: str) -> set[str]:
    rows = traces.execute(f"""
        SELECT run_id FROM eval_grades
        WHERE setlist_run = {lit(setlist_run)} AND grader_key = {lit(grader_key)}
    """).fetchall()
    return {r[0] for r in rows}


def read_grades(traces: TracesDB, setlist_run: str, grader_key: str) -> list[dict]:
    """A setlist's grades for one grader setting, with each run's column and repeat."""
    rows = traces.execute(f"""
        SELECT g.run_id, g.question, r.labels."column"::VARCHAR, r.labels."repeat"::INTEGER, r.status,
               g.correct, g.careful, g.efficient, g.correct_reason, g.careful_reason, g.efficient_reason,
               g.judge_unsure, g.judge_values::JSON::VARCHAR, g.rules::JSON::VARCHAR, g.rules_passed,
               g.judge_cost_usd, g.judge_error, g.override::JSON::VARCHAR, r.estimated_cost_usd
        FROM eval_grades g JOIN agent_runs r USING (run_id)
        WHERE g.setlist_run = {lit(setlist_run)} AND g.grader_key = {lit(grader_key)}
    """).fetchall()
    keys = ["run_id", "question", "column", "repeat", "status", "correct", "careful", "efficient",
            "correct_reason", "careful_reason", "efficient_reason", "judge_unsure", "judge_values", "rules",
            "rules_passed", "judge_cost_usd", "judge_error", "override", "run_cost_usd"]
    out = []
    for row in rows:
        item = dict(zip(keys, row))
        for k in ("judge_values", "rules", "override"):
            item[k] = json.loads(item[k]) if item[k] else None
        out.append(item)
    return out
