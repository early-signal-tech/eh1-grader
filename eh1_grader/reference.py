"""Running a question's reference SQL on the data the run used.

The question set says what data a question runs on; the reference SQL runs there,
attached read-only (and checked to be a single read, like the agent's own queries), so
a question set can't change anything. Data is opened once per source and results are
cached per (source, query), so grading 3 repeats × 4 presets runs each query once.
"""

import json
import threading

import duckdb
from duckdb_agent.sources import IN_MEMORY, connect, use_statement
from duckdb_agent.sql import check_read_only_sql

PREVIEW_ROWS = 30
PREVIEW_CHARS = 6000


def source_key(source: dict) -> str:
    return json.dumps(source, sort_keys=True)


def _open(source: dict) -> tuple[duckdb.DuckDBPyConnection, str | None, str | None]:
    """A connection to a question set source ({kind: tpch | db | files, ...})."""
    kind = source.get("kind")
    if kind == "tpch":
        return connect(IN_MEMORY, None, (), source["tpchScale"]), IN_MEMORY, None
    if kind == "db":
        schema = source.get("schema")
        return connect(source["dbPath"], schema), source["dbPath"], schema
    if kind == "files":
        return connect(IN_MEMORY, None, source["files"], None), IN_MEMORY, None
    raise ValueError(f"unknown source kind {kind!r}")


def _preview(columns: list[str], rows: list[tuple]) -> str:
    lines = [" | ".join(columns)]
    for row in rows[:PREVIEW_ROWS]:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    if len(rows) > PREVIEW_ROWS:
        lines.append(f"… {len(rows) - PREVIEW_ROWS} more rows")
    text = "\n".join(lines)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "\n… (cut)"


class References:
    """Opens each source once and remembers each query's result. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bases: dict[str, tuple] = {}
        self._results: dict[tuple[str, str], dict] = {}

    def run(self, source: dict, sql: str) -> dict:
        """{sql, rows, columns, preview, error}: what the reference query returned."""
        key = (source_key(source), sql)
        with self._lock:
            if key in self._results:
                return self._results[key]
            try:
                if source_key(source) not in self._bases:
                    self._bases[source_key(source)] = _open(source)
                base, db, schema = self._bases[source_key(source)]
                problem = check_read_only_sql(base, sql)
                if problem:
                    raise ValueError(problem)
                conn = base.cursor()
                use = use_statement(db, schema)
                if use:
                    conn.execute(use)
                cursor = conn.execute(sql)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                result = {"sql": sql, "rows": len(rows), "columns": columns, "preview": _preview(columns, rows), "error": None}
            except Exception as e:  # a bad reference shouldn't stop the grading
                result = {"sql": sql, "rows": None, "columns": None, "preview": None, "error": f"{type(e).__name__}: {e}"}
            self._results[key] = result
            return result

    def close(self) -> None:
        with self._lock:
            for base, _, _ in self._bases.values():
                base.close()
            self._bases.clear()
