# eh1-grader

The engine behind **EH-1**, the Duck Harness evaluator plugin. It grades runs of the
[DuckDB analytics agent](https://github.com/early-signal-tech/duckdb-analytics-agent)
from their traces, the way the agent's `EVAL_QUESTIONS.md` does:

| Score | 0–2 | Comes from |
|---|---|---|
| **CORRECT** | do the answer's facts match the reference? | a judge model, given the reference SQL's results on the run's own data |
| **CAREFUL** | does it state the assumptions and caveats the question needs? | the judge, given the question's own criteria |
| **EFFICIENT** | few iterations, no wasted or failed queries, no truncated dumps | the traces (no judge) |

Plus **trace rules**: pass/fail checks on what a run did (no `read_only_violation`,
called `list_tables` first, …), also straight from the traces.

The questions, their criteria, reference SQL, and rules come from a **question set**
(format: `duck-harness-app/docs/QUESTION_SETS.md`). DH-1's Setlist runs a set; EH-1
grades that setlist. The UI is a device in
[duck-harness-app](https://github.com/early-signal-tech/duck-harness-app).

## Where grades go

`eval_grades` in the agent's traces database, next to `agent_runs`: one row per run per
grader setting (`grader_key`, a hash of the judge model, effort, tolerance, iteration
budget, and the question set's id and revision). Grading again with the same settings
skips what's done; different settings make new rows beside the old ones.

```sql
SELECT g.question, r.labels."column"::VARCHAR AS col, g.correct, g.careful, g.efficient,
       g.rules_passed, g.correct_reason
FROM eval_grades g JOIN agent_runs r USING (run_id)
WHERE g.setlist_run = 'sl-…' ORDER BY g.question;
```

## Use

Needs the agent checkout next to this one (it's a path dependency), and an
`ANTHROPIC_API_KEY` (in `.env`, or passed by the app) for the judge.

```bash
uv sync
uv run eh1-grade --list                                    # setlists in the traces
uv run eh1-grade --setlist sl-2026-11-02-1819-4vtm \
    --set ../duckdb-analytics-agent/evals/analytics-eval.json
uv run eh1-serve --traces-db ../duckdb-analytics-agent/agent_traces.duckdb   # for the app
```

The judge defaults to `claude-sonnet-5-5` at low effort. It answers with structured
output (a JSON schema), since Sonnet 5.5 and Opus 5.5 reject a forced tool choice.

## How EFFICIENT is scored

- **0** if the run hit a limit, or took more than twice the iteration budget (default 8).
- Otherwise the **waste** is counted: failed tool calls, repeated identical queries, and
  truncated results after the first.
- **2:** no waste, and within budget.
- **1:** at most two pieces of waste.
- **0:** more waste than that.

## Code

```
eh1_grader/
  traces.py      reading runs and tool calls; the eval_grades table
  rules.py       trace rules
  efficiency.py  EFFICIENT
  reference.py   reference SQL on the run's data (read-only, cached)
  judge.py       CORRECT and CAREFUL (structured output)
  grade.py       grading a setlist: what to grade, in parallel, written as it goes
  serve.py       the app protocol (JSON Lines)
  cli.py         eh1-grade
```
