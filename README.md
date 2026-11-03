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

## Setlist names

Rename a setlist in EH-1 and your name and note go in `eval_setlists` (one row per
setlist), never into the agent's run labels. They win over the name DH-1 labelled the
runs with. Clear them and DH-1's name, or the suggested title, shows again.

## One run (RUN mode)

EH-1 can grade a single run, against a question or a **quick check** you write for it:
- **A question:** the one its setlist labels name, the same prompt in a set, or one you
  pick. This uses the same grader key as grading the setlist, so the scorecard sees it.
- **A quick check:** CORRECT IS, plus optional reference SQL and CAREFUL MEANS. Its grades
  have `question_set = 'quick'` and keep the check in `eval_grades.quick_check`. A
  changed check is a new grade beside the old one.

Reference SQL runs on the run's own data. That's `agent_runs.data_source` when the agent
recorded it (newer runs). Otherwise it's the database path the run reported, and runs on
in-memory data from before then can't run it.

## Overrides

When you disagree with the judge, set your own CORRECT and CAREFUL (0–2) and a reason.
It's stored next to the judge's scores in `eval_grades.override` (`{correct, careful, why,
at}`), never replacing them. Scorecards use yours where it's set. Regrading a run keeps
your override. EFFICIENT comes from the traces, so it isn't overridden.

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
  traces.py      reading runs and tool calls; the eval_grades and eval_setlists tables
  rules.py       trace rules
  efficiency.py  EFFICIENT
  reference.py   reference SQL on the run's data (read-only, cached)
  judge.py       CORRECT and CAREFUL (structured output)
  grade.py       grading a setlist: what to grade, in parallel, written as it goes
  serve.py       the app protocol (JSON Lines)
  cli.py         eh1-grade
```
