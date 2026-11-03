"""EFFICIENT, 0-2, straight from the traces (no judge).

EVAL_QUESTIONS.md's scale: 2 = few iterations, no wasted or failed queries, no
truncated dumps; 1 = some waste; 0 = loops, many errors, or hits the limit.

Here that is:
- 0 if the run hit a limit (iterations, tokens, context) or used more than twice the
  iteration budget.
- Otherwise count the waste: failed tool calls, repeated identical queries, and every
  truncated result after the first. 2 with no waste and within budget; 1 with at most
  two pieces of waste and at most twice the budget; else 0.
"""

import re

from .traces import Run

LIMIT_STATUSES = {"max_iterations", "max_tokens", "context_window_exceeded"}

DEFAULT_ITERATION_BUDGET = 8


def _normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def efficiency(run: Run, iteration_budget: int = DEFAULT_ITERATION_BUDGET) -> tuple[int, str]:
    """(score, reason)."""
    if run.status in LIMIT_STATUSES:
        return 0, f"hit a limit ({run.status.replace('_', ' ')})"
    failed = sum(1 for c in run.tool_calls if not c.success)
    seen, repeated = set(), 0
    for c in run.tool_calls:
        if c.tool == "query_database" and c.sql:
            key = _normalize(c.sql)
            repeated += key in seen
            seen.add(key)
    truncated = max(0, sum(1 for c in run.tool_calls if c.truncated) - 1)
    waste = failed + repeated + truncated
    parts = [f"{run.iterations} iterations"]
    if failed:
        parts.append(f"{failed} failed")
    if repeated:
        parts.append(f"{repeated} repeated")
    if truncated:
        parts.append(f"{truncated} extra truncated")
    reason = " · ".join(parts)
    if run.iterations > 2 * iteration_budget:
        return 0, reason + f" (budget {iteration_budget})"
    if waste == 0 and run.iterations <= iteration_budget:
        return 2, reason
    if waste <= 2:
        return 1, reason
    return 0, reason
