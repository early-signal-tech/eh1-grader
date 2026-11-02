"""Trace rules: pass/fail checks on what a run did, read from its tool calls.

The rule shapes are the question set format's (duck-harness-app docs/QUESTION_SETS.md):
    {"error": "read_only_violation", "expect": "present" | "absent"}
    {"tool": "list_tables", "min": 1, "max": 2}
    {"before": "list_tables", "after": "describe_table"}
    {"sql": "(?i)\\bsummarize\\b", "expect": "present" | "absent", "max": 1}
    {"iterations": {"max": 2}}
    {"queries": {"max": 3}}
Each may carry a "label", which is what the scorecard shows.
"""

import re

from .traces import Run


def rule_regex(pattern: str) -> re.Pattern:
    """Case-insensitive, as in the app ((?i) is allowed but not needed)."""
    return re.compile(pattern.removeprefix("(?i)"), re.IGNORECASE)


def describe(rule: dict) -> str:
    if rule.get("label"):
        return rule["label"]
    if "error" in rule:
        return f"{'has' if rule['expect'] == 'present' else 'no'} {rule['error']} errors"
    if "tool" in rule:
        lo, hi = rule.get("min"), rule.get("max")
        n = f"{lo}–{hi}×" if lo is not None and hi is not None else f"at least {lo}×" if lo is not None \
            else f"at most {hi}×" if hi is not None else ""
        return f"calls {rule['tool']}" + (f" {n}" if n else "")
    if "before" in rule:
        return f"{rule['before']} before {rule['after']}"
    if "sql" in rule:
        return f"{'a query matches' if rule['expect'] == 'present' else 'no query matches'} /{rule['sql']}/"
    if "iterations" in rule:
        return f"at most {rule['iterations']['max']} iterations"
    return f"at most {rule['queries']['max']} queries"


def check_rule(rule: dict, run: Run) -> tuple[bool, str]:
    """(passed, what was found)."""
    calls = run.tool_calls
    if "error" in rule:
        n = sum(1 for c in calls if not c.success and c.error_type == rule["error"])
        passed = n > 0 if rule["expect"] == "present" else n == 0
        return passed, f"{n} {rule['error']} error{'s' if n != 1 else ''}"
    if "tool" in rule:
        n = sum(1 for c in calls if c.tool == rule["tool"])
        passed = n >= rule.get("min", 0) and (rule.get("max") is None or n <= rule["max"])
        return passed, f"{rule['tool']} called {n}×"
    if "before" in rule:
        first = {}
        for i, c in enumerate(calls):
            first.setdefault(c.tool, i)
        b, a = first.get(rule["before"]), first.get(rule["after"])
        if a is None:
            return True, f"{rule['after']} wasn't called"
        if b is None:
            return False, f"{rule['after']} was called without {rule['before']}"
        return b < a, f"{rule['before']} {'came first' if b < a else 'came after'}"
    if "sql" in rule:
        pattern = rule_regex(rule["sql"])
        n = sum(1 for c in calls if c.sql and pattern.search(c.sql))
        passed = n > 0 if rule["expect"] == "present" else n == 0
        if rule.get("max") is not None and n > rule["max"]:
            passed = False
        return passed, f"{n} matching quer{'y' if n == 1 else 'ies'}"
    if "iterations" in rule:
        return run.iterations <= rule["iterations"]["max"], f"{run.iterations} iterations"
    if "queries" in rule:
        n = sum(1 for c in calls if c.tool == "query_database")
        return n <= rule["queries"]["max"], f"{n} queries"
    return False, "unknown rule"


def check_rules(rules: list[dict], run: Run) -> tuple[list[dict], bool | None]:
    """Each rule's result, and whether all passed (None when there are no rules)."""
    results = []
    for rule in rules or []:
        passed, detail = check_rule(rule, run)
        results.append({"label": describe(rule), "passed": passed, "detail": detail})
    return results, (all(r["passed"] for r in results) if results else None)
