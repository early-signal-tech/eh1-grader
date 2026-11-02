"""The judge: a Claude call that scores CORRECT and CAREFUL (0-2 each) for one run.

It gets the question, the question's own criteria (reference, correct, careful, fail,
bonus, note), what the reference SQL returned on the run's data, and the agent's answer.
It answers with structured output (output_config.format, a JSON schema), not a forced
tool call: Sonnet 5.5 and Opus 5.5 reject forced tool_choice.
"""

import json
from dataclasses import dataclass

import anthropic
from duckdb_agent.usage import estimate_cost as agent_estimate_cost

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
DEFAULT_JUDGE_EFFORT = "low"
JUDGE_MAX_TOKENS = 8000
MAX_ANSWER_CHARS = 20_000

# USD per million tokens for judge models the agent's table doesn't price (it prices
# Haiku 5.5). From the Claude API model reference; estimates only.
JUDGE_PRICING = {
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
}

SCORE = {"type": "integer", "enum": [0, 1, 2]}

SCHEMA = {
    "type": "object",
    "properties": {
        "correct": SCORE,
        "correct_reason": {"type": "string"},
        "careful": SCORE,
        "careful_reason": {"type": "string"},
        "unsure": {"type": "boolean"},
        "values": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "agent": {"type": "string"},
                    "reference": {"type": "string"},
                    "matches": {"type": "boolean"},
                },
                "required": ["name", "agent", "reference", "matches"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["correct", "correct_reason", "careful", "careful_reason", "unsure", "values"],
    "additionalProperties": False,
}

SYSTEM = """You grade answers from a SQL analytics agent. The agent was asked a question about a \
database, explored it with read-only SQL, and wrote an answer. You score the answer on two scales.

CORRECT (0-2): are the answer's facts right?
  2 = the numbers and claims match the reference (within the tolerance below); any correct \
query counts, it needn't match the reference SQL.
  1 = right approach, minor errors or omissions.
  0 = wrong or invented numbers, or it answers a different question.

CAREFUL (0-2): does it state the assumptions and caveats the question needs?
  2 = states what it assumed and flags the data issues that matter here.
  1 = some of them.
  0 = overclaims, or misses an obvious issue.

How to use the question's notes:
- "reference" lists facts the answer should contain; the reference query results are the source \
of truth for numbers.
- "correct" and "careful" describe what earns a 2 on each scale.
- "fail": if the answer does this, score CORRECT 0 (and CAREFUL 0 if it's about care).
- "bonus" can lift CAREFUL to 2; it never changes CORRECT.
- "note" is context for you (e.g. a known gap); follow it.
- When the right answer is that the data can't answer the question, an answer that says so \
clearly is correct.

Be strict but fair. Set "unsure" when the answer is ambiguous enough that a careful person \
could score it either way; still give your best scores. In each reason, say in a sentence or two \
what decided the score. In "values", list the key numbers or facts you compared (agent vs \
reference); use an empty list when there's nothing to compare."""


@dataclass
class Verdict:
    correct: int | None
    careful: int | None
    correct_reason: str | None
    careful_reason: str | None
    unsure: bool | None
    values: list[dict] | None
    cost_usd: float | None
    error: str | None = None


def estimate_cost(model: str, usage) -> float | None:
    cost = agent_estimate_cost(model, usage)
    if cost is not None:
        return cost
    rates = JUDGE_PRICING.get(model)
    if rates is None:
        return None
    read = (getattr(usage, "cache_read_input_tokens", 0) or 0) * 0.1
    write = (getattr(usage, "cache_creation_input_tokens", 0) or 0) * 1.25
    prompt = (getattr(usage, "input_tokens", 0) or 0) + read + write
    return (prompt * rates["input"] + (getattr(usage, "output_tokens", 0) or 0) * rates["output"]) / 1_000_000


def build_prompt(question: dict, check: dict, references: list[dict], answer: str, tolerance: str) -> str:
    notes = {k: check[k] for k in ("reference", "correct", "careful", "fail", "bonus", "note") if check.get(k)}
    parts = [
        "<question>\n" + question["prompt"] + "\n</question>",
        "<criteria>\n" + json.dumps(notes, indent=2, ensure_ascii=False) + "\n</criteria>",
        f"<tolerance>Numbers within {tolerance} of the reference count as matching; rounding is fine.</tolerance>",
    ]
    for i, ref in enumerate(references, 1):
        body = ref["preview"] if ref["error"] is None else f"(the reference query failed: {ref['error']})"
        parts.append(f'<reference_query n="{i}">\n{ref["sql"]}\n</reference_query>\n'
                     f'<reference_result n="{i}" rows="{ref["rows"]}">\n{body}\n</reference_result>')
    clipped = answer if len(answer) <= MAX_ANSWER_CHARS else answer[:MAX_ANSWER_CHARS] + "\n… (cut)"
    parts.append("<agent_answer>\n" + (clipped or "(no answer)") + "\n</agent_answer>")
    return "\n\n".join(parts)


def judge(client: anthropic.Anthropic, question: dict, check: dict, references: list[dict], answer: str,
          model: str = DEFAULT_JUDGE_MODEL, effort: str = DEFAULT_JUDGE_EFFORT, tolerance: str = "±0.5%") -> Verdict:
    """Score one answer. API or parsing failures come back as a Verdict with `error` set."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=JUDGE_MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(question, check, references, answer, tolerance)}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.APIError as e:
        return Verdict(None, None, None, None, None, None, None, error=f"{type(e).__name__}: {e}")
    cost = estimate_cost(model, response.usage)
    if response.stop_reason in ("refusal", "max_tokens"):
        return Verdict(None, None, None, None, None, None, cost, error=f"judge stopped: {response.stop_reason}")
    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, ValueError) as e:
        return Verdict(None, None, None, None, None, None, cost, error=f"judge answer unreadable: {e}")
    return Verdict(
        correct=data["correct"], careful=data["careful"],
        correct_reason=data["correct_reason"], careful_reason=data["careful_reason"],
        unsure=data["unsure"], values=data["values"], cost_usd=cost,
    )
