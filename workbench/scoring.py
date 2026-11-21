"""Deterministic scoring of the three evidence channels (requirement 2).

* text:     overlap of turn terms with the sentiment trigger lexicon,
            with a full hit decomposition.
* acoustic: fixed-order accumulation over pitch stats and the configured
            energy-envelope quantiles (turns are always accumulated in
            ascending turn-id order by the caller).
* tool:     deterministic evaluation of the manifest formula via a
            whitelisted AST interpreter.  No runtime randomness, no
            network access.  Results carry the formula fingerprint and
            the dimension tag.
"""

from __future__ import annotations

import ast
import math

from .fingerprint import fingerprint


# ---------------------------------------------------------------------------
# Text evidence
# ---------------------------------------------------------------------------

def score_text(turn, sentiment_lexicon):
    """Score a turn by lexicon overlap.  Returns score + hit decomposition."""
    hits = []
    total = 0.0
    for position, term in enumerate(turn["terms"]):
        polarity = sentiment_lexicon.get(term)
        if polarity is not None:
            hits.append({"term": term, "position": position, "polarity": polarity})
            total += polarity
    score = total / len(turn["terms"])
    return {
        "channel": "text",
        "turn_id": turn["turn_id"],
        "score": score,
        "hits": hits,
        "term_count": len(turn["terms"]),
    }


# ---------------------------------------------------------------------------
# Acoustic evidence
# ---------------------------------------------------------------------------

def score_acoustic(acoustic, quantile_keys):
    """Score acoustic evidence with a fixed accumulation order.

    The accumulation order is fixed: pitch_mean, pitch_var, then the
    configured energy quantiles in ascending quantile-key order.
    """
    components = []
    components.append(("pitch_mean", acoustic["pitch_mean"] / 250.0))
    components.append(("pitch_var", acoustic["pitch_var"] / 2500.0))
    for key in sorted(quantile_keys):
        components.append((f"energy_q{key}", acoustic["energy_quantiles"][key]))
    # math.fsum: deterministic, order-sensitive accumulation.
    score = math.fsum(value for _, value in components) / len(components)
    return {
        "channel": "acoustic",
        "turn_id": acoustic["turn_id"],
        "score": score,
        "components": [{"name": name, "value": value} for name, value in components],
    }


# ---------------------------------------------------------------------------
# Tool evidence: safe deterministic formula evaluation
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                   ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                   ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}
_ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}
_ALLOWED_FUNCS = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
                  "log": math.log, "exp": math.exp, "tanh": math.tanh,
                  "floor": math.floor, "ceil": math.ceil}


def _eval_node(node, env):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left, env),
                                              _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _ALLOWED_FUNCS and not node.keywords:
        func = _ALLOWED_FUNCS[node.func.id]
        return func(*[_eval_node(arg, env) for arg in node.args])
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def evaluate_formula(formula, params):
    """Evaluate a manifest formula deterministically on the given params."""
    tree = ast.parse(formula, mode="eval")
    try:
        result = _eval_node(tree, dict(params))
    except (ValueError, ArithmeticError) as exc:
        raise ValueError(f"formula {formula!r} failed: {exc}") from exc
    if not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError(f"formula {formula!r} produced non-finite result")
    return float(result)


def call_tool(tool, params):
    """Execute a tool call deterministically.

    Returns the value plus the formula fingerprint and dimension tag so
    downstream stages can verify provenance and dimension compatibility.
    """
    value = evaluate_formula(tool["formula"], params)
    return {
        "channel": "tool",
        "tool": tool["name"],
        "value": value,
        "formula_fingerprint": fingerprint(tool["formula"]),
        "dimension": tool["dimension"],
    }
