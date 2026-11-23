"""File loading and stream-level strong validation."""
from __future__ import annotations

from .codec import decode_record
from .errors import SpecError
from .models import NormalizedAlert
from .rules import RuleSpec, load_rule_spec
from .topology import DependencyGraph
from .util import read_json, read_ndjson


def load_graph(path) -> DependencyGraph:
    return DependencyGraph(read_json(path))


def load_rules(path, graph: DependencyGraph) -> RuleSpec:
    return load_rule_spec(read_json(path), set(graph.services))


def load_stream(path, graph: DependencyGraph, rules: RuleSpec
                ) -> list[NormalizedAlert]:
    """Decode + normalize every NDJSON row and enforce stream invariants.

    Record-level violations raise E_FIELD (see :mod:`alert_pipeline.codec`);
    cross-record ordering violations raise E_SPEC at the first offending row.
    """
    known = set(graph.services)
    alerts: list[NormalizedAlert] = []
    prev_ts = -1
    for lineno, text in read_ndjson(path):
        alert = decode_record(text, lineno, rules.default_silence_ms, known)
        if alert.timestamp_ms < prev_ts:
            raise SpecError(
                "alert timestamps must be monotonically non-decreasing",
                lineno, "timestamp_ms",
                f"{alert.timestamp_ms} < previous {prev_ts}")
        prev_ts = alert.timestamp_ms
        alerts.append(alert)
    return alerts
